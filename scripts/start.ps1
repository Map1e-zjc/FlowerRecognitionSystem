# ============================================================
# AI 花卉识别系统 · 一键启动
#
# 默认模式（推荐，答辩现场用这个）：
#   后端 FastAPI 直接托管前端构建产物，**单端口** 8000 访问，
#   运行期不需要 Node / Vite / esbuild，前后端同源、断网也能演示。
#       .\scripts\start.ps1
#
# 开发模式（改前端代码时需要）：
#   额外启动 Vite 开发服务器（5173），带热更新与 /api 代理。
#       .\scripts\start.ps1 -Dev
#
# 常用参数：
#   -Port 8080      指定后端端口
#   -Build          启动前先构建前端（首次或前端改动后）
#   -NoBrowser      不自动打开浏览器
#   -Reload         后端开启热重载（开发用）
#
# 停止：在本窗口按 Ctrl+C
# ============================================================
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$Dev,
    [switch]$Build,
    [switch]$NoBrowser,
    [switch]$Reload
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# ---- 环境：缓存全部落 D 盘项目内（见 docs/05 §7.2）----
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache\pip'
$env:TORCH_HOME    = Join-Path $ProjectRoot '.cache\torch'
$env:HF_HOME       = Join-Path $ProjectRoot '.cache\hf'
$env:HF_ENDPOINT   = 'https://hf-mirror.com'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:npm_config_cache = Join-Path $ProjectRoot '.cache\npm'
$env:npm_config_registry = 'https://registry.npmmirror.com/'

$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

function Write-Step($text) { Write-Host "  $text" }
function Write-Head($text) { Write-Host ''; Write-Host "=== $text ===" -ForegroundColor Cyan }

Write-Host ''
Write-Host '############################################################' -ForegroundColor DarkCyan
Write-Host '#        AI 花卉识别系统 · 启动中                          #' -ForegroundColor DarkCyan
Write-Host '############################################################' -ForegroundColor DarkCyan

# ------------------------------------------------------------------
Write-Head '1. 检查运行环境'
if (-not (Test-Path $python)) {
    Write-Host "  ✗ 未找到虚拟环境：$python" -ForegroundColor Red
    Write-Host '    请先执行：python -m venv .venv  然后安装依赖（见 README）'
    exit 1
}
Write-Step "✓ Python 虚拟环境：$python"

$distIndex = Join-Path $ProjectRoot 'frontend\dist\index.html'
$needBuild = $Build -or (-not (Test-Path $distIndex))
if ($needBuild) {
    Write-Host '  ! 前端尚未构建，正在构建（首次约 20 秒）…' -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot 'frontend.ps1') -Action build
    if ($LASTEXITCODE -ne 0) {
        Write-Host '  ✗ 前端构建失败。若报 spawn EPERM，请以更宽权限重跑本脚本。' -ForegroundColor Red
        exit 1
    }
}
if (Test-Path $distIndex) {
    Write-Step '✓ 前端构建产物就绪（单端口托管模式）'
} else {
    Write-Host '  ! 无前端构建产物，仅提供后端 API 与 /docs' -ForegroundColor Yellow
}

$dbFile = Join-Path $ProjectRoot 'flowers.db'
if (-not (Test-Path $dbFile)) {
    Write-Host '  ! 数据库不存在，正在建表并导入种子数据…' -ForegroundColor Yellow
    Push-Location (Join-Path $ProjectRoot 'backend')
    & $python -m app.db.init_db
    Pop-Location
}
Write-Step '✓ 数据库就绪'

# ------------------------------------------------------------------
Write-Head '2. 启动后端服务'

$uvicornArgs = @('-m', 'uvicorn', '--app-dir', 'backend', 'app.main:app',
                 '--host', '127.0.0.1', '--port', "$Port", '--log-level', 'warning')
if ($Reload) { $uvicornArgs += '--reload' }

$backend = Start-Process -FilePath $python -ArgumentList $uvicornArgs -PassThru -NoNewWindow
Write-Step "✓ 后端已启动（PID $($backend.Id)）"

# 等待就绪
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 1000
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/healthz" -TimeoutSec 3 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    Write-Host '  ✗ 后端启动超时，请检查上面的报错信息' -ForegroundColor Red
    exit 1
}

# 读取健康状态用于展示（用 Python，避免 PowerShell 的中文编码问题）
$healthJson = & $python -c "import json,urllib.request;print(json.dumps(json.loads(urllib.request.urlopen('http://127.0.0.1:$Port/healthz').read())['data'],ensure_ascii=False))"
try {
    $h = $healthJson | ConvertFrom-Json
    Write-Step "✓ 健康检查通过"
    Write-Step "    数据库      : $($h.database)（花卉 $($h.flowers) 条 / 模型指标 $($h.model_metrics) 条）"
    Write-Step "    GPU         : $(if ($h.gpu_available) { $h.gpu_name } else { '不可用（将使用 CPU）' })"
    Write-Step "    识别模型    : $(if ($h.model_loaded) { $h.model_name } else { "未加载 — $($h.model_error)" })"
} catch {
    Write-Step '✓ 健康检查通过'
}

# ------------------------------------------------------------------
$frontendPort = 0
$vite = $null
if ($Dev) {
    Write-Head '3. 启动前端开发服务器（Vite）'
    if (-not (Test-Path (Join-Path $ProjectRoot 'frontend\node_modules'))) {
        Write-Host '  ! 前端依赖未安装，正在安装…' -ForegroundColor Yellow
        & (Join-Path $PSScriptRoot 'frontend.ps1') -Action install
    }
    $vite = Start-Process -FilePath 'npm.cmd' -ArgumentList @('run', 'dev') `
                         -WorkingDirectory (Join-Path $ProjectRoot 'frontend') -PassThru -NoNewWindow
    $frontendPort = 5173
    Start-Sleep -Seconds 6
    Write-Step "✓ Vite 开发服务器已启动（PID $($vite.Id)），端口 $frontendPort"
    Write-Host '  ! 开发模式依赖 esbuild 子进程，受限环境下可能报 spawn EPERM' -ForegroundColor Yellow
}

# ------------------------------------------------------------------
$url = if ($Dev) { "http://127.0.0.1:$frontendPort" } else { "http://127.0.0.1:$Port" }

Write-Host ''
Write-Host '############################################################' -ForegroundColor Green
Write-Host '#                    启动完成                              #' -ForegroundColor Green
Write-Host '############################################################' -ForegroundColor Green
Write-Host ''
Write-Host "  访问地址   : $url" -ForegroundColor Yellow
Write-Host "  接口文档   : http://127.0.0.1:$Port/docs" -ForegroundColor Gray
Write-Host "  健康检查   : http://127.0.0.1:$Port/healthz" -ForegroundColor Gray
if ($Dev) { Write-Host "  后端地址   : http://127.0.0.1:$Port" -ForegroundColor Gray }
Write-Host ''
Write-Host '  首次使用请先「注册」一个账号，登录后可上传图片识别。' -ForegroundColor Gray
Write-Host ''
Write-Host '  按 Ctrl+C 停止服务。' -ForegroundColor Gray
Write-Host ''

if (-not $NoBrowser) {
    try { Start-Process $url } catch { Write-Host "  （自动打开浏览器失败，请手动访问上面的地址）" -ForegroundColor Yellow }
}

# ------------------------------------------------------------------
# 前台等待，Ctrl+C 时统一收尾
try {
    while ($true) {
        Start-Sleep -Seconds 2
        if ($backend.HasExited) {
            Write-Host "`n  后端进程已退出（代码 $($backend.ExitCode)）" -ForegroundColor Red
            break
        }
    }
} finally {
    Write-Host "`n  正在停止服务…" -ForegroundColor Yellow
    foreach ($p in @($vite, $backend)) {
        if ($p -and -not $p.HasExited) {
            try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch { }
        }
    }
    Write-Host '  已停止。' -ForegroundColor Green
}
