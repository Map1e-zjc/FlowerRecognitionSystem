# ============================================================
# 前端（Vue3 + Vite）统一入口脚本
#
# 为什么需要它：本机有两处必须处理的环境问题，手工敲 npm 很容易踩坑 ——
#   1) registry.npmjs.org 不可达 → 必须走 npmmirror（已在 frontend/.npmrc 设置）
#   2) npm 默认缓存在 C 盘受限目录，会报 EPERM → 必须用**环境变量**重定向到
#      项目内 .cache/npm（因为项目路径含空格，写进 .npmrc 会被截断而静默失效）
#
# 用法：
#   .\scripts\frontend.ps1 -Action install     # 安装依赖
#   .\scripts\frontend.ps1 -Action dev        # 启动开发服务器（5173）
#   .\scripts\frontend.ps1 -Action build      # 类型检查 + 生产构建
#   .\scripts\frontend.ps1 -Action typecheck  # 只做类型检查
#   .\scripts\frontend.ps1 -Action preview    # 预览生产构建
# ============================================================
[CmdletBinding()]
param(
    [ValidateSet('install', 'dev', 'build', 'typecheck', 'preview')]
    [string]$Action = 'install',

    [switch]$NoTypeCheck
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontendDir = Join-Path $ProjectRoot 'frontend'

# --- 关键：npm 缓存必须用环境变量重定向（见 frontend/.npmrc 里的说明）---
$env:npm_config_cache = Join-Path $ProjectRoot '.cache\npm'
$env:npm_config_registry = 'https://registry.npmmirror.com/'
New-Item -ItemType Directory -Force -Path $env:npm_config_cache | Out-Null

Write-Host ''
Write-Host '=== 前端环境 ===' -ForegroundColor Cyan
Write-Host "项目根   : $ProjectRoot"
Write-Host "registry : $env:npm_config_registry"
Write-Host "cache    : $env:npm_config_cache"
Write-Host ''

Set-Location $FrontendDir

switch ($Action) {
    'install' {
        Write-Host '=== npm install ===' -ForegroundColor Cyan
        $sw = [Diagnostics.Stopwatch]::StartNew()
        npm install
        $code = $LASTEXITCODE
        $sw.Stop()
        Write-Host ("耗时 {0:N1} 分钟" -f $sw.Elapsed.TotalMinutes)
        exit $code
    }
    'dev' {
        Write-Host '=== npm run dev （http://127.0.0.1:5173）===' -ForegroundColor Cyan
        npm run dev
        exit $LASTEXITCODE
    }
    'build' {
        if ($NoTypeCheck) {
            Write-Host '=== npm run build:only （跳过类型检查）===' -ForegroundColor Cyan
            npm run build:only
        } else {
            Write-Host '=== npm run build （类型检查 + 构建）===' -ForegroundColor Cyan
            npm run build
        }
        exit $LASTEXITCODE
    }
    'typecheck' {
        Write-Host '=== npm run type-check ===' -ForegroundColor Cyan
        npm run type-check
        exit $LASTEXITCODE
    }
    'preview' {
        Write-Host '=== npm run preview ===' -ForegroundColor Cyan
        npm run preview
        exit $LASTEXITCODE
    }
}
