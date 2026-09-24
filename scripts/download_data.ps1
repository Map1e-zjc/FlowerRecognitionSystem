# ============================================================
# 下载并校验 Oxford Flowers-102 数据集
#   用法：  .\scripts\download_data.ps1              # 默认 hf-mirror 源
#           .\scripts\download_data.ps1 -Source oxford
#           .\scripts\download_data.ps1 -VerifyOnly
# 注意：官方 ox.ac.uk 源在本网络不可达（DNS SERVFAIL），默认走 hf-mirror（ADR-011）
# ============================================================
[CmdletBinding()]
param(
    [ValidateSet('hf', 'oxford')]
    [string]$Source = 'hf',

    [string]$Root = 'data',

    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# 缓存重定向
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache\pip'
$env:TORCH_HOME    = Join-Path $ProjectRoot '.cache\torch'
$env:HF_HOME       = Join-Path $ProjectRoot '.cache\hf'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "未找到虚拟环境：$python`n请先执行：python -m venv .venv"
    exit 1
}

if ($VerifyOnly) {
    Write-Host '=== 仅校验数据集 ===' -ForegroundColor Cyan
    & $python -m ml.download_data --root $Root --verify
    exit $LASTEXITCODE
}

Write-Host "=== 下载 Flowers-102（源：$Source）===" -ForegroundColor Cyan
& $python -m ml.download_data --root $Root --source $Source
if ($LASTEXITCODE -ne 0) {
    Write-Error "下载或校验失败（exit=$LASTEXITCODE）"
    exit $LASTEXITCODE
}

Write-Host ''
Write-Host '=== 完成 ===' -ForegroundColor Green
$size = (Get-ChildItem (Join-Path $ProjectRoot $Root) -Recurse -File -ErrorAction SilentlyContinue |
         Measure-Object -Property Length -Sum).Sum
Write-Host ("数据目录占用：{0:N1} MB" -f ($size / 1MB))
Write-Host '提示：校验通过后可删除 _raw 子目录以回收约 3 GB 磁盘。'
