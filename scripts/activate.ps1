# ============================================================
# 激活虚拟环境并重定向所有缓存到 D 盘（保护 C 盘）
#   用法：  . .\scripts\activate.ps1
#   注意：  必须用 ". " 点号调用才会影响当前终端
# ============================================================

$ProjectRoot = Split-Path -Parent $PSScriptRoot

# --- 缓存重定向（C 盘仅剩 ~14 GB，必须走 D 盘）---
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache\pip'
$env:TORCH_HOME    = Join-Path $ProjectRoot '.cache\torch'
$env:HF_HOME       = Join-Path $ProjectRoot '.cache\hf'

# --- HuggingFace 镜像（本机 huggingface.co 直连超时，必须走镜像才能下 timm 预训练权重）---
$env:HF_ENDPOINT = 'https://hf-mirror.com'

# --- 让 Python 输出 UTF-8，避免中文日志乱码 ---
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8       = '1'

# --- 激活 venv ---
$activate = Join-Path $ProjectRoot '.venv\Scripts\Activate.ps1'
if (-not (Test-Path $activate)) {
    Write-Error "未找到虚拟环境：$activate`n请先执行：python -m venv .venv"
    return
}
. $activate

Write-Host ''
Write-Host '=== AI 花卉识别系统 · 开发环境 ===' -ForegroundColor Cyan
Write-Host "项目根目录 : $ProjectRoot"
Write-Host "Python     : $((Get-Command python).Source)"
Write-Host "PIP_CACHE  : $env:PIP_CACHE_DIR"
Write-Host "TORCH_HOME : $env:TORCH_HOME"
Write-Host "HF_HOME    : $env:HF_HOME"
Write-Host ''
