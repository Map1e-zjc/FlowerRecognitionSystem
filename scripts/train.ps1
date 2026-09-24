# ============================================================
# 一键训练 Flowers-102（M2 用）
#   用法：  .\scripts\train.ps1                          # 训练 resnet50
#           .\scripts\train.ps1 -Model vit_b16
#           .\scripts\train.ps1 -Model resnet50 -ExtraArgs '--epochs 2','--max-train-batches 3'
#           .\scripts\train.ps1 -Model vit_b16 -Resume
# ============================================================
[CmdletBinding()]
param(
    [ValidateSet('resnet50', 'efficientnet_b0', 'convnext_tiny', 'vit_b16')]
    [string]$Model = 'resnet50',

    [switch]$Resume,

    [switch]$Strong,

    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache\pip'
$env:TORCH_HOME    = Join-Path $ProjectRoot '.cache\torch'
$env:HF_HOME       = Join-Path $ProjectRoot '.cache\hf'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "未找到虚拟环境：$python"
    exit 1
}

$config = Join-Path $ProjectRoot "ml\configs\$Model.yaml"
if (-not (Test-Path $config)) {
    Write-Error "未找到配置：$config"
    exit 1
}

$cmdArgs = @('-m', 'ml.train', '--config', $config)
if ($Resume) { $cmdArgs += '--resume' }
if ($Strong) { $cmdArgs += '--strong' }
if ($ExtraArgs.Count -gt 0) { $cmdArgs += $ExtraArgs }

Write-Host "=== 训练 $Model ===" -ForegroundColor Cyan
Write-Host ("命令：python " + ($cmdArgs -join ' '))
Write-Host ''

$sw = [Diagnostics.Stopwatch]::StartNew()
& $python @cmdArgs
$code = $LASTEXITCODE
$sw.Stop()

Write-Host ''
if ($code -eq 0) {
    Write-Host ("=== 训练成功，耗时 {0:N1} 分钟 ===" -f $sw.Elapsed.TotalMinutes) -ForegroundColor Green
    $summary = Join-Path $ProjectRoot "ml\outputs\logs\${Model}_summary.json"
    if (Test-Path $summary) {
        Write-Host '--- 训练摘要 ---'
        Get-Content $summary -Raw -Encoding UTF8 | ConvertFrom-Json |
            Select-Object model, best_epoch, best_val_top1, best_val_top5, train_minutes, params_m |
            Format-List
    }
} else {
    Write-Host "=== 训练失败（exit=$code）===" -ForegroundColor Red
}
exit $code
