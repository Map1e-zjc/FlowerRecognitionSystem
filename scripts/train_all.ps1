# ============================================================
# M2：按顺序训练全部 4 个模型并导出指标汇总
#   用法：  .\scripts\train_all.ps1                       # 全部 4 个模型
#           .\scripts\train_all.ps1 -Models resnet50,vit_b16
#           .\scripts\train_all.ps1 -SkipTrain            # 只评估+汇总（权重已存在时）
#           .\scripts\train_all.ps1 -ContinueOnError      # 单个模型失败也继续
#
# 说明：
#   * 逐个训练（串行）—— 8 GB 显存无法并行两个训练任务；
#   * 每个模型训练完立即评估并落 *_eval.json，最后一个模型训完自动 export_metrics；
#   * 全程日志同时写入 ml\outputs\logs\ 与终端；总耗时约 3—4 小时（ADR-005）。
# ============================================================
[CmdletBinding()]
param(
    [string[]]$Models = @('resnet50', 'efficientnet_b0', 'convnext_tiny', 'vit_b16'),
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$ContinueOnError
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
if (-not (Test-Path $python)) { Write-Error "未找到虚拟环境：$python"; exit 1 }

$ckptDir = Join-Path $ProjectRoot 'ml\outputs\checkpoints'
$logDir  = Join-Path $ProjectRoot 'ml\outputs\logs'
New-Item -ItemType Directory -Force -Path $ckptDir, $logDir | Out-Null

$results = [System.Collections.Generic.List[object]]::new()
$grandSw = [Diagnostics.Stopwatch]::StartNew()
$failed = @()

foreach ($m in $Models) {
    Write-Host ''
    Write-Host ('#' * 70) -ForegroundColor DarkCyan
    Write-Host "# 模型：$m   (开始于 $(Get-Date -Format 'HH:mm:ss'))" -ForegroundColor Cyan
    Write-Host ('#' * 70) -ForegroundColor DarkCyan

    $config = Join-Path $ProjectRoot "ml\configs\$m.yaml"
    $ckpt   = Join-Path $ckptDir "${m}_best.pth"
    $sw = [Diagnostics.Stopwatch]::StartNew()

    # ---------- 训练 ----------
    if (-not $SkipTrain) {
        & $python -m ml.train --config $config
        if ($LASTEXITCODE -ne 0) {
            Write-Host "✗ $m 训练失败（exit=$LASTEXITCODE）" -ForegroundColor Red
            $failed += "$m(train)"
            if (-not $ContinueOnError) { break } else { continue }
        }
    }

    if (-not (Test-Path $ckpt)) {
        Write-Host "✗ $m 缺少权重 $ckpt，跳过评估" -ForegroundColor Yellow
        $failed += "$m(no-weights)"
        if (-not $ContinueOnError) { break } else { continue }
    }

    # ---------- 评估 ----------
    if (-not $SkipEval) {
        & $python -m ml.evaluate --weights $ckpt --split test
        if ($LASTEXITCODE -ne 0) {
            Write-Host "✗ $m 评估失败（exit=$LASTEXITCODE）" -ForegroundColor Red
            $failed += "$m(eval)"
            if (-not $ContinueOnError) { break } else { continue }
        }
    }

    $sw.Stop()
    $summaryPath = Join-Path $logDir "${m}_summary.json"
    $evalPath    = Join-Path $logDir "${m}_eval.json"
    $row = [pscustomobject]@{ Model = $m; Minutes = [math]::Round($sw.Elapsed.TotalMinutes, 1) }
    if (Test-Path $evalPath) {
        $ev = Get-Content $evalPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $row | Add-Member NoteProperty Top1 $ev.top1
        $row | Add-Member NoteProperty Top5 $ev.top5
        $row | Add-Member NoteProperty MacroF1 $ev.macro_f1
    }
    $results.Add($row)
    Write-Host ("✓ $m 完成，耗时 {0:N1} 分钟" -f $sw.Elapsed.TotalMinutes) -ForegroundColor Green
}

# ---------- 汇总 ----------
Write-Host ''
Write-Host '=== 汇总指标 → metrics.json ===' -ForegroundColor Cyan
& $python -m ml.export_metrics

$grandSw.Stop()
Write-Host ''
Write-Host '=== 全部完成 ===' -ForegroundColor Green
Write-Host ("总耗时：{0:N1} 分钟" -f $grandSw.Elapsed.TotalMinutes)
$results | Format-Table -AutoSize
if ($failed.Count -gt 0) {
    Write-Host ("失败项：{0}" -f ($failed -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host '下一步：python -m ml.export_metrics（已自动执行）；查看 ml\outputs\metrics.json'
