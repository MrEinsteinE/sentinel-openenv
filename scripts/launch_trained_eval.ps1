#!/usr/bin/env pwsh
# scripts/launch_trained_eval.ps1 - run the TRAINED Qwen3-1.7B + LoRA eval as
# a one-shot HF Job. Skips SFT/GRPO entirely; downloads the trained adapter
# from MODEL_REPO (Hub), applies it, runs the held-out eval with per-turn
# capture, updates run_summary.json["f1_per_tier"], regenerates
# baseline_vs_trained.png, and pushes everything back to GitHub.
#
# Phase 3 update (combined eval): when the existing zero-shot baseline JSON
# is summary-only (no per-turn data), the job ALSO re-runs the zero-shot
# eval in verbose mode FIRST, before applying the LoRA. This gives both
# verbose JSONs needed by tools/find_before_after.py from a single launch.
# Set SENTINEL_SKIP_ZEROSHOT_RERUN=1 to force-skip the zero-shot pass if
# the verbose JSON is already on disk.
#
# Wall clock on l4x1 is:
#   ~60-90 min  trained-only (zero-shot already verbose on disk)
#   ~150-180 min combined (zero-shot rerun + trained eval)
#
# Prerequisites are identical to launch_hf_job.ps1:
#   1) Activate venv with huggingface_hub>=0.27.
#   2) `hf auth login` so HF_TOKEN flows via -s HF_TOKEN.
#   3) $env:GITHUB_TOKEN with contents:write on MrEinsteinE/sentinel-openenv.
#
# Usage:
#     $env:GITHUB_TOKEN = "ghp_xxx"
#     ./scripts/launch_trained_eval.ps1

$ErrorActionPreference = "Stop"

# `hf jobs uv run` is marked experimental in huggingface_hub and emits a
# UserWarning on stderr at import time. With $ErrorActionPreference = "Stop",
# PowerShell treats any stderr output from a native command as a terminating
# error and kills the script BEFORE the job is ever submitted. Silence the
# warning so the launcher actually reaches `hf @argv`.
$env:HF_HUB_DISABLE_EXPERIMENTAL_WARNING = "1"

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
}

function Get-OrDefault {
    param([string]$Name, [string]$Default)
    $v = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrEmpty($v)) { return $Default } else { return $v }
}

# 4h is generous for the combined zero-shot rerun + trained eval (~3h on l4x1).
# Override with $env:TIMEOUT='2h' for trained-only when the verbose zero-shot
# JSON is already on disk.
$Flavor      = Get-OrDefault "FLAVOR"        "l4x1"
$Timeout     = Get-OrDefault "TIMEOUT"       "4h"
$SentinelUrl = Get-OrDefault "SENTINEL_URL"  "https://elliot89-sentinel.hf.space"
$GitRepo     = Get-OrDefault "GIT_REPO"      "https://github.com/MrEinsteinE/sentinel-openenv"
$GitBranch   = Get-OrDefault "GIT_BRANCH"    "main"
$ModelName   = Get-OrDefault "MODEL_NAME"    "unsloth/Qwen3-1.7B"
$ModelRepo   = Get-OrDefault "MODEL_REPO"    "Elliot89/sentinel-overseer-qwen3-1.7b"

$HfCli = Get-Command hf -ErrorAction SilentlyContinue
if (-not $HfCli) {
    Write-Host "[launch] error: 'hf' CLI not found on PATH." -ForegroundColor Red
    Write-Host "  Install with: pip install -U 'huggingface_hub>=0.27'" -ForegroundColor Red
    exit 1
}

$WhoamiOut = & hf auth whoami 2>&1
$WhoamiCode = $LASTEXITCODE
if ($WhoamiCode -ne 0) {
    Write-Host "[launch] error: not logged in to Hugging Face." -ForegroundColor Red
    Write-Host "  Run: hf auth login --token hf_xxx --add-to-git-credential" -ForegroundColor Red
    exit 1
}

$WhoamiText = ($WhoamiOut | Out-String)
$HfUser = $null
foreach ($line in ($WhoamiText -split "`r?`n")) {
    $trimmed = $line.Trim()
    if ($trimmed -match '^user:\s*(\S+)') {
        $HfUser = $Matches[1]
        break
    }
}
if (-not $HfUser) {
    $cand = ($WhoamiText -split "`r?`n") |
        Where-Object { $_.Trim() -ne "" -and $_.Trim() -notmatch '^[\u2713\u2717xX]\s' }
    if ($cand) { $HfUser = ($cand | Select-Object -Last 1).Trim() }
}
if (-not $HfUser) { $HfUser = "<unknown>" }

if (-not $env:GITHUB_TOKEN) {
    Write-Host "[launch] error: GITHUB_TOKEN is not set in this shell." -ForegroundColor Red
    Write-Host "  The PAT must have contents:write on MrEinsteinE/sentinel-openenv." -ForegroundColor Red
    exit 1
}

$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ScriptPath = Join-Path $RepoRoot "training/grpo_hf_job.py"

if (-not (Test-Path -LiteralPath $ScriptPath)) {
    Write-Host "[launch] error: $ScriptPath not found." -ForegroundColor Red
    exit 1
}

Write-Host "[launch] mode=TRAINED-EVAL-ONLY"
Write-Host "[launch] flavor=$Flavor timeout=$Timeout"
Write-Host "[launch] SENTINEL_URL=$SentinelUrl"
Write-Host "[launch] MODEL_NAME=$ModelName  (will load LoRA from $ModelRepo)"
Write-Host "[launch] GIT_REPO=$GitRepo ($GitBranch)"
Write-Host "[launch] hf user=$HfUser"
Write-Host ""

$argv = @(
    "jobs", "uv", "run",
    "--flavor", $Flavor,
    "--timeout", $Timeout,
    "-s", "HF_TOKEN",
    "-s", "GITHUB_TOKEN=$env:GITHUB_TOKEN",
    "-e", "SENTINEL_URL=$SentinelUrl",
    "-e", "GIT_REPO=$GitRepo",
    "-e", "GIT_BRANCH=$GitBranch",
    "-e", "MODEL_NAME=$ModelName",
    "-e", "MODEL_REPO=$ModelRepo",
    "-e", "SENTINEL_TRAINED_EVAL_ONLY=1",
    "-e", "VLLM_USE_V1=0",
    $ScriptPath
)

& hf @argv
exit $LASTEXITCODE
