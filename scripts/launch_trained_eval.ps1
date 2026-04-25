#!/usr/bin/env pwsh
# scripts/launch_trained_eval.ps1 - run the TRAINED Qwen3-1.7B + LoRA eval as
# a one-shot HF Job. Skips SFT/GRPO entirely; downloads the trained adapter
# from MODEL_REPO (Hub), applies it, runs the held-out eval with per-turn
# capture, updates run_summary.json["f1_per_tier"], regenerates
# baseline_vs_trained.png, and pushes everything back to GitHub.
#
# Use this AFTER training has completed (LoRA pushed to Hub) when the
# eval_data/baseline_qwen3_1_7b_trained.json file is missing locally
# (e.g. the original run's artifact filter dropped it). Wall clock on l4x1
# is ~60-90 min (eval uses transformers.generate, not vLLM).
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

# 2h is generous for a single eval pass over EVAL_SEEDS_BY_TASK
# (~50 episodes x ~13 steps x ~200 tokens). On l4x1 with HF generate
# expect ~60-90 min, like the zero-shot pass.
$Flavor      = Get-OrDefault "FLAVOR"        "l4x1"
$Timeout     = Get-OrDefault "TIMEOUT"       "2h"
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
