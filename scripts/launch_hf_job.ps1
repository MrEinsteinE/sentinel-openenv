#!/usr/bin/env pwsh
# scripts/launch_hf_job.ps1 - launch the Overseer trainer on HF Jobs (PowerShell).
#
# Recommended path for Windows users - invoking bash from PowerShell often
# loses the active venv's PATH on directories that contain spaces (such as
# this repo: "D:\OpenEnv Hackathon\sentinel"), which breaks `hf` lookup.
# Run this from the same PowerShell where you activated the venv.
#
# Prerequisites (one-time):
#   1) Activate the venv that has huggingface_hub>=0.27 installed.
#   2) `hf auth login` so HF_TOKEN is implicit via `-s HF_TOKEN`.
#   3) Set $env:GITHUB_TOKEN to a fine-grained PAT with contents:write on
#      MrEinsteinE/sentinel-openenv.
#
# Usage:
#     $env:GITHUB_TOKEN = "ghp_xxx"
#     ./scripts/launch_hf_job.ps1
#
# Override defaults via env vars before invoking, e.g.:
#     $env:FLAVOR = "a100-large"
#     $env:STEP200_MIN_REWARD = "0.90"
#     ./scripts/launch_hf_job.ps1

$ErrorActionPreference = "Stop"

# Force Python (which `hf` is built on) to emit UTF-8 to stdout, and tell
# PowerShell to read it as UTF-8. Without this, Windows defaults to cp1252
# and `hf auth whoami`'s check-mark glyph (U+2713) crashes the encoder with
# "'charmap' codec can't encode character '\u2713'".
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
    # Best effort - older PS hosts may not allow this; the env vars above are
    # what actually matters for the child process.
}

function Get-OrDefault {
    param([string]$Name, [string]$Default)
    $v = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrEmpty($v)) { return $Default } else { return $v }
}

$Flavor           = Get-OrDefault "FLAVOR"             "l4x1"
$Timeout          = Get-OrDefault "TIMEOUT"            "6h"
$SentinelUrl      = Get-OrDefault "SENTINEL_URL"       "https://elliot89-sentinel.hf.space"
$GitRepo          = Get-OrDefault "GIT_REPO"           "https://github.com/MrEinsteinE/sentinel-openenv"
$GitBranch        = Get-OrDefault "GIT_BRANCH"         "main"
$ModelName        = Get-OrDefault "MODEL_NAME"         "unsloth/Qwen3-1.7B"
$ModelRepo        = Get-OrDefault "MODEL_REPO"         "Elliot89/sentinel-overseer-qwen3-1.7b"
$Step100MinReward = Get-OrDefault "STEP100_MIN_REWARD" "0.05"
$Step200MinReward = Get-OrDefault "STEP200_MIN_REWARD" "0.85"

$HfCli = Get-Command hf -ErrorAction SilentlyContinue
if (-not $HfCli) {
    Write-Host "[launch] error: 'hf' CLI not found on PATH." -ForegroundColor Red
    Write-Host "  Install with: pip install -U 'huggingface_hub>=0.27'" -ForegroundColor Red
    Write-Host "  (Make sure the venv that has it is activated in this PowerShell.)" -ForegroundColor Red
    exit 1
}

# Confirm we're logged in and surface the username early. This catches the
# common 403 case where the token lacks job.write or the user is logged in
# under the wrong account.
$WhoamiOut = & hf auth whoami 2>&1
$WhoamiCode = $LASTEXITCODE
if ($WhoamiCode -ne 0) {
    Write-Host "[launch] error: not logged in to Hugging Face." -ForegroundColor Red
    Write-Host "  Run: hf auth login --token hf_xxx --add-to-git-credential" -ForegroundColor Red
    Write-Host "  (HF Jobs needs job.write - generate a Write-scope token at" -ForegroundColor Red
    Write-Host "   https://huggingface.co/settings/tokens)" -ForegroundColor Red
    exit 1
}

# Parse username from one of two known formats:
#   newer:  "[U+2713] Logged in\n  user: Elliot89"
#   older:  "Elliot89"
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
    # Fallback: pick the last non-empty, non-banner line.
    $cand = ($WhoamiText -split "`r?`n") |
        Where-Object { $_.Trim() -ne "" -and $_.Trim() -notmatch '^[\u2713\u2717xX]\s' }
    if ($cand) { $HfUser = ($cand | Select-Object -Last 1).Trim() }
}
if (-not $HfUser) { $HfUser = "<unknown>" }

$ExpectedNs = $ModelRepo.Split("/")[0]
if ($HfUser -ne $ExpectedNs) {
    Write-Host "[launch] warning: logged in as '$HfUser' but MODEL_REPO targets namespace '$ExpectedNs'." -ForegroundColor Yellow
    Write-Host "  The HF Job will run under '$HfUser'. Pushing the adapter to '$ModelRepo'" -ForegroundColor Yellow
    Write-Host "  will 403 unless that account has write access there." -ForegroundColor Yellow
}

if (-not $env:GITHUB_TOKEN) {
    Write-Host "[launch] error: GITHUB_TOKEN is not set in this shell." -ForegroundColor Red
    Write-Host "  Set it first, e.g.:" -ForegroundColor Red
    Write-Host "      `$env:GITHUB_TOKEN = 'ghp_xxx'" -ForegroundColor Red
    Write-Host "  The PAT must have contents:write on MrEinsteinE/sentinel-openenv." -ForegroundColor Red
    exit 1
}

$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ScriptPath = Join-Path $RepoRoot "training/grpo_hf_job.py"

if (-not (Test-Path -LiteralPath $ScriptPath)) {
    Write-Host "[launch] error: $ScriptPath not found. Run from repo root." -ForegroundColor Red
    exit 1
}

Write-Host "[launch] flavor=$Flavor timeout=$Timeout"
Write-Host "[launch] SENTINEL_URL=$SentinelUrl"
Write-Host "[launch] MODEL_REPO=$ModelRepo"
Write-Host "[launch] GIT_REPO=$GitRepo ($GitBranch)"
Write-Host "[launch] abort thresholds: step100<$Step100MinReward, step200<$Step200MinReward"
Write-Host "[launch] CLI=$($HfCli.Source)"
Write-Host "[launch] hf user=$HfUser"
Write-Host ""

# `hf jobs uv run` accepts repeated -s/-e flags. Build the arg array so each
# token is passed as a single argument (no shell-quoting fragility).
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
    "-e", "STEP100_MIN_REWARD=$Step100MinReward",
    "-e", "STEP200_MIN_REWARD=$Step200MinReward",
    "-e", "VLLM_USE_V1=0",
    $ScriptPath
)

& hf @argv
exit $LASTEXITCODE
