"""Regenerate training/grpo_colab.ipynb from scratch with the canonical
cell sequence. Run with `python scripts/regen_grpo_notebook.py`.

This is the authoritative source for what the notebook looks like — keep this
script and the notebook in sync. Whenever you need to change the notebook,
edit this script and re-run.
"""

from __future__ import annotations
import json
import pathlib

import nbformat

# ──────────────────────────────────────────────────────────────────────────
#                              CELL CONTENTS
# ──────────────────────────────────────────────────────────────────────────

CELL0_HEADER = """\
# SENTINEL Overseer — single-stage GRPO trainer (Colab fallback)

> **Primary trainer:** Hugging Face Jobs — `bash scripts/launch_hf_job.sh` (wraps `training/grpo_hf_job.py`).
> **This notebook:** the same pipeline, cell-by-cell, for judges who want to reproduce on a free Colab L4. **Identical pinned deps, identical reward model, identical artifacts.**

## What this notebook does

| Cell | Phase | What runs |
|:---:|---|---|
| 2  | 0   | Clone the GitHub repo into `/content/sentinel-openenv` |
| 3  | 0   | Install the **exact** pinned stack from `grpo_hf_job.py:PINS` (torch 2.7.0 / unsloth 2026.4.4 / TRL 0.21.0 / transformers 4.56.2 / vLLM 0.9.2 / peft 0.18.0). Verifies every import after. |
| 5  | 1   | Configure env vars + capture `t_start` for the run-summary wall clock (set-once, survives re-runs) |
| 7  | 2   | Wake the SENTINEL Space (`/health` poll, ~60 s cold start) + smoke-test one episode |
| 9  | 3   | Patch the `aimv2` registration collision, then load Qwen3-1.7B 4-bit + colocate vLLM (`fast_inference=True`) |
| 11 | 4   | **Zero-shot baseline F1** — the bar we have to clear |
| 13 | 5a  | Apply LoRA r=16 to q/k/v/o projections |
| 14 | 5b  | SFT warmup (1 epoch) on `training/sft_data/sft_warmup.jsonl` (idempotent — skips if already done) |
| 16 | 6   | **GRPO smoke (5 steps)** — gates the long run |
| 18 | 7   | **GRPO long run (400 steps)** — auto-abort at step 100/200 if reward stalls |
| 20 | 8   | Save best checkpoint + trained eval + `baseline_vs_trained.png` |
| 22 | 9   | Push LoRA to `Elliot89/sentinel-overseer-qwen3-1.7b` + git-commit artifacts |

**Scope note.** GRPO trains on **`action_screen` only** (`TASK_FILTER` in `grpo_hf_job.py`) — the warm-up tier — to fit in the Colab L4 wall-clock budget. The full 3-tier eval still runs in Cell 11 and Cell 20 (so the published F1 covers `action_screen` ⊕ `war_room` ⊕ `drift_ops`).

## Runtime budget

| Hardware | Total wall clock | Cells dominating |
|---|---:|---|
| Colab L4 (24 GB) | ≈ 4 h | Cell 18 (long GRPO ~3 h) + Cell 11 (zero-shot eval ~30 min) |
| Colab A100 (40 GB) | ≈ 1.5 h | Cell 18 dominates |
| HF Jobs `l4x1` | ≈ 4 h, no kernel disconnects | use `bash scripts/launch_hf_job.sh` instead of this notebook |

## Prerequisites (set in Colab → ⚙ Secrets, **before** Cell 3)

| Secret | Required for | Notes |
|---|---|---|
| `HF_TOKEN` | base-model download (Cell 9) + LoRA push (Cell 22) | needs `repo:write` |
| `GITHUB_TOKEN` | git-commit artifacts back to the repo (Cell 22, optional) | `contents:write` on `MrEinsteinE/sentinel-openenv` |
| `SENTINEL_URL` *(env var, optional)* | point Cell 7 at a different env | defaults to public Space |

## 🚦 Session restart recovery

Colab disconnects mid-run more often than not. Cells 9, 13, 14 have **idempotency guards**, and Cells 16, 18, 20, 22 are **resume-safe** — they re-import everything they need from `training.grpo_hf_job`, so a cold restart at any of those points works without backtracking. Use this table:

| State when kernel reconnected | Run from |
|---|---|
| Nothing loaded yet | Cell 2 → 3 → 5 → 7 → 9 → 11 → … |
| Cell 3 just upgraded `torch` / `numpy` / `unsloth` (you'll see warnings) | **Runtime > Restart session**, then run from Cell 2 — Cell 3 is a fast no-op the second time |
| Base model loaded, no LoRA | Cell 5 → 9 (no-op, skipped) → 13 → 14 → … |
| LoRA applied, mid-SFT | Cell 5 → 13 (no-op) → 14 (no-op if `outputs/sft_warmup_1ep` exists) → 16 → … |
| SFT done, smoke ran | Cell 5 → 16 (re-run) or 18 |
| Long run finished | Cell 5 → 20 → 22 |

`t_start` is captured **once** in Cell 5 and survives re-runs (the wall-clock in `run_summary.json` measures from your first Cell 5 execution, not the most recent). Avoid re-running Cell 9 *with* a PEFT-wrapped model in scope; the guard will print a warning and skip.

### Common errors → fixes

| Error you saw | Cell | Cause | Fix |
|---|---|---|---|
| `numpy.dtype size changed ... Expected 96 from C header, got 88 from PyObject` | 9 | Older Cell 3 had `numpy<2.0`, conflicting with Colab's pre-built torch wheels (built against numpy 2.x). **Fixed in Cell 3** (`numpy>=1.26`). | Restart session, run from Cell 2. |
| `Could not load libtorchcodec` / `undefined symbol: ...c10_cuda_check_implementation` | 9 | Colab's pre-installed `torchcodec` was compiled against torch 2.5; our torch 2.7 upgrade breaks the C10 CUDA ABI. `transformers.processing_utils` hard-imports `torchcodec`. **Fixed in Cell 3** — we uninstall `torchcodec` after the main install. | Restart session, run from Cell 2. |
| `Unsloth: Please install vLLM before enabling fast_inference!` | 9 | Cell 3's vLLM install silently failed. **Fixed in Cell 9** — auto-falls-back to non-vLLM mode. | No action — training continues, just slower. |
| `vllm 0.9.2 requires torchaudio==2.7.0, which is not installed` | 3 | Earlier we uninstalled `torchaudio` along with `torchcodec`, breaking vLLM's hard dep. **Fixed in Cell 3** — we now re-install matched `torchaudio==2.7.0` + `torchvision==0.22.0` after the uninstall. | No action — pip warning is gone. |
| `'OutStream' object has no attribute 'watch_fd_thread'` → `ImportError: critical package 'unsloth'` | 3 / 9 | `unsloth_zoo`'s tqdm/Inductor patcher reads `sys.stdout.watch_fd_thread`; older Colab `ipykernel` doesn't have that attribute. **Fixed by deferring unsloth's import to Cell 9** (Cell 3 only does `find_spec`) and adding a no-op `watch_fd_thread` shim at the top of Cell 9. | Restart session, run from Cell 2. The patch is idempotent. |
| `ValueError: 'aimv2' is already used by a Transformers config` | 9 | `transformers>=4.50` natively registers `aimv2`; `unsloth_zoo==2026.4.4` re-registers it without `exist_ok=True`. **Fixed in Cell 9** — one-shot monkeypatch on `_LazyConfigMapping.register`. | Restart session, run from Cell 2. The patch is idempotent. |
| `WARNING:torchao:Skipping import of cpp extensions due to incompatible torch version` | 3 | `torchao==0.17.0` (matches the production HF Jobs pin) wants torch ≥ 2.11; our torch is 2.7. cpp extensions are skipped, Python fallbacks run. | **No action — benign.** Quantization is slightly slower, but everything works. |
| `SENTINEL not reachable at https://elliot89-sentinel.hf.space` | 7 | Public Space cold-starting (~60–90 s). | Re-run Cell 7. The 18×5 s poll inside `warmup_sentinel` will catch it. |
| `step100_resft` after Cell 18 | 18 | Mean reward at step 100 < 0.05 — model can't learn from current SFT init. | Re-run Cell 14 with `epochs=3`, then re-run Cell 18. |
| `step200_sft_only` after Cell 18 | 18 | GRPO underperforms SFT — auto-abort kept the SFT-only checkpoint. | **Not a bug.** Skip to Cell 20; the SFT model is your final. |
"""

CELL1_HEADER = "## 0. Bootstrap — clone repo, install pinned deps"

CELL2_BOOTSTRAP = """\
import os, sys, subprocess, pathlib

REPO_URL = os.environ.get('GIT_REPO', 'https://github.com/MrEinsteinE/sentinel-openenv')
REPO_DIR = pathlib.Path(os.environ.get('SENTINEL_WORKDIR', '/content/sentinel-openenv'))
BRANCH   = os.environ.get('GIT_BRANCH', 'main')

if not (REPO_DIR / '.git').exists():
    if REPO_DIR.exists():
        subprocess.run(['rm', '-rf', str(REPO_DIR)], check=True)
    subprocess.run(['git', 'clone', '--depth=1', '--branch', BRANCH, REPO_URL, str(REPO_DIR)], check=True)

# Propagate so training.grpo_hf_job picks up the same path on import.
os.environ['SENTINEL_WORKDIR'] = str(REPO_DIR)
os.environ['SENTINEL_SKIP_BOOTSTRAP'] = '1'  # we already cloned manually above

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / 'training'))
print(f'✓ repo at {REPO_DIR}')
"""

CELL3_INSTALL = """\
# Pinned dependency set — these are the EXACT versions from
# training/grpo_hf_job.py:PINS. Same versions are used by HF Jobs (production)
# and this notebook (Colab fallback) so artifacts are bit-identical.
#
# Why we don't %%capture: install errors must be visible to judges.
#
# Why numpy is range-pinned (>=1.26) instead of exact:
#  • The HF Jobs runner (training/grpo_hf_job.py) pins numpy<2.0 because uv
#    creates a fresh venv where every C extension is built against numpy 1.x
#    consistently — no ABI clash.
#  • Colab is different: its pre-installed torch/scipy/sklearn wheels are
#    compiled against numpy 2.x. Forcing numpy<2.0 here would crash those
#    wheels at import:
#       "numpy.dtype size changed ... Expected 96 from C header, got 88 from PyObject"
#    Letting pip resolve numpy>=1.26 (lands on 2.x on Colab) keeps ABI consistent.

%pip install --quiet --upgrade pip

# Core stack — exact pins matching PINS in training/grpo_hf_job.py.
%pip install --quiet \\
  'torch==2.7.0' \\
  'unsloth==2026.4.4' 'unsloth_zoo==2026.4.4' \\
  'torchao==0.17.0' \\
  'trl==0.21.0' 'transformers==4.56.2' \\
  'peft==0.18.0' 'accelerate==1.13.0' \\
  'bitsandbytes==0.49.2' 'datasets>=2.18.0' \\
  'huggingface_hub>=0.27.0' 'matplotlib>=3.8.0' 'numpy>=1.26' \\
  'fastapi>=0.104.0' 'uvicorn[standard]>=0.24.0' 'pydantic>=2.6.0' \\
  'requests>=2.31.0' 'openai>=1.58.0'

# vLLM is ~3 GB and pulls custom CUDA wheels — install can take 5-10 min and
# occasionally fails on Colab. We DO NOT use shell-style `||` fallbacks — IPython's
# %pip magic does NOT interpret shell `||`, it passes the whole line to pip
# as args. We verify via importlib below and auto-fall-back if needed.
%pip install --quiet 'vllm==0.9.2'

# Defuse Colab's pre-installed torch* companion wheels — they were compiled
# against torch 2.5.x and break with C10 CUDA ABI errors after our torch==2.7
# upgrade above. The most common one to bite is torchcodec, which
# transformers.processing_utils transitively imports via video_utils.
# SENTINEL is text-only so we don't need any of these at runtime.
%pip uninstall --quiet -y torchcodec torchaudio torchvision 2>/dev/null

# But vllm 0.9.2 hard-declares torchaudio==2.7.0 in its install_requires, so
# pip's resolver complains if it's missing. Re-install MATCHED torchaudio +
# torchvision (compiled against torch 2.7.0, so no ABI mismatch). We don't
# re-install torchcodec — transformers handles its absence gracefully and
# we'd just hit the C10 ABI error all over again.
%pip install --quiet 'torchaudio==2.7.0' 'torchvision==0.22.0'

# ── Post-install verification (must run before Cell 9) ─────────────────────
# CRITICAL: do NOT actually import unsloth or vllm here. Their imports have
# side effects:
#   • unsloth touches sys.stdout.watch_fd_thread (older Colab ipykernel
#     doesn't have it → AttributeError before our Cell 9 monkeypatch fires).
#   • unsloth also patches transformers — we want this to happen BEFORE
#     transformers is imported (otherwise its "should be imported before
#     transformers" warning fires and some optimisations don't apply).
#   • vllm's first import is multi-second and we already auto-fallback in Cell 9.
# Just verify they're installed via find_spec; defer the actual import to
# Cell 9, which applies the necessary monkeypatches first.
import importlib, importlib.util, os

print('✓ deps installed; verifying critical imports …')

def _check(name, *, fallback_env=None, fallback_msg='', fatal=False, lite=False):
    \"\"\"Return True if `name` resolves. If lite=True, only check find_spec
    (no actual import). If fatal=True and missing, raise ImportError.\"\"\"
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f'  ✗ {name:14s} not installed')
        if fatal:
            raise ImportError(f'critical package {name!r} could not be found')
        if fallback_env:
            os.environ[fallback_env] = '0'
            print(f'    → set {fallback_env}=0; {fallback_msg}')
        return False
    if lite:
        print(f'  ✓ {name:14s} installed (deferred import to Cell 9)')
        return True
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, '__version__', '?')
        print(f'  ✓ {name:14s} {ver}')
        return True
    except Exception as e:
        print(f'  ✗ {name:14s} import raised: {type(e).__name__}: {str(e)[:100]}')
        if fatal:
            raise ImportError(f'critical package {name!r} could not be imported') from e
        if fallback_env:
            os.environ[fallback_env] = '0'
            print(f'    → set {fallback_env}=0; {fallback_msg}')
        return False

_check('numpy',        fatal=True)
_check('torch',        fatal=True)
# transformers / peft / trl are imported here — that's fine, they're side-effect-free.
_check('transformers', fatal=True)
_check('peft',         fatal=True)
_check('trl',          fatal=True)
# unsloth / vllm: lite check only — Cell 9 does the real import behind the patches.
_check('unsloth', fatal=True, lite=True)
_check('vllm', fallback_env='SENTINEL_USE_VLLM', lite=True,
              fallback_msg='Cell 9 will fall back to HF transformers (slower but works)')

print()
print('▶ If torch/numpy/unsloth were UPGRADED above and ANY row shows ✗:')
print('     Runtime > Restart session, then re-run from Cell 2.')
print('  (Cell 3 will be a fast no-op the second time around.)')
"""

CELL4_HEADER = "## 1. Configuration + auth"

CELL5_CONFIG = """\
import os, time

# CRITICAL: vLLM 0.9.x's v1 engine raises "AoT scheduling is required for full
# cuda graph" when unsloth_zoo constructs LLM(...) with default cudagraph
# settings. Falling back to the legacy v0 engine is the documented workaround.
# Must be set BEFORE anything imports vllm — vllm.envs reads it at import time.
os.environ.setdefault('VLLM_USE_V1', '0')

# Where the SENTINEL env lives + what we're training/pushing.
os.environ.setdefault('SENTINEL_URL', 'https://elliot89-sentinel.hf.space')
os.environ.setdefault('MODEL_NAME',   'unsloth/Qwen3-1.7B')
os.environ.setdefault('MODEL_REPO',   'Elliot89/sentinel-overseer-qwen3-1.7b')

# GRPO auto-abort thresholds (see TrackingCallback in grpo_hf_job.py).
# These match the production defaults; keep them or tune at your own risk.
os.environ.setdefault('STEP100_MIN_REWARD', '0.05')
os.environ.setdefault('STEP200_MIN_REWARD', '0.85')

# vLLM colocate toggle. Cell 3 may have already set this to '0' if the vLLM
# install failed; setdefault won't override that decision.
os.environ.setdefault('SENTINEL_USE_VLLM', '1')

# Pull HF / GitHub tokens from Colab Secrets when available (silent off-Colab).
try:
    from google.colab import userdata
    for k in ('HF_TOKEN', 'GITHUB_TOKEN'):
        try:
            v = userdata.get(k)
            if v:
                os.environ[k] = v
        except Exception:
            pass
except Exception:
    pass

if os.environ.get('HF_TOKEN'):
    from huggingface_hub import login
    login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
    print('✓ HF login OK')
else:
    print('⚠ HF_TOKEN not set — base model download will use anonymous mode;')
    print('  the LoRA push in Cell 22 will be skipped.')

# Capture wall-clock start ONCE per session. Cell 22 subtracts t_start from
# time.time() to write run_summary.json's wall_clock_s. If we reset t_start
# every time Cell 5 is re-run, a re-run during Cell 22 recovery would log
# minutes instead of hours. Use globals() to make this idempotent.
if 't_start' not in globals():
    t_start = time.time()
    print('✓ config loaded, timer armed (first Cell 5 run)')
else:
    elapsed_min = (time.time() - t_start) / 60
    print(f'✓ config loaded, timer preserved ({elapsed_min:.1f} min since first run)')
"""

CELL6_HEADER = "## 2. Wake up the SENTINEL Space + smoke-test the env"

CELL7_WARMUP = """\
from training.grpo_hf_job import warmup_sentinel, build_tool_env_cls, SENTINEL_URL

warmup_sentinel(SENTINEL_URL)

ToolEnv = build_tool_env_cls(SENTINEL_URL)
_env = ToolEnv()
obs = _env.reset(task_id='action_screen', seed=1)
print('first observation (truncated to 400 chars):')
print(obs[:400])
"""

CELL8_HEADER = "## 3. Load Qwen3-1.7B (4-bit QLoRA + vLLM colocate)"

CELL9_LOAD = """\
import sys, importlib.util

# ── Pre-import monkeypatch #1: defuse OutStream.watch_fd_thread AttributeError
# unsloth's import side-effects (specifically the tqdm/Inductor patcher in
# unsloth_zoo) read `sys.stdout.watch_fd_thread`. Newer ipykernel ships this
# attribute on `OutStream`; the older ipykernel that Colab boots with does
# NOT. The plain attribute access raises:
#   AttributeError: 'OutStream' object has no attribute 'watch_fd_thread'
# and unsloth's import then bombs with `ImportError: critical package 'unsloth'`.
# Add a no-op shim so the access succeeds. Restarting the kernel doesn't help —
# Colab respawns with the same ipykernel, so we patch every time.
class _NoopWatchFdThread:
    def start(self): pass
    def join(self, timeout=None): pass
    def is_alive(self): return False
    def __bool__(self): return False
for _stream in (sys.stdout, sys.stderr):
    if not hasattr(_stream, 'watch_fd_thread'):
        try:
            _stream.watch_fd_thread = _NoopWatchFdThread()
        except Exception:
            pass  # some streams (e.g. detached file objects) don't allow attr set
# Some unsloth code paths read this off the underlying class instead of the
# instance — patch the class too if we can find it.
try:
    _OutStreamCls = type(sys.stdout)
    if 'OutStream' in _OutStreamCls.__name__ and not hasattr(_OutStreamCls, 'watch_fd_thread'):
        _OutStreamCls.watch_fd_thread = _NoopWatchFdThread()
except Exception:
    pass
print('✓ patched OutStream.watch_fd_thread (no-op) — safe vs unsloth_zoo tqdm patch')

# ── Pre-import monkeypatch #2: defuse the `aimv2` registration crash ──────
# transformers >= 4.50 natively registers a config named "aimv2".
# unsloth_zoo 2026.4.4's temporary_patches still calls
#   CONFIG_MAPPING.register("aimv2", ...)
# WITHOUT exist_ok=True, which raises:
#   ValueError: 'aimv2' is already used by a Transformers config, ...
# Patch _LazyConfigMapping.register at the source so it always passes
# exist_ok=True. Must run BEFORE `from unsloth import FastLanguageModel`
# (unsloth's import chain triggers the registration eagerly).
try:
    import transformers.models.auto.configuration_auto as _ca
    if hasattr(_ca, '_LazyConfigMapping') and not getattr(
        _ca._LazyConfigMapping.register, '_sentinel_safe', False
    ):
        _orig_register = _ca._LazyConfigMapping.register
        def _safe_register(self, key, value, exist_ok=False):
            return _orig_register(self, key, value, exist_ok=True)
        _safe_register._sentinel_safe = True
        _ca._LazyConfigMapping.register = _safe_register
        print('✓ patched _LazyConfigMapping.register (exist_ok=True) — safe vs aimv2')
except Exception as _e:
    print(f'⚠ aimv2 monkeypatch skipped: {type(_e).__name__}: {_e}')

# Now safe to import unsloth — its `from unsloth_zoo.temporary_patches import …`
# chain will hit our two patches above instead of the original AttributeError /
# ValueError pair.
from unsloth import FastLanguageModel

# Final auto-fallback: if SENTINEL_USE_VLLM=1 but vllm is not actually
# importable (Cell 3 install failed, kernel doesn't have it, etc.), unsloth
# raises `ImportError: Please install vLLM before enabling fast_inference!`.
# Detect this and silently downgrade to non-colocate mode — training still
# works, just slower (HF transformers generation instead of vLLM).
use_vllm = os.environ.get('SENTINEL_USE_VLLM', '1') == '1'
if use_vllm and importlib.util.find_spec('vllm') is None:
    print('⚠ SENTINEL_USE_VLLM=1 but vllm is not installed — falling back to use_vllm=False.')
    print('  Training will still work, just slower (no colocate generation).')
    use_vllm = False
    os.environ['SENTINEL_USE_VLLM'] = '0'

# Idempotency guard: re-running this cell after Cell 13 has wrapped the model
# with PEFT would blow away the LoRA-wrapped model and silently de-train the
# run. Detect `peft_config` (the marker get_peft_model attaches) and skip.
_already_peft = (
    'model' in dir()
    and getattr(globals().get('model'), 'peft_config', None) is not None
)

if _already_peft:
    print('⚠ a PEFT-wrapped model is already in scope — skipping reload.')
    print('  → If you want to start over, Runtime > Restart session, then run from Cell 2.')
    print('  → Otherwise, jump to Cell 14 (SFT) or Cell 16 (smoke).')
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        os.environ['MODEL_NAME'],
        max_seq_length=4096,
        load_in_4bit=True,
        fast_inference=use_vllm,
    )
    print(f'✓ base Qwen3-1.7B loaded (4-bit, vllm={use_vllm})')
"""

CELL10_HEADER = "## 4. Zero-shot baseline (the F1 we must beat)"

CELL11_ZEROSHOT = """\
from training.grpo_hf_job import _import_project, run_local_eval

project = _import_project()
baseline_summary = run_local_eval(model, tokenizer, 'qwen3_1_7b_zeroshot', project)
baseline_f1 = baseline_summary['per_task_f1']
print('zero-shot per-tier F1:', {k: round(v['f1'], 3) for k, v in baseline_f1.items()})
"""

CELL12_HEADER = "## 5. Apply the LoRA adapter (rank 16) and run SFT warmup"

CELL13_LORA = """\
# Idempotency guard: get_peft_model can be called only ONCE on a base model.
# Calling it twice stacks adapters on adapters — the model would behave
# nondeterministically, and Cell 22's push_lora_to_hub would upload the wrong
# tensors. Skip if a PEFT wrapper is already in place.
if getattr(model, 'peft_config', None) is not None:
    print('⚠ LoRA already applied — skipping. (Idempotent re-run is safe.)')
    print('  → To re-apply with different r/alpha, Restart session and re-run from Cell 2.')
else:
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        use_gradient_checkpointing='unsloth',
        random_state=42,
    )
    print('✓ LoRA r=16 applied to q/k/v/o projections')
"""

CELL14_SFT = """\
from pathlib import Path
from training.grpo_hf_job import run_sft

sft_out   = Path('outputs/sft_warmup_1ep')
sft_state = sft_out / 'trainer_state.json'

# Idempotency: if trainer_state.json is in outputs/sft_warmup_1ep, SFT already
# ran in this session. Re-running would do another epoch on top of the
# already-SFT'd weights — usually NOT what you want. Skip with a clear hint.
if sft_state.exists():
    print(f'⚠ SFT already ran ({sft_state}). Skipping to avoid double-training.')
    print('  → Override: !rm -rf outputs/sft_warmup_1ep, then re-run this cell.')
    print('  → Otherwise continue to Cell 16 (smoke).')
else:
    run_sft(model, tokenizer, epochs=1, output_dir=str(sft_out))
    print('✓ SFT warmup complete (1 epoch)')
"""

CELL15_HEADER = "## 6. GRPO smoke test (5 steps — gates the long run)"

CELL16_SMOKE = """\
from pathlib import Path
from training.grpo_hf_job import (
    TrackingCallback, _build_grpo_trainer, make_grpo_dataset,
    PLOTS_DIR, CKPT_DIR, SMOKE_STEPS, _import_project,
)

# Restart-resilience: re-resolve `project` and `use_vllm` if a kernel restart
# dropped them. Cell 11 / Cell 9 normally bind these.
if 'project' not in globals():
    project = _import_project()
if 'use_vllm' not in globals():
    use_vllm = os.environ.get('SENTINEL_USE_VLLM', '1') == '1'

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

smoke_ds = make_grpo_dataset(n_samples=64)
smoke_cb = TrackingCallback(
    plots_dir=PLOTS_DIR,
    ckpt_dir=CKPT_DIR,
    model=model,
    plot_loss_fn=project['plot_loss'],
    plot_reward_fn=project['plot_reward'],
    is_smoke=True,
)
smoke_trainer = _build_grpo_trainer(
    model, tokenizer, smoke_ds, smoke_cb,
    output_dir='outputs/grpo_smoke', max_steps=SMOKE_STEPS, use_vllm=use_vllm,
)
smoke_trainer.train()

ok, msg = smoke_cb.smoke_pass()
print()
print('=' * 60)
print('  smoke gate:', '✓ PASS' if ok else '✗ FAIL')
print('=' * 60)
print(msg)

# Soft-fail policy: don't halt the kernel. The long run in Cell 18 will still
# produce a usable model in many cases, and grpo_hf_job.py auto-extends SFT to
# 2 then 3 epochs in a similar situation. Surface a clear recovery path
# instead of raising.
if not ok:
    print()
    print('→ Recovery if smoke FAILed:')
    print('   1) Re-run Cell 14 (SFT) with epochs=2, then re-run THIS cell.')
    print('   2) If still failing, re-run Cell 14 with epochs=3, then re-run THIS cell.')
    print('   3) If still failing, proceed to Cell 18 anyway — long run may recover,')
    print('      but the auto-abort at step 100 will likely fire ("step100_resft").')
"""

CELL17_HEADER = """\
## 7. GRPO long run (400 steps, plots + checkpoints every 25)

This trains the policy with binary GRPO reward against `graders.grade_overseer_decision`. Plots and checkpoints land in `training/plots/` and `training/checkpoints/step_<N>/` every 25 steps.

**Auto-abort is built in.** Looking at `long_cb.abort_reason` after this cell:

| `abort_reason`            | What it means                                                                                                | Recovery |
|---|---|---|
| `None`                    | Trained the full 400 steps. Best checkpoint = `long_cb.best_step`.                                            | Continue to Cell 20 |
| `'step100_resft'`         | Mean reward at step 100 < `STEP100_MIN_REWARD` (default 0.05). Model can't learn from current SFT init.       | Re-run Cell 14 with `epochs=3`, then re-run this cell. |
| `'step100_resft_recovered'`| Hit the step 100 abort once, but Cell 14 retry succeeded.                                                     | Continue to Cell 20 |
| `'step200_sft_only'`      | Mean reward at step 200 < `STEP200_MIN_REWARD` (default 0.85). GRPO underperforms SFT — keep the SFT model. | Skip to Cell 20; the SFT-only checkpoint is your final. |

The trainer **does not raise** on abort — `should_training_stop=True` is set on the control object, and the cell completes normally. Cell 18 prints `abort_reason` so you can branch the recovery."""

CELL18_LONG = """\
from training.grpo_hf_job import (
    GRPO_CONFIG, TrackingCallback, _build_grpo_trainer, make_grpo_dataset,
    PLOTS_DIR, CKPT_DIR, _import_project,
)

# Restart-resilience: same pattern as Cell 16.
if 'project' not in globals():
    project = _import_project()
if 'use_vllm' not in globals():
    use_vllm = os.environ.get('SENTINEL_USE_VLLM', '1') == '1'

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

long_ds = make_grpo_dataset(
    n_samples=GRPO_CONFIG['max_steps'] * GRPO_CONFIG['gradient_accumulation_steps']
)
long_cb = TrackingCallback(
    plots_dir=PLOTS_DIR,
    ckpt_dir=CKPT_DIR,
    model=model,
    plot_loss_fn=project['plot_loss'],
    plot_reward_fn=project['plot_reward'],
)
long_trainer = _build_grpo_trainer(
    model, tokenizer, long_ds, long_cb,
    output_dir='outputs/grpo_long',
    max_steps=GRPO_CONFIG['max_steps'],
    use_vllm=use_vllm,
)
long_trainer.train()
print()
print('abort_reason =', long_cb.abort_reason)
print('best step    =', long_cb.best_step, '@ reward', round(long_cb.best_reward, 3))
"""

CELL19_HEADER = "## 8. Save best checkpoint + trained eval + comparison plot"

CELL20_SAVE = """\
from training.grpo_hf_job import (
    EVAL_DIR, CKPT_DIR, PLOTS_DIR,
    _load_baselines, _import_project, run_local_eval,
)

if 'project' not in globals():
    project = _import_project()

final_dir = CKPT_DIR / 'qwen3-1.7b-sentinel-best'
final_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f'✓ saved adapter -> {final_dir}')

trained_summary = run_local_eval(model, tokenizer, 'trained_qwen3_1_7b_grpo', project)
f1_per_tier = trained_summary['per_task_f1']

baselines = _load_baselines(EVAL_DIR)

# baseline_f1 was computed in Cell 11; if a cold restart skipped Cell 11, fall
# back to whatever's on disk in eval_data/baseline_qwen3_1_7b_zeroshot.json.
if 'baseline_f1' not in globals():
    print('⚠ baseline_f1 not in scope (cold restart?) — using disk-cached zero-shot.')
    baseline_f1 = baselines.get('qwen3_1_7b_zeroshot', {})

baselines['qwen3_1_7b_zeroshot']      = baseline_f1
baselines['trained_qwen3_1_7b_grpo']  = f1_per_tier
project['plot_baseline_vs_trained'](
    baselines,
    trained_label='trained_qwen3_1_7b_grpo',
    out_path=str(PLOTS_DIR / 'baseline_vs_trained.png'),
    tier='action_screen',
)

from IPython.display import Image, display
for name in ('grpo_loss.png', 'grpo_reward.png', 'baseline_vs_trained.png'):
    p = PLOTS_DIR / name
    if p.exists():
        display(Image(filename=str(p)))

zs_f1 = baseline_f1.get('action_screen', {}).get('f1', float('nan'))
tr_f1 = f1_per_tier.get('action_screen', {}).get('f1', float('nan'))
print(f'\\nzero-shot action_screen F1: {zs_f1:.3f}')
print(f'trained   action_screen F1: {tr_f1:.3f}')
"""

CELL21_HEADER = """\
## 9. Push LoRA to HF + git push artifacts

Two things happen below — both are **best-effort** and skip silently if the relevant token is missing.

| Step | What it does | Required |
|---|---|---|
| `_write_summary(...)` | Writes `training/run_summary.json` with pinned versions, GRPO config, F1 per tier, abort path, **wall clock from `t_start`**, and the best-step index. | none — pure local write |
| `push_lora_to_hub(final_dir)` | Uploads `training/checkpoints/qwen3-1.7b-sentinel-best/` to the model repo `Elliot89/sentinel-overseer-qwen3-1.7b`. | `HF_TOKEN` with `repo:write` |
| `git_push_artifacts(...)` | Adds + commits + pushes `training/plots/`, `training/run_summary.json`, and `eval_data/baseline_*.json` back to the GitHub repo. Mirrors the eval JSONs to the model repo as a fallback if git push is rejected. | `GITHUB_TOKEN` with `contents:write` on `MrEinsteinE/sentinel-openenv` |

If you only have `HF_TOKEN` set, you'll get the LoRA push but no GitHub commit — that's fine; the public model repo carries the eval JSONs as backup under `eval/`."""

CELL22_PUSH = """\
import time
from training.grpo_hf_job import (
    _write_summary, push_lora_to_hub, git_push_artifacts, CKPT_DIR,
)

# Resume-safe: define final_dir locally — Cell 20 also sets this, but a cold
# Cell 22 (per the recovery table) needs it computed here.
final_dir = CKPT_DIR / 'qwen3-1.7b-sentinel-best'

# Wall clock measured from Cell 5's t_start. _write_summary expects a duration
# in seconds, NOT an epoch.
elapsed_s = time.time() - t_start

# long_cb may not be in scope after a cold restart — degrade gracefully.
_long_cb = globals().get('long_cb')
abort_reason = _long_cb.abort_reason if _long_cb is not None else None
best_step    = _long_cb.best_step    if _long_cb is not None else 0

_write_summary(
    f1_per_tier=f1_per_tier,
    baseline_f1=baseline_f1,
    abort_path=abort_reason,
    wall_clock_s=elapsed_s,
    best_step=best_step,
)

# Best-effort pushes; both skip silently if their token is missing.
push_lora_to_hub(final_dir)

action_screen_f1 = f1_per_tier.get('action_screen', {}).get('f1', 0.0)
git_push_artifacts(
    f'colab: training artifacts (action_screen F1 {action_screen_f1:.3f}, '
    f'wall {elapsed_s/60:.1f} min, abort={abort_reason or "none"})'
)

print()
print('=' * 60)
print(f'  ✓ DONE in {elapsed_s/60:.1f} min '
      f'(action_screen F1 = {action_screen_f1:.3f})')
print('=' * 60)
"""


# ──────────────────────────────────────────────────────────────────────────
#                              ASSEMBLY
# ──────────────────────────────────────────────────────────────────────────


def md(cell_id: str, source: str) -> dict:
    cell = nbformat.v4.new_markdown_cell(source)
    cell["id"] = cell_id
    return cell


def code(cell_id: str, source: str) -> dict:
    cell = nbformat.v4.new_code_cell(source)
    cell["id"] = cell_id
    return cell


def main() -> None:
    nb = nbformat.v4.new_notebook()

    cells = [
        md(  "intro-md",   CELL0_HEADER),
        md(  "cell0-md",   CELL1_HEADER),
        code("cell0-code", CELL2_BOOTSTRAP),
        code("cell0b-code",CELL3_INSTALL),
        md(  "cell1-md",   CELL4_HEADER),
        code("cell1-code", CELL5_CONFIG),
        md(  "cell2-md",   CELL6_HEADER),
        code("cell2-code", CELL7_WARMUP),
        md(  "cell3-md",   CELL8_HEADER),
        code("cell3-code", CELL9_LOAD),
        md(  "cell4-md",   CELL10_HEADER),
        code("cell4-code", CELL11_ZEROSHOT),
        md(  "cell5-md",   CELL12_HEADER),
        code("cell5a-code",CELL13_LORA),
        code("cell5b-code",CELL14_SFT),
        md(  "cell6-md",   CELL15_HEADER),
        code("cell6-code", CELL16_SMOKE),
        md(  "cell7-md",   CELL17_HEADER),
        code("cell7-code", CELL18_LONG),
        md(  "cell8-md",   CELL19_HEADER),
        code("cell8-code", CELL20_SAVE),
        md(  "cell9-md",   CELL21_HEADER),
        code("cell9-code", CELL22_PUSH),
    ]

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
        "colab": {
            "name": "SENTINEL Overseer — single-stage GRPO trainer",
            "provenance": [],
        },
    }

    nbformat.validate(nb)
    out = pathlib.Path(__file__).resolve().parent.parent / "training" / "grpo_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✓ wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
