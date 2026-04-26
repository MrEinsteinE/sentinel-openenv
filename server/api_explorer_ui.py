"""
server/api_explorer_ui.py — Interactive API Explorer Gradio tab.

Lets judges click "Try it" on every OpenEnv endpoint (`/health`, `/tasks`,
`/reset`, `/step`, `/state`, `/grader`) plus every SENTINEL/Live endpoint
(`/live/oversee`, `/live/stats`, `/live/health`) and see:

    1) the live JSON response from the actual FastAPI route
    2) the equivalent `curl` command pointed at the public Space URL
    3) a 🏆 Live Reward Scoreboard at the top that auto-refreshes from
       `/grader` after every call — cumulative responder reward,
       cumulative overseer reward, F1, confusion matrix.

The tab is purely populator-style (adds components to the current Gradio
context, no inner `gr.Blocks`) so it composes cleanly with the live tab
and the replay viewer in `server/live_ui.py:combine_with_live_tab()`.

Implementation note
-------------------
We make HTTP calls to `http://localhost:7860` (the same FastAPI process
the UI is mounted on). This is intentional: it exercises the *real*
request path so the curl command shown is provably equivalent to what
the UI does. The `SENTINEL_LOOPBACK` env var overrides if needed for
tests / external mounts.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import gradio as gr
import requests

_LOOPBACK = os.environ.get("SENTINEL_LOOPBACK", "http://localhost:7860")
_PUBLIC_BASE = "https://elliot89-sentinel.hf.space"


# ── Helpers ────────────────────────────────────────────────────────────────

def _format_response_md(resp: requests.Response | Exception, t_ms: int) -> str:
    if isinstance(resp, Exception):
        return (
            f"**❌ Request failed** · ⏱ {t_ms} ms\n\n"
            f"```\n{type(resp).__name__}: {resp}\n```"
        )
    try:
        body = resp.json()
        body_str = json.dumps(body, indent=2)
    except Exception:
        body_str = (resp.text or "(empty body)")[:4000]
    status_emoji = "✅" if resp.status_code < 400 else "❌"
    ctype = resp.headers.get("content-type", "?").split(";")[0]
    return (
        f"**{status_emoji} HTTP {resp.status_code}** · ⏱ **{t_ms} ms** · "
        f"`{ctype}`\n\n"
        f"```json\n{body_str}\n```"
    )


def _build_curl_md(method: str, path: str, body: dict[str, Any] | None) -> str:
    public_url = f"{_PUBLIC_BASE}{path}"
    if method == "GET":
        return f"```bash\ncurl {public_url}\n```"
    body_str = json.dumps(body or {}, indent=2)
    return (
        f"```bash\ncurl -X POST {public_url} \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d '{body_str}'\n```"
    )


def _http_get(path: str, timeout: float = 10.0) -> tuple[str, str, str]:
    """Returns (response_md, curl_md, scoreboard_md)."""
    t0 = time.time()
    try:
        r = requests.get(f"{_LOOPBACK}{path}", timeout=timeout)
        t_ms = int((time.time() - t0) * 1000)
        return (
            _format_response_md(r, t_ms),
            _build_curl_md("GET", path, None),
            _scoreboard_md(),
        )
    except Exception as e:
        t_ms = int((time.time() - t0) * 1000)
        return (
            _format_response_md(e, t_ms),
            _build_curl_md("GET", path, None),
            _scoreboard_md(),
        )


def _http_post(path: str, body: dict[str, Any], timeout: float = 30.0) -> tuple[str, str, str]:
    t0 = time.time()
    try:
        r = requests.post(f"{_LOOPBACK}{path}", json=body, timeout=timeout)
        t_ms = int((time.time() - t0) * 1000)
        return (
            _format_response_md(r, t_ms),
            _build_curl_md("POST", path, body),
            _scoreboard_md(),
        )
    except Exception as e:
        t_ms = int((time.time() - t0) * 1000)
        return (
            _format_response_md(e, t_ms),
            _build_curl_md("POST", path, body),
            _scoreboard_md(),
        )


# ── Reward Scoreboard (the headline feature) ───────────────────────────────

def _scoreboard_md() -> str:
    """Pull /grader and format as a rich markdown scoreboard."""
    try:
        r = requests.get(f"{_LOOPBACK}/grader", timeout=5)
        if r.status_code == 400:
            return (
                "### 🏆 Live Reward Scoreboard *(auto-updates after every endpoint call)*\n\n"
                "*No active episode. Use the **▶️ Try /reset** card below "
                "(or the **Replay Viewer** tab) to start one — then watch this "
                "scoreboard fill with cumulative rewards as `/step` runs.*"
            )
        if r.status_code != 200:
            return f"### 🏆 Live Reward Scoreboard\n*scoreboard offline (HTTP {r.status_code})*"
        d = r.json()
    except Exception as e:
        return f"### 🏆 Live Reward Scoreboard\n*scoreboard offline: {e}*"

    m = d.get("overseer_metrics", {})
    conf = d.get("overseer_confusion", {}) or {}
    eid = (d.get("episode_id") or "—")[:14]
    done_emoji = "✅ done" if d.get("done") else "⏳ running"
    f1_val = float(m.get("f1", 0.0) or 0.0)
    f1_emoji = "🟢" if f1_val >= 0.85 else ("🟡" if f1_val >= 0.5 else "🔴")
    return (
        "### 🏆 Live Reward Scoreboard *(auto-refreshes after every endpoint call)*\n\n"
        f"| Episode | Task | Step | Status |\n"
        f"|---|---|:---:|:---:|\n"
        f"| `{eid}…` | `{d.get('task_id', '—')}` | "
        f"`{d.get('step_count', 0)}` | {done_emoji} |\n\n"
        f"| 🤖 Responder cum reward | 🛡️ Overseer cum reward | {f1_emoji} Overseer F1 | TP / FP / TN / FN |\n"
        f"|:---:|:---:|:---:|:---:|\n"
        f"| **`{d.get('responder_cumulative_reward', 0):+.3f}`** | "
        f"**`{d.get('overseer_cumulative_reward', 0):+.3f}`** | "
        f"**`{f1_val:.3f}`** | "
        f"`TP={conf.get('tp', 0)} · FP={conf.get('fp', 0)} · "
        f"TN={conf.get('tn', 0)} · FN={conf.get('fn', 0)}` |\n\n"
        f"*Precision = `{m.get('precision', 0):.3f}` · "
        f"Recall = `{m.get('recall', 0):.3f}` · "
        f"Drift events triggered = `{len(d.get('drift_events', []))}`*"
    )


# ── Endpoint catalog (rendered as a Markdown table) ────────────────────────

_ENDPOINT_CATALOG_MD = """
| Method | Path | Description |
|:---:|---|---|
| `GET`  | `/health`         | Server health check (`{"status":"ok","version":"0.1.0"}`) |
| `GET`  | `/api/info`       | Service descriptor (name, version, tasks, docs URL) |
| `GET`  | `/tasks`          | All 3 task tiers + canonical action schemas (responder + overseer) |
| `POST` | `/reset`          | Start a new episode — `{"task_id","seed","mode"}` |
| `POST` | `/step`           | Submit one action — `{"role","responder"\\|"overseer":{...}}` |
| `GET`  | `/state`          | Full current `EpisodeState` (turn phase, history, drift events) |
| `GET`  | `/grader`         | Per-episode F1, confusion matrix, **cumulative rewards** 🏆 |
| `POST` | `/live/oversee`   | **SENTINEL/Live** — real-time verdict, no scenario state |
| `GET`  | `/live/stats`     | **SENTINEL/Live** — lifetime counters since server start |
| `GET`  | `/live/health`    | **SENTINEL/Live** — feature health (trained backend present?) |
| `GET`  | `/docs`           | FastAPI Swagger UI (interactive OpenAPI explorer) |

> **Note on `/stop`:** the OpenEnv contract has no explicit stop/close endpoint — episodes terminate naturally when `/step` returns `done: true`. Call `/reset` again to start a fresh one. Concurrent sessions are supported (`SUPPORTS_CONCURRENT_SESSIONS=True`); per-session state is keyed by `episode_id`.
"""


# ── Try-it handlers (one per endpoint card) ────────────────────────────────

def _try_health() -> tuple[str, str, str]:
    return _http_get("/health")


def _try_api_info() -> tuple[str, str, str]:
    return _http_get("/api/info")


def _try_tasks() -> tuple[str, str, str]:
    return _http_get("/tasks")


def _try_state() -> tuple[str, str, str]:
    return _http_get("/state")


def _try_grader() -> tuple[str, str, str]:
    return _http_get("/grader")


def _try_reset(task_id: str, seed_str: str, mode: str) -> tuple[str, str, str]:
    body: dict[str, Any] = {"task_id": task_id, "mode": mode}
    if seed_str and seed_str.strip():
        try:
            body["seed"] = int(seed_str.strip())
        except ValueError:
            pass
    return _http_post("/reset", body)


def _try_step_responder(
    responder_role: str,
    action_type: str,
    service: str,
    runbook_action: str,
    reasoning: str,
) -> tuple[str, str, str]:
    params: dict[str, Any] = {}
    if service and service.strip():
        params["service"] = service.strip()
    if runbook_action and runbook_action.strip():
        params["runbook_action"] = runbook_action.strip()
    body = {
        "role": "responder",
        "responder": {
            "responder_role": responder_role,
            "action_type": action_type,
            "parameters": params,
            "reasoning": reasoning or "",
        },
    }
    return _http_post("/step", body)


def _try_step_overseer(
    decision: str,
    justification: str,
) -> tuple[str, str, str]:
    body = {
        "role": "overseer",
        "overseer": {
            "decision": decision,
            "justification": justification or "no rationale provided",
        },
    }
    return _http_post("/step", body)


def _try_live_oversee(
    action_description: str,
    proposed_tool: str,
    proposed_target: str,
    severity_hint: str,
    backend: str,
) -> tuple[str, str, str]:
    body: dict[str, Any] = {"action_description": action_description}
    if proposed_tool.strip():
        body["proposed_tool"] = proposed_tool.strip()
    if proposed_target.strip():
        body["proposed_target"] = proposed_target.strip()
    if severity_hint in {"low", "medium", "high", "critical"}:
        body["severity_hint"] = severity_hint
    if backend in {"heuristic", "trained"} and backend != "heuristic":
        body["backend"] = backend
    return _http_post("/live/oversee", body)


def _try_live_stats() -> tuple[str, str, str]:
    return _http_get("/live/stats")


def _try_live_health() -> tuple[str, str, str]:
    return _http_get("/live/health")


def _refresh_score_only() -> str:
    return _scoreboard_md()


# ── Card-builder helper ────────────────────────────────────────────────────

def _endpoint_card(
    title: str,
    method: str,
    path: str,
    summary: str,
    *,
    open: bool = False,
):
    """Open an Accordion + render the response/curl panes. Returns the
    (response_md, curl_md) components so the caller can wire its own
    inputs/Try button. Used for cards with custom inputs.
    """
    accordion = gr.Accordion(
        label=f"{method}  {path}  —  {title}",
        open=open,
    )
    return accordion


# ── The populator (canonical entry point) ──────────────────────────────────

_API_CSS = """
.api-scoreboard { padding: 14px 18px; border-radius: 12px;
                  background: linear-gradient(135deg, #0f172a, #1e3a8a);
                  color: #e2e8f0; border: 1px solid #1e293b;
                  margin-bottom: 14px; }
.api-scoreboard td, .api-scoreboard th { color: #e2e8f0 !important; }
.api-catalog { font-size: 0.95rem; }
"""


def _populate_api_explorer_ui() -> None:
    """Adds the full API Explorer UI into the current Gradio context."""

    gr.Markdown(
        "# 🔌 API Explorer — every OpenEnv endpoint, one click each\n\n"
        "This tab calls the **same FastAPI routes** that an external client "
        "(curl, Python `requests`, your agent framework) would hit. Every "
        "response shows the raw JSON **and** the equivalent `curl` command "
        f"pointed at the **public Space URL** ({_PUBLIC_BASE}) so you can "
        "replay it from your own terminal verbatim.\n"
    )

    score_md = gr.Markdown(_scoreboard_md(), elem_classes=["api-scoreboard"])
    score_refresh = gr.Button("🔄 Refresh scoreboard", size="sm")
    score_refresh.click(fn=_refresh_score_only, inputs=None, outputs=[score_md])

    gr.Markdown("### 📋 Endpoint catalog *(every route exposed by the FastAPI app)*")
    gr.Markdown(_ENDPOINT_CATALOG_MD, elem_classes=["api-catalog"])

    gr.Markdown(
        "### 🚀 Try it — click any **▶️ Try** button below to call the live API.\n"
        "*All buttons hit `localhost:7860` (the same process this UI is mounted "
        "on). The curl panel always shows the public Space URL so the snippet "
        "is pastable from your own machine.*"
    )

    # ─────────────── GET /health ────────────────────────────────────────
    with gr.Accordion("GET  /health  —  is the server up?", open=True):
        h_btn = gr.Button("▶️ Try /health", variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response**")
                h_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                h_curl = gr.Markdown(_build_curl_md("GET", "/health", None))
        h_btn.click(fn=_try_health, inputs=None,
                    outputs=[h_resp, h_curl, score_md])

    # ─────────────── GET /api/info ──────────────────────────────────────
    with gr.Accordion("GET  /api/info  —  service descriptor", open=False):
        ai_btn = gr.Button("▶️ Try /api/info", variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response**")
                ai_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                ai_curl = gr.Markdown(_build_curl_md("GET", "/api/info", None))
        ai_btn.click(fn=_try_api_info, inputs=None,
                     outputs=[ai_resp, ai_curl, score_md])

    # ─────────────── GET /tasks ─────────────────────────────────────────
    with gr.Accordion("GET  /tasks  —  three task tiers + action schemas", open=False):
        gr.Markdown(
            "*Returns `action_screen` (5 steps), `war_room` (15 steps), "
            "`drift_ops` (25 steps) — plus the canonical Responder and "
            "Overseer action schemas your agent should emit.*"
        )
        t_btn = gr.Button("▶️ Try /tasks", variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response**")
                t_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                t_curl = gr.Markdown(_build_curl_md("GET", "/tasks", None))
        t_btn.click(fn=_try_tasks, inputs=None,
                    outputs=[t_resp, t_curl, score_md])

    # ─────────────── POST /reset ────────────────────────────────────────
    with gr.Accordion("POST  /reset  —  start a new episode", open=False):
        gr.Markdown(
            "*This drives the **Reward Scoreboard above** — after a successful "
            "reset, the scoreboard switches from the 'no active episode' "
            "message to live cumulative rewards.*"
        )
        with gr.Row():
            r_task = gr.Dropdown(
                choices=["action_screen", "war_room", "drift_ops"],
                value="war_room",
                label="task_id",
            )
            r_seed = gr.Textbox(label="seed", value="42",
                                 placeholder="integer or blank for random")
            r_mode = gr.Dropdown(
                choices=["alternating", "train_overseer", "train_responder"],
                value="alternating",
                label="mode",
            )
        r_btn = gr.Button("▶️ Try /reset", variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response** — initial Observation")
                r_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                r_curl = gr.Markdown(
                    _build_curl_md("POST", "/reset",
                                   {"task_id": "war_room", "seed": 42,
                                    "mode": "alternating"}))
        r_btn.click(fn=_try_reset, inputs=[r_task, r_seed, r_mode],
                    outputs=[r_resp, r_curl, score_md])

    # ─────────────── POST /step (Responder) ─────────────────────────────
    with gr.Accordion(
        "POST  /step  —  submit ONE action  (Responder OR Overseer)",
        open=False,
    ):
        gr.Markdown(
            "*The OpenEnv `/step` endpoint takes a discriminated `Action` "
            "payload — either a Responder proposal or an Overseer decision, "
            "depending on `turn_phase`. Both subforms below build the right "
            "JSON shape and POST to the same `/step` route. **Watch the "
            "Reward Scoreboard at the top — it updates after every successful "
            "step!***"
        )
        gr.Markdown("#### ── Sub-form A: Responder action ────────────────────")
        with gr.Row():
            sr_role = gr.Dropdown(
                choices=["generic", "network_ops", "db_admin", "sec_ops"],
                value="generic",
                label="responder_role",
            )
            sr_action = gr.Dropdown(
                choices=[
                    "query_logs", "check_metrics", "check_dependencies",
                    "check_recent_deploys", "check_service_status",
                    "restart_service", "rollback_deploy", "scale_service",
                    "disable_feature_flag", "clear_cache",
                    "execute_runbook_step",
                ],
                value="query_logs",
                label="action_type",
            )
        with gr.Row():
            sr_service = gr.Textbox(label="parameters.service",
                                     placeholder="api-gateway",
                                     value="api-gateway")
            sr_runbook = gr.Textbox(
                label="parameters.runbook_action (only for execute_runbook_step)",
                placeholder="optional", value="",
            )
        sr_reasoning = gr.Textbox(
            label="reasoning",
            placeholder="brief plan-of-record — e.g. 'gather evidence first'",
            value="gather evidence before remediating",
        )
        sr_btn = gr.Button("▶️ Try /step  (responder)",
                           variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response** — Observation + reward + done + info")
                sr_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                sr_curl = gr.Markdown(
                    _build_curl_md("POST", "/step", {
                        "role": "responder",
                        "responder": {
                            "responder_role": "generic",
                            "action_type": "query_logs",
                            "parameters": {"service": "api-gateway"},
                            "reasoning": "gather evidence before remediating",
                        },
                    }))
        sr_btn.click(
            fn=_try_step_responder,
            inputs=[sr_role, sr_action, sr_service, sr_runbook, sr_reasoning],
            outputs=[sr_resp, sr_curl, score_md],
        )

        gr.Markdown("#### ── Sub-form B: Overseer action ────────────────────")
        with gr.Row():
            so_decision = gr.Dropdown(
                choices=["approve", "flag", "block", "escalate"],
                value="approve",
                label="decision",
            )
            so_just = gr.Textbox(
                label="justification (min 1 char; cite evidence for full reward)",
                value="diagnostic action — gathers evidence, low blast radius",
            )
        so_btn = gr.Button("▶️ Try /step  (overseer)",
                           variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response** — Observation + reward + done + info")
                so_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                so_curl = gr.Markdown(
                    _build_curl_md("POST", "/step", {
                        "role": "overseer",
                        "overseer": {
                            "decision": "approve",
                            "justification": "diagnostic action — gathers evidence, low blast radius",
                        },
                    }))
        so_btn.click(
            fn=_try_step_overseer,
            inputs=[so_decision, so_just],
            outputs=[so_resp, so_curl, score_md],
        )

    # ─────────────── GET /state ─────────────────────────────────────────
    with gr.Accordion("GET  /state  —  full EpisodeState snapshot", open=False):
        s_btn = gr.Button("▶️ Try /state", variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response**")
                s_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                s_curl = gr.Markdown(_build_curl_md("GET", "/state", None))
        s_btn.click(fn=_try_state, inputs=None,
                    outputs=[s_resp, s_curl, score_md])

    # ─────────────── GET /grader (the rewards endpoint!) ────────────────
    with gr.Accordion(
        "GET  /grader  —  F1, confusion matrix, **cumulative rewards 🏆**",
        open=False,
    ):
        gr.Markdown(
            "*This is what populates the **Live Reward Scoreboard at the top**. "
            "The `score` field (top-level) is the OpenEnv normalised metric "
            "(Overseer F1 in [0,1]); it's what automated evaluators read.*"
        )
        g_btn = gr.Button("▶️ Try /grader", variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response**")
                g_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                g_curl = gr.Markdown(_build_curl_md("GET", "/grader", None))
        g_btn.click(fn=_try_grader, inputs=None,
                    outputs=[g_resp, g_curl, score_md])

    gr.Markdown("---")
    gr.Markdown("### 🛡️ SENTINEL/Live endpoints *(the layer that ships beyond OpenEnv)*")

    # ─────────────── POST /live/oversee ─────────────────────────────────
    with gr.Accordion(
        "POST  /live/oversee  —  real-time agent oversight (no scenario state)",
        open=False,
    ):
        with gr.Row():
            lo_action = gr.Textbox(
                label="action_description",
                value="DROP TABLE users",
                placeholder="free-form proposal",
            )
            lo_tool = gr.Textbox(label="proposed_tool",
                                  value="DROP TABLE", placeholder="optional")
            lo_target = gr.Textbox(label="proposed_target",
                                    value="users", placeholder="optional")
        with gr.Row():
            lo_severity = gr.Dropdown(
                choices=["low", "medium", "high", "critical"],
                value="critical", label="severity_hint",
            )
            lo_backend = gr.Radio(
                choices=["heuristic", "trained"],
                value="heuristic", label="backend",
            )
        lo_btn = gr.Button("▶️ Try /live/oversee",
                            variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response** — verdict + risk_score + shield")
                lo_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                lo_curl = gr.Markdown(_build_curl_md("POST", "/live/oversee", {
                    "action_description": "DROP TABLE users",
                    "proposed_tool": "DROP TABLE",
                    "proposed_target": "users",
                    "severity_hint": "critical",
                }))
        lo_btn.click(
            fn=_try_live_oversee,
            inputs=[lo_action, lo_tool, lo_target, lo_severity, lo_backend],
            outputs=[lo_resp, lo_curl, score_md],
        )

    # ─────────────── GET /live/stats ────────────────────────────────────
    with gr.Accordion(
        "GET  /live/stats  —  lifetime counters since server start",
        open=False,
    ):
        ls_btn = gr.Button("▶️ Try /live/stats",
                            variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response**")
                ls_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                ls_curl = gr.Markdown(_build_curl_md("GET", "/live/stats", None))
        ls_btn.click(fn=_try_live_stats, inputs=None,
                     outputs=[ls_resp, ls_curl, score_md])

    # ─────────────── GET /live/health ───────────────────────────────────
    with gr.Accordion(
        "GET  /live/health  —  feature-level health (trained backend present?)",
        open=False,
    ):
        lh_btn = gr.Button("▶️ Try /live/health",
                            variant="primary", size="sm")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Response**")
                lh_resp = gr.Markdown("_(no response yet)_")
            with gr.Column():
                gr.Markdown("**Equivalent curl**")
                lh_curl = gr.Markdown(_build_curl_md("GET", "/live/health", None))
        lh_btn.click(fn=_try_live_health, inputs=None,
                     outputs=[lh_resp, lh_curl, score_md])

    # ─────────────── /docs link ─────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        f"### 📖 [Open Swagger UI in a new tab → `/docs`]({_PUBLIC_BASE}/docs)\n\n"
        "FastAPI's auto-generated interactive OpenAPI documentation. "
        "Has request schemas, response schemas, and a built-in 'Execute' "
        "button for every endpoint."
    )
