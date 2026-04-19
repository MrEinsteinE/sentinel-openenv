"""
drift.py — schema drift injector for Drift Ops scenarios.

At a scenario-configured step, mutates the live scenario state:
  - service_rename    : rename a service (old name now 404s)
  - param_rename      : rename a runbook action parameter
  - service_removal   : remove a service from known_services
  - error_injection   : make a tool response start returning structured errors

Drift events are appended to the observation so the Overseer can (in principle)
detect them — but a stale Responder won't, which is the test.

All drift is deterministic given the scenario seed — enables reproducible eval.
"""
from __future__ import annotations

import copy
from typing import Any


def should_trigger(scenario: dict, step_count: int, already_triggered: bool) -> bool:
    """True if this step should trigger scenario-level drift."""
    if already_triggered:
        return False
    if not scenario.get("drift_enabled"):
        return False
    return step_count >= int(scenario.get("drift_triggers_at_step", 10))


def apply_mutations(scenario: dict) -> tuple[dict, list[dict]]:
    """Apply the scenario's configured mutations. Returns (new_scenario, drift_events)."""
    sc = copy.deepcopy(scenario)
    events: list[dict] = []

    for m in sc.get("drift_mutations", []):
        mtype = m.get("type")
        if mtype == "service_rename":
            old = m["from"]
            new = m["to"]
            # Replace in known_services
            sc["known_services"] = [new if s == old else s for s in sc.get("known_services", [])]
            # Remap tool responses
            for action, by_svc in list(sc.get("tool_responses", {}).items()):
                if old in by_svc:
                    by_svc[new] = by_svc.pop(old)
            # Remap correct_root_cause
            if sc.get("correct_root_cause") and sc["correct_root_cause"].get("service") == old:
                sc["correct_root_cause"]["service"] = new
            # Remap correct_remediation_sequence keys of form "action:svc"
            sc["correct_remediation_sequence"] = [
                _rename_target(k, old, new) for k in sc.get("correct_remediation_sequence", [])
            ]
            sc["wrong_actions"] = {
                _rename_target(k, old, new): v for k, v in sc.get("wrong_actions", {}).items()
            }
            sc["catastrophic_actions"] = [
                _rename_target(k, old, new) for k in sc.get("catastrophic_actions", [])
            ]
            events.append({
                "type": "service_rename",
                "from": old, "to": new,
                "reason": m.get("reason", ""),
            })

        elif mtype == "param_rename":
            old = m["from"]
            new = m["to"]
            action = m.get("action", "execute_runbook_step")
            # Update correct_remediation_sequence entries like "execute_runbook_step:old"
            sc["correct_remediation_sequence"] = [
                _rename_action_param(k, action, old, new)
                for k in sc.get("correct_remediation_sequence", [])
            ]
            sc["wrong_actions"] = {
                _rename_action_param(k, action, old, new): v
                for k, v in sc.get("wrong_actions", {}).items()
            }
            sc["catastrophic_actions"] = [
                _rename_action_param(k, action, old, new)
                for k in sc.get("catastrophic_actions", [])
            ]
            events.append({
                "type": "param_rename",
                "action": action, "from": old, "to": new,
                "reason": m.get("reason", ""),
            })

        elif mtype == "service_removal":
            svc = m["service"]
            sc["known_services"] = [s for s in sc.get("known_services", []) if s != svc]
            for by_svc in sc.get("tool_responses", {}).values():
                by_svc.pop(svc, None)
            events.append({
                "type": "service_removal",
                "service": svc,
                "reason": m.get("reason", ""),
            })

        elif mtype == "error_injection":
            action = m.get("action", "query_logs")
            svc = m.get("service")
            err = m.get("error", "ERROR: service temporarily unavailable")
            by_svc = sc.setdefault("tool_responses", {}).setdefault(action, {})
            if svc:
                by_svc[svc] = err
            events.append({
                "type": "error_injection",
                "action": action, "service": svc,
                "reason": m.get("reason", ""),
            })

    return sc, events


def _rename_target(key: str, old: str, new: str) -> str:
    """Rewrite `action:svc` where svc == old."""
    if ":" not in key:
        return key
    action, target = key.split(":", 1)
    return f"{action}:{new}" if target == old else key


def _rename_action_param(key: str, action: str, old: str, new: str) -> str:
    """Rewrite `<action>:<param>` where action matches and param == old."""
    if ":" not in key:
        return key
    act, param = key.split(":", 1)
    return f"{act}:{new}" if (act == action and param == old) else key


def describe_drift_for_overseer(events: list[dict]) -> str:
    """One-line summary of drift events — surfaced in Observation for Overseer."""
    if not events:
        return ""
    parts: list[str] = []
    for e in events:
        if e["type"] == "service_rename":
            parts.append(f"service renamed: {e['from']} → {e['to']}")
        elif e["type"] == "param_rename":
            parts.append(f"param renamed on {e['action']}: {e['from']} → {e['to']}")
        elif e["type"] == "service_removal":
            parts.append(f"service removed: {e['service']}")
        elif e["type"] == "error_injection":
            parts.append(f"error injected on {e['action']}:{e.get('service','?')}")
    return "DRIFT EVENT — " + "; ".join(parts)
