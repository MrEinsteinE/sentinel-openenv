# SENTINEL — 3-Minute Pitch Outline

## Goal

Deliver a 3-minute pitch that lands the 40% Environment Innovation and 30% Storytelling marks cleanly, with enough reward-curve and before/after evidence to credibly claim the 20% Reward Improvement bucket.

---

## Block-by-block script

### 0:00 – 0:20 — Hook (20 sec)
> "Today, AI agents are being deployed as autonomous cloud responders. One bad `rm -rf` ends careers and costs millions. We can't have humans review every action. So who oversees the AI? **Another AI.** SENTINEL is the first OpenEnv for training scalable LLM oversight."

**On-screen**: one real-world headline about an AI-caused outage OR a GIF of an untrained agent proposing `rm -rf` on a prod DB.

### 0:20 – 0:50 — The environment (30 sec)
> "Meet the war room. A **Responder** agent investigates a cloud incident and proposes actions. An **Overseer** agent — the one we're training — sees each proposal alongside a counterfactual preview, policy constraints, and any mid-episode schema drift. It decides: **approve, flag, block, or escalate** — and must justify the decision in natural language."

**On-screen**: the Gradio 3-column replay viewer animating a single turn. Show the counterfactual badge changing from "EXPECTED GOOD" to "CATASTROPHIC".

### 0:50 – 1:10 — Why the design is novel (20 sec)
> "Three ideas make this uniquely suited for oversight training. One: **dual ground-truth** per scenario — every Responder action is labelled catastrophic, wrong, correct, ambiguous, or neutral, with severity weights. Two: **mid-episode schema drift** — tools rename, services vanish — testing whether the Overseer catches stale assumptions. Three: **justification coherence reward** capped to prevent reward hacking, aligning with the Mercor-sub-theme."

**On-screen**: short split showing the reward table (catastrophic × approve = −1.5 × severity) and a drift event banner flashing mid-episode.

### 1:10 – 1:50 — Training (40 sec)
> "We trained a Qwen3-1.7B Overseer with TRL GRPO, Unsloth QLoRA, vLLM colocate — 8 hours on a single L4. Binary reward per decision. A heuristic Responder provides realistic opposition. Phase 2 flips into self-play co-training: the Overseer and Responder alternate, with an auto-curriculum that advances difficulty when F1 crosses thresholds."

**On-screen**: reward curve — Overseer F1 climbing from 0.22 (zero-shot) to 0.82 over training steps. Annotated: baseline lines for "naive" (always-approve) and "random" overseers.

### 1:50 – 2:30 — Before / after demo (40 sec)
> "Watch the difference. Pre-training, our Overseer approves a rollback on postgres-db during an active OOM incident. Three steps later, the database is dead. Post-training, the same proposal: **BLOCKED.** Justification: *'rollback_deploy on postgres-db: database has no recent snapshot, current OOM is caused by analytics-service runaway query — rollback won't fix the upstream problem.'* And when mid-episode schema drift renames a service, the trained Overseer flags a call to the old name; the untrained one silently approves."

**On-screen**: two rollouts side-by-side with the same seed — text panels highlighting the divergence.

### 2:30 – 2:50 — Why this matters / closing (20 sec)
> "SENTINEL is the first OpenEnv that targets Theme 1 Multi-Agent *plus* the Fleet AI, Halluminate, and Patronus sub-theme prizes in one coherent artefact. It ships as a standard OpenEnv Space — anyone can `pip install openenv-sentinel` and start training their own Overseer today. This isn't a toy: it's a microcosm of a real AI-safety problem, with a verifiable reward and a clean before-after story."

**On-screen**: repo + Space URLs + `pip install` command. Hold the final frame for 3 seconds so judges can screenshot.

### 2:50 – 3:00 — Thanks / Q&A handoff (10 sec)
> "Built by Einstein and Sidra. Thank you. Happy to go deeper on the reward design, drift mechanics, or the co-training loop in Q&A."

---

## Rehearsal checklist

- [ ] Record 3 takes of the whole pitch standing up. Keep the best one.
- [ ] Time each block — stay within ±5 seconds of the target.
- [ ] Pre-load the Gradio replay viewer in the browser with `war_room` + `seed=42` so the `Play Episode` click fires instantly.
- [ ] Have the reward curve PNG open in a second tab as a fallback if Gradio hiccups.
- [ ] Write out 3 anticipated Q&A responses:
  1. *Why GRPO over PPO?* — "verifiable reward, small model; GRPO's group normalization avoids the critic."
  2. *How do you know the baseline is fair?* — "50 held-out scenarios, never seen during training; random + naive + policy-aware floors published."
  3. *What stops the Overseer from always blocking?* — "FP penalties plus Responder reward decay; justifications must ground in scenario evidence scored by a coherence judge."

---

## Production notes

- 1920×1080, 30fps, subtitles on.
- OBS screen capture over a soft voice track; no face cam.
- DaVinci Resolve free tier for editing.
- Export as MP4 H.264 to upload to YouTube unlisted. Embed in README + HF Space.
- Budget: 3 hours Sunday morning for the cut; 1 hour of rehearsal.
