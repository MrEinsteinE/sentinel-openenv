# SENTINEL — Pitch Deck

15-slide Marp deck for the 3-minute pitch. Source of truth is `slides.md` (markdown) with styling in `theme.css`.

## Render — pick the path that fits your setup

### Option A — Zero install · **web.marp.app** (recommended if Node is missing or old)

1. Open https://web.marp.app/
2. Paste the contents of `slides.md` into the left pane
3. Open `theme.css`, copy its contents, and paste them into the frontmatter `style:` block (or click "Upload" and attach theme.css)
4. Click **Export → PDF** (or PPTX) · download

Takes 60 seconds. Renders identically to the CLI.

### Option B — VS Code Marp extension (best for iterating)

Install [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode).
Open `slides.md` → live preview pane appears.
Command palette → *Marp: Export slide deck* → PDF / PPTX.

### Option C — Marp CLI (requires Node.js ≥ 20)

```bash
npm install -g @marp-team/marp-cli
marp slides.md --theme theme.css --pdf --allow-local-files --output sentinel-pitch.pdf
marp slides.md --theme theme.css --pptx --allow-local-files --output sentinel-pitch.pptx
marp slides.md --theme theme.css --watch --server    # live preview on :8080
```

### If your Node is old (pre-20)

Install a modern Node via [nodejs.org](https://nodejs.org/) (current LTS) or via `nvm`. This project only used Node for the Marp step — nothing else depends on it.

## Slide map (15 slides)

| # | Class | Topic | Purpose |
|---|---|---|---|
| 1 | `title` | Title + authors | 5-second hand-off |
| 2 | `stat` | `rm -rf /` | Visceral hook |
| 3 | — | Research question | Framing |
| 4 | — | Environment diagram | Architecture |
| 5 | — | Three task tiers | Scope |
| 6 | `split` | Dual reward (shaped + binary) | Novelty #1 |
| 7 | `stat` | **72B < random** | Baseline hammer |
| 8 | — | Full baseline table | Receipts |
| 9 | `split` | Why LLMs fail (behaviour gap) | Narrative pivot |
| 10 | — | 3-stage training pipeline | Execution credibility |
| 11 | `split` | Before/After war_room s42 | Demo anchor |
| 12 | — | Drift Ops gap | Research-grade point |
| 13 | — | Theme coverage (4 bonus lanes) | Prize-math reminder |
| 14 | `stat` | F1: 0.09 → 0.85 | The ask |
| 15 | `split` | Ship + try it | CTA |
| 16 | `title` | Thank you + Q&A | Outro |

## Timing target

Block aim: **2:45–2:55 total** (15 sec buffer on the 3-min cap).

| Slides | Block | Seconds |
|---|---|---:|
| 1 | hand-off | 5 |
| 2 | hook | 15 |
| 3–5 | problem + env | 40 |
| 6–7 | dual reward + baseline bomb | 30 |
| 8 | baseline table | 15 |
| 9 | behaviour gap | 20 |
| 10 | training pipeline | 20 |
| 11 | before/after | 35 |
| 12 | drift gap | 15 |
| 13 | themes | 10 |
| 14–15 | ask + ship | 15 |
| 16 | outro | 5 |
| | **Total** | **~225 s** |

## Iterating content

All numbers come from `../eval_data/baseline_*.json` and `../eval_data/demo_*.md`. If you re-run baselines on-site with the trained model, refresh slide 8 + slide 11 before the pitch.

## Export checklist before pitch day

- [ ] `sentinel-pitch.pdf` on local disk (primary)
- [ ] `sentinel-pitch.pptx` on local disk (backup for judges' laptops)
- [ ] Both on a USB stick (ultimate fallback if wifi dies)
- [ ] Gradio UI pre-loaded with `war_room` + `seed=42` in a separate browser tab
- [ ] YouTube unlisted 2-min video link on a sticky note
