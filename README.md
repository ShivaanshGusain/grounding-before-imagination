# Grounding Before Imagination: go/no-go experiment (Stage A)

This tests the proposal's core question on a domain we fully control:

> Can **text alone** produce a starting state for a **frozen, action-conditioned world model**
> such that rolling it forward gives correct futures? And does training the text→state
> **writer through the predictor** beat matching the start state only?

Stage A uses a synthetic 2D pushing world (a pusher, 3 pucks, wood/metal mass, smooth/rough
friction). Everything is generated, so labels are exact and runs are fast. This is the
**debugging domain**: here text fully specifies the scene, so a parser plus the real
simulator would win. The main paper result must later come from real video (Stage B, DROID).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `world_model.py` | Pretrains a small JEPA-style encoder + action-conditioned predictor on simulated video (EMA target + VICReg), calibrates the predictor on target-encoder latents, then **freezes** it. Prints a sanity report. |
| 2 | `build_dataset.py` | 12k scenes × 4 action sequences × 8 steps. Stores scene **text** (description only), actions, frozen latents and outcome labels. Two actions (up-right, down-left) are excluded from all **writer- and probe-training** sequences. The frozen world model *is* pretrained on all 9 actions; that is the mechanism being tested (the predictor knows their dynamics, so the writer only has to place the start state). |
| 3 | `text_features.py` | Frozen text features: `hf` (Qwen2.5-0.5B mean-pooled hidden states) or `bow` (hashed n-grams + entity-bound tokens, no download). |
| 4 | `writer.py` | Trains the text→state writer on the first 60% of train scenes: `static` (match E(o₀)), `predictive` (loss through the frozen predictor, horizons ≤ 4), `random` (through an untrained predictor). |
| 5 | `evaluate.py` | Grounding curves + usable-information probes. Probes train on the other 40% of train scenes, so writer features are out-of-sample. Writes `results/<enc>/report.md`. |

## Running on the GX10

```bash
# inside an NGC PyTorch container (or any env with CUDA torch):
pip install transformers accelerate      # torch should come from the container
bash run_all.sh smoke                    # tiny run, BOW text only: checks nothing crashes
bash run_all.sh full                     # real run: BOW and Qwen text, 3 seeds
```

Everything the script prints is also saved to `logs/run_<profile>_<timestamp>.log` (via `tee`).
For anything you run by hand: `your_command 2>&1 | tee -a logs/manual.log`.

Overrides: `ENCODERS=hf`, `SEEDS=0`, `SCENES=4000`, `WM_STEPS=4000`, `HF_MODEL=...`.
Runtime is unmeasured. Expect to fix small bugs on the first runs.

## What results you get

**0. World-model sanity** (end of `logs/wm.log`). Check this first. If it fails, nothing
downstream means anything. **Ignore it for the smoke run (60 steps is too few).**
- `state_probe_R2_encoder_t0` high (≳0.9) and close to `..._train`: latents encode positions.
- `rollout_err_over_persistence_t*` < 1: the predictor beats "nothing moves".
- `online_target_gap` small (≲0.1) and `latent_std_mean` not ≈0 (no collapse).

**1. Grounding table (H1, H2).** Rollout error ÷ error of copying the *true* start latent, per
horizon t=1..8, seen vs unseen actions.
- **H1:** `predictive` clearly below 1, below the controls `pred_state_randP` (same written state,
  random dynamics) and `pred_state_shufA` (same state, wrong actions), and approaching `privileged`.
- **H2:** `predictive` < `static`, especially at t > 4 (beyond the training horizon) and on
  actions unseen by the writer.

**2. Usable-information table (H3).** Held-out log loss for 4 outcome questions about the red
puck (contacted? moved? hit another puck? final 3×3 cell?), linear and MLP probes. Actions enter
every condition as per-step (dx, dy) directions, so text-only probes can extrapolate to the
held-out diagonal actions. Conditions: `a_text`, `b_writer_state`, `c_rollout_pred` (**method**),
`c_rollout_stat`, the controls `e1_state_randP`, `e2_state_shufA`, `e3_randwriter`, and the
privileged references `d_priv_start` and `f_priv_future`.

**BOW is a lower bound / pipeline check.** Draw conclusions from the `hf` report.
Seeds vary the writer, probes and random-predictor draw; the world model and split are fixed.

## Decision rule (from the proposal)

| Pattern | Meaning | Next step |
|---|---|---|
| `c` ≈ `a` and `c` ≈ `b` | World model adds nothing from text | Stop or negative result |
| `d` ≫ `a`, but `c` ≈ `a` | Dynamics help, text grounding fails | Work on the writer (text JEPA, generative writer, `--text-style qualitative`) |
| `c` < `a`, `b`, `e1`, `e2` across seeds (MLP probe); predictive < static on unseen actions / long horizons | **Go** | Stage B on DROID / JEPA-WMs |

## Stage B (after Stage A works)

`python inspect_jepa_wms.py --name jepa_wm_wall` prints Meta's pretrained JEPA-WM structure,
so we can write an adapter and swap it in for our own world model. Then DROID
(V-JEPA 2-AC / `jepa_wm_droid`) with captioned scenes.
