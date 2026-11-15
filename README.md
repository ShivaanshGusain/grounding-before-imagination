# Grounding Before Imagination: go/no-go experiment (Stage A)

This tests the proposal's core question on a domain we fully control:

> Can **text alone** produce a starting state for a **frozen, action-conditioned world model**
> such that rolling it forward gives correct futures? And does training the text→state
> **writer through the predictor** beat matching the start state only?

Stage A uses a synthetic 2D pushing world (a pusher, 3 pucks, wood/metal mass, smooth/rough
friction). Everything is generated, so labels are exact and runs are fast. This is the
**debugging domain**: in it, text fully specifies the scene, so a parser plus the real
simulator would win. The main paper result must later come from real video (Stage B, DROID).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `world_model.py` | Pretrains a small JEPA-style encoder + action-conditioned predictor on simulated video (EMA target + VICReg), then **freezes** it. Prints a sanity report. |
| 2 | `build_dataset.py` | 12k scenes × 4 action sequences × 8 steps. Stores scene **text** (description only), actions, frozen latents and outcome labels. Two actions (up-right, down-left) are **held out** of all training sequences. |
| 3 | `text_features.py` | Frozen text features: `hf` (Qwen2.5-0.5B mean-pooled hidden states) or `bow` (hashed n-grams, no download). |
| 4 | `writer.py` | Trains the text→state writer: `static` (match E(o₀)), `predictive` (loss through frozen predictor, horizons ≤ 4), `random` (through an untrained predictor, control). |
| 5 | `evaluate.py` | Grounding curves + usable-information probes → `results/<enc>/report.md`. |

## Running on the GX10

```bash
# copy this folder to the GX10, then inside an NGC PyTorch container (or any env with CUDA torch):
pip install transformers accelerate      # torch should come from the container
bash run_all.sh smoke                    # tiny run, BOW text only — checks nothing crashes
bash run_all.sh full                     # real run: BOW and Qwen text, 3 seeds
```

Useful overrides: `ENCODERS=hf`, `SEEDS=0`, `SCENES=4000`, `WM_STEPS=4000`, `HF_MODEL=...`.
I have not run this code (the laptop has no torch), so expect to fix small bugs on the first
smoke run. Runtime is unmeasured. Rough guess: world model tens of minutes, everything else
much less.

## What results you get

**0. World-model sanity** (end of `logs/wm.log`). Check this first; if it fails, nothing
downstream means anything.
- `state_probe_R2_encoder_t0` should be high (≳0.9): latents encode positions.
- `rollout_err_over_persistence_t*` < 1: the predictor beats "nothing moves".

**1. Grounding table (H1, H2).** For each writer and horizon t=1..8: rollout error ÷ error of
copying the *true* start state. Also start-state retrieval@1, with seen and unseen actions
reported separately.
- **H1 holds** if `predictive` is clearly below `random` and close to the `privileged` row.
- **H2 holds** if `predictive` < `static`, especially at t > 4 (beyond training horizon) and on
  **unseen actions**.

**2. Usable-information table (H3).** Held-out log loss for 4 outcome questions about the red
puck (contacted? moved? hit another puck? final 3×3 cell?), with linear and MLP probes:
- `a_text` text + actions only
- `b_writer_state` + written start state, no rollout
- `c_rollout_pred` + rollout of the predictive writer (**the method**)
- `c_rollout_stat` + rollout of the static writer
- `e_random_pred` + rollout through an untrained predictor (control)
- `d_priv_start` rollout from the real start latent (privileged reference)
- `f_priv_future` true future latents (reader ceiling)

## Decision rule (from the proposal)

| Pattern | Meaning | Next step |
|---|---|---|
| `c` ≈ `a` and `c` ≈ `b` | World model adds nothing from text | Stop or negative result |
| `d` ≫ `a`, but `c` ≈ `a` | Dynamics help, text grounding fails | Work on the writer (text JEPA, generative writer, `--text-style qualitative`) |
| `c` < `a`, `b`, `e` across seeds; predictive < static on unseen actions / long horizons | **Go** | Stage B on DROID / JEPA-WMs |

Expected trap: with linear probes, `a_text` cannot model action × state interactions, so
`c` may win by a lot. Read the **MLP-probe** table as the stricter test. Unseen-action rows
are the most informative, because the text-only probes never saw those action dims during
training.

## Stage B (after Stage A works)

`python inspect_jepa_wms.py --name jepa_wm_wall` prints Meta's pretrained JEPA-WM structure,
so we can write an adapter and swap it in for our own world model. Then move to DROID
(V-JEPA 2-AC / `jepa_wm_droid`) with captioned scenes.
