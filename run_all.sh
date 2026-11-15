#!/usr/bin/env bash
# Full go/no-go pipeline (Stage A: synthetic pushing domain).
#   bash run_all.sh smoke   # ~minutes, tiny sizes, BOW text only: checks that everything runs
#   bash run_all.sh full    # the real run (GPU)
set -euo pipefail
cd "$(dirname "$0")"

PROFILE=${1:-full}
HF_MODEL=${HF_MODEL:-Qwen/Qwen2.5-0.5B}
if [[ "$PROFILE" == "smoke" ]]; then
  WM_STEPS=60; SCENES=240; EPOCHS=3; SEEDS="0"; ENCODERS=${ENCODERS:-"bow"}
else
  WM_STEPS=${WM_STEPS:-8000}; SCENES=${SCENES:-12000}; EPOCHS=${EPOCHS:-80}; SEEDS=${SEEDS:-"0,1,2"}
  ENCODERS=${ENCODERS:-"bow hf"}
fi
mkdir -p runs results logs

echo "== 1. pretrain + freeze the world model"
python world_model.py --steps "$WM_STEPS" --out runs/wm.pt 2>&1 | tee logs/wm.log

echo "== 2. grounding dataset (frozen encoder latents, texts, labels)"
python build_dataset.py --wm runs/wm.pt --out runs/data.pt --scenes "$SCENES" 2>&1 | tee logs/data.log

for ENC in $ENCODERS; do
  echo "== 3. text features: $ENC"
  python text_features.py --data runs/data.pt --encoder "$ENC" --model "$HF_MODEL" --out "runs/text_${ENC}.pt" 2>&1 | tee "logs/text_${ENC}.log"
  for SEED in ${SEEDS//,/ }; do
    for MODE in static predictive random; do
      echo "== 4. writer $MODE / $ENC / seed $SEED"
      python writer.py --mode "$MODE" --text "runs/text_${ENC}.pt" --seed "$SEED" --epochs "$EPOCHS" \
        --out "runs/writer_${MODE}_${ENC}_s${SEED}.pt" 2>&1 | tee "logs/writer_${MODE}_${ENC}_s${SEED}.log"
    done
  done
  echo "== 5. evaluate: $ENC"
  python evaluate.py --text "runs/text_${ENC}.pt" --tag "$ENC" --seeds "$SEEDS" --outdir "results/${ENC}" 2>&1 | tee "logs/eval_${ENC}.log"
done

echo "done. Reports: results/*/report.md   World-model sanity: tail logs/wm.log"
