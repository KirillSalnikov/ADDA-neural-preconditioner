#!/usr/bin/env bash
# Train spectral hex-prism preconditioner in real-ADDA-validated chunks.
#
# This intentionally does not pick checkpoints by training loss. After every
# chunk it runs apps/eval_spectral_adda.py and only promotes a checkpoint when
# the real ADDA residual improves on the target case.

set -euo pipefail
trap 'rc=$?; echo "ERROR: train_spectral_hex_real_loop.sh failed at line $LINENO with status $rc" >&2' ERR

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-0}"
SEED_BASE="${SEED_BASE:-42}"
CYCLES="${CYCLES:-4}"
STEPS_PER_CYCLE="${STEPS_PER_CYCLE:-250}"
LR="${LR:-1e-6}"
TRAIN_GRID="${TRAIN_GRID:-64}"
TRAIN_GRID_MIN="${TRAIN_GRID_MIN:-$TRAIN_GRID}"
TRAIN_GRID_MAX="${TRAIN_GRID_MAX:-$TRAIN_GRID}"
VAL_GRID="${VAL_GRID:-64}"
VAL_GRIDS="${VAL_GRIDS:-$VAL_GRID}"
SCORE_MODE="${SCORE_MODE:-min}"
M_RE="${M_RE:-3.0}"
M_IM="${M_IM:-0.0}"
DPL="${DPL:-15}"
KD="${KD:-0.41887902047863906}"
RADIUS="${RADIUS:-32}"
LOSS="${LOSS:-planewave_krylov}"
KRYLOV_ITERS="${KRYLOV_ITERS:-8}"
SPECTRAL_FREQ_CHUNK_SIZE="${SPECTRAL_FREQ_CHUNK_SIZE:-0}"
SPECTRAL_FREQ_CHECKPOINT_CHUNKS="${SPECTRAL_FREQ_CHECKPOINT_CHUNKS:-0}"
SPECTRAL_CORRECTION_HIDDEN="${SPECTRAL_CORRECTION_HIDDEN:-0}"
SPECTRAL_CORRECTION_LAYERS="${SPECTRAL_CORRECTION_LAYERS:-2}"
SPECTRAL_CORRECTION_SCALE="${SPECTRAL_CORRECTION_SCALE:-1.0}"
SPECTRAL_CORRECTION_PENALTY="${SPECTRAL_CORRECTION_PENALTY:-0.0}"
SPECTRAL_FREEZE_BASE="${SPECTRAL_FREEZE_BASE:-0}"
ANCHOR_NUM_PROBES="${ANCHOR_NUM_PROBES:-1}"
ANCHOR_KRYLOV_WEIGHT="${ANCHOR_KRYLOV_WEIGHT:-1.0}"
ANCHOR_PROBE_WEIGHT="${ANCHOR_PROBE_WEIGHT:-0.2}"
ANCHOR_RIGHT_PROBE_WEIGHT="${ANCHOR_RIGHT_PROBE_WEIGHT:-0.2}"
ANCHOR_PROBE_CHUNK="${ANCHOR_PROBE_CHUNK:-1}"
NP="${NP:-16}"
VAL_MAXITER="${VAL_MAXITER:-160}"
VAL_TIMEOUT="${VAL_TIMEOUT:-240}"
EXPORT_SYMMETRY="${EXPORT_SYMMETRY:-none}"
HEX_DL_MIN="${HEX_DL_MIN:-1.0}"
HEX_DL_MAX="${HEX_DL_MAX:-1.0}"
CURRICULUM_FRAC="${CURRICULUM_FRAC:-0.3}"
STOP_AFTER_REJECTS="${STOP_AFTER_REJECTS:-0}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
NAME_PREFIX="${NAME_PREFIX:-SPECTRAL_V4_HEX_REAL_G${TRAIN_GRID_MIN}to${TRAIN_GRID_MAX}_M${M_RE}_${TIMESTAMP}}"
RUN_ROOT="${RUN_ROOT:-runs/${NAME_PREFIX}}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:-models/spectral/checkpoints/best_hex_prism_real.pt}"
BEST_SCORE_FILE="${BEST_SCORE_FILE:-models/spectral/checkpoints/best_hex_prism_real.score}"
START_CHECKPOINT="${START_CHECKPOINT:-$BEST_CHECKPOINT}"

if [[ ! -f "$START_CHECKPOINT" ]]; then
  START_CHECKPOINT="results/SPECTRAL_V4_HEX_G64_M3_PLANEWAVE8_R32_V2/final_model.pt"
fi
if [[ ! -f "$START_CHECKPOINT" ]]; then
  echo "No start checkpoint found: $START_CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$(dirname "$BEST_CHECKPOINT")"

score_from_summary() {
  python3 - "$1" "$2" <<'PY'
import json
import math
import sys

path = sys.argv[1]
mode = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    rows = json.load(f)
vals = [r.get("min_residual") for r in rows if r.get("min_residual") is not None]
vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
if not vals:
    print("inf")
elif mode == "min":
    print(min(vals))
elif mode == "max":
    print(max(vals))
elif mode == "mean_log":
    print(math.exp(sum(math.log(max(v, 1e-300)) for v in vals) / len(vals)))
else:
    raise SystemExit(f"unknown SCORE_MODE={mode!r}; use min, max, or mean_log")
PY
}

is_better() {
  python3 - "$1" "$2" <<'PY'
import math
import sys

candidate = float(sys.argv[1])
best = float(sys.argv[2])
raise SystemExit(0 if math.isfinite(candidate) and candidate < best else 1)
PY
}

eval_checkpoint() {
  local checkpoint="$1"
  local label="$2"
  local out_dir="$RUN_ROOT/eval_${label}"
  mkdir -p "$out_dir"
  local grid_args=()
  local IFS=','
  read -ra val_grid_items <<< "$VAL_GRIDS"
  for grid in "${val_grid_items[@]}"; do
    grid="${grid//[[:space:]]/}"
    if [[ -n "$grid" ]]; then
      grid_args+=(--grid "$grid")
    fi
  done

  python3 -u apps/eval_spectral_adda.py \
    --checkpoint "$checkpoint" \
    "${grid_args[@]}" \
    --shape prism --ay 6 --az 1.0 \
    --m_re "$M_RE" --m_im "$M_IM" \
    --dpl "$DPL" --kd "$KD" \
    --threshold-rel 1e-6 \
    --max-radius "$RADIUS" \
    --blend-identity 1.0 \
    --symmetry "$EXPORT_SYMMETRY" \
    --np "$NP" \
    --maxiter "$VAL_MAXITER" \
    --timeout "$VAL_TIMEOUT" \
    --output-dir "$out_dir" \
    2>&1 | tee "$out_dir/evaluator.log" >&2

  score_from_summary "$out_dir/summary.json" "$SCORE_MODE"
}

echo "=== Real-ADDA spectral training loop ==="
echo "start_checkpoint=$START_CHECKPOINT"
echo "best_checkpoint=$BEST_CHECKPOINT"
echo "run_root=$RUN_ROOT"
echo "cycles=$CYCLES steps_per_cycle=$STEPS_PER_CYCLE train_grid=${TRAIN_GRID_MIN}..${TRAIN_GRID_MAX} val_grids=$VAL_GRIDS score_mode=$SCORE_MODE export_symmetry=$EXPORT_SYMMETRY"
echo "stop_after_rejects=$STOP_AFTER_REJECTS"
echo

current_checkpoint="$START_CHECKPOINT"
if [[ -f "$BEST_SCORE_FILE" ]]; then
  best_score="$(cat "$BEST_SCORE_FILE")"
else
  echo "Initial real-ADDA evaluation..."
  best_score="$(eval_checkpoint "$current_checkpoint" initial)"
  if [[ "$(realpath "$current_checkpoint")" != "$(realpath -m "$BEST_CHECKPOINT")" ]]; then
    cp "$current_checkpoint" "$BEST_CHECKPOINT"
  fi
  echo "$best_score" > "$BEST_SCORE_FILE"
fi
echo "current_best_score=$best_score"
reject_count=0

for cycle in $(seq 1 "$CYCLES"); do
  name="${NAME_PREFIX}_C${cycle}"
  cycle_seed=$((SEED_BASE + cycle - 1))
  checkpoint_chunk_args=()
  if [[ "$SPECTRAL_FREQ_CHECKPOINT_CHUNKS" == "1" || "$SPECTRAL_FREQ_CHECKPOINT_CHUNKS" == "true" ]]; then
    checkpoint_chunk_args+=(--spectral_freq_checkpoint_chunks)
  fi
  correction_args=()
  if [[ "$SPECTRAL_CORRECTION_HIDDEN" != "0" ]]; then
    correction_args+=(--spectral_correction_hidden "$SPECTRAL_CORRECTION_HIDDEN")
    correction_args+=(--spectral_correction_layers "$SPECTRAL_CORRECTION_LAYERS")
    correction_args+=(--spectral_correction_scale "$SPECTRAL_CORRECTION_SCALE")
    correction_args+=(--spectral_correction_penalty "$SPECTRAL_CORRECTION_PENALTY")
  fi
  if [[ "$SPECTRAL_FREEZE_BASE" == "1" || "$SPECTRAL_FREEZE_BASE" == "true" ]]; then
    correction_args+=(--spectral_freeze_base)
  fi
  echo
  echo "=== Cycle $cycle/$CYCLES: training $name (seed=$cycle_seed) ==="

  python3 -u train_v7/train.py \
    --device "$DEVICE" \
    --name "$name" \
    --save \
    --spectral --squared_kernel \
    --freq_hidden 256 --freq_layers 5 \
    --global_hidden 256 --global_layers 3 \
    --resume "$current_checkpoint" \
    --only_shape hex_prism \
    --hex_dl_min "$HEX_DL_MIN" --hex_dl_max "$HEX_DL_MAX" \
    --grid_min "$TRAIN_GRID_MIN" --grid_max "$TRAIN_GRID_MAX" \
    --m_re_min "$M_RE" --m_re_max "$M_RE" \
    --m_im_min "$M_IM" --m_im_max "$M_IM" \
    --kd_min "$KD" --kd_max "$KD" \
    --loss "$LOSS" \
    --krylov_iters "$KRYLOV_ITERS" \
    --anchor_num_probes "$ANCHOR_NUM_PROBES" \
    --anchor_probe_chunk "$ANCHOR_PROBE_CHUNK" \
    --anchor_krylov_weight "$ANCHOR_KRYLOV_WEIGHT" \
    --anchor_probe_weight "$ANCHOR_PROBE_WEIGHT" \
    --anchor_right_probe_weight "$ANCHOR_RIGHT_PROBE_WEIGHT" \
    --spectral_truncate_radius "$RADIUS" \
    --spectral_identity_blend 1.0 \
    --spectral_freq_chunk_size "$SPECTRAL_FREQ_CHUNK_SIZE" \
    "${checkpoint_chunk_args[@]}" \
    "${correction_args[@]}" \
    --export_max_radius "$RADIUS" \
    --export_identity_blend 1.0 \
    --export_threshold_rel 1e-6 \
    --seed "$cycle_seed" \
    --num_steps "$STEPS_PER_CYCLE" \
    --lr "$LR" \
    --weight_decay 1e-4 \
    --gradient_clipping 1.0 \
    --curriculum_frac "$CURRICULUM_FRAC" \
    --val_interval 0 \
    --solve_val_interval 0 \
    --skip_final_eval \
    --log_interval 50 \
    --save_interval "$STEPS_PER_CYCLE"

  candidate="results/${name}/final_model.pt"
  echo "=== Cycle $cycle/$CYCLES: real ADDA eval ==="
  candidate_score="$(eval_checkpoint "$candidate" "cycle_${cycle}")"
  echo "candidate_score=$candidate_score best_score=$best_score"

  if is_better "$candidate_score" "$best_score"; then
    cp "$candidate" "$BEST_CHECKPOINT"
    echo "$candidate_score" > "$BEST_SCORE_FILE"
    best_score="$candidate_score"
    current_checkpoint="$BEST_CHECKPOINT"
    reject_count=0
    echo "PROMOTED: $candidate -> $BEST_CHECKPOINT"
  else
    current_checkpoint="$BEST_CHECKPOINT"
    reject_count=$((reject_count + 1))
    echo "REJECTED: keeping $BEST_CHECKPOINT"
    if [[ "$STOP_AFTER_REJECTS" != "0" && "$reject_count" -ge "$STOP_AFTER_REJECTS" ]]; then
      echo "STOP_AFTER_REJECTS reached ($reject_count); stopping early"
      break
    fi
  fi
done

echo
echo "DONE: best_score=$best_score best_checkpoint=$BEST_CHECKPOINT"
