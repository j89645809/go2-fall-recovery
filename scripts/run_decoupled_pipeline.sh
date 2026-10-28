#!/usr/bin/env bash
set -euo pipefail

cd /home/zy/old_data/unitree_rl_mjlab
PY=/home/zy/old_data/miniconda3/envs/unitree_rl_mjlab/bin/python
PRE=run_logs/decoupled_native_20260916
POST=run_logs/decoupled_native_posttrain_20260916
CHECKPOINT=logs/rsl_rl/go2_recovery_frnet_aligned/2026-09-14_16-56-16_targeted_full_joint_stairs_slope_1000/model_13798.pt

wait_for_shards() {
  local directory="$1"
  while true; do
    local live=0
    for pose in front back left right; do
      if [[ -f "$directory/$pose.pid" ]]; then
        pid=$(cat "$directory/$pose.pid")
        if kill -0 "$pid" 2>/dev/null; then live=$((live + 1)); fi
      fi
    done
    if (( live == 0 )); then break; fi
    sleep 60
  done
  for pose in front back left right; do
    test -s "$directory/$pose.json"
  done
}

launch_eval() {
  local directory="$1"
  local checkpoint="$2"
  mkdir -p "$directory"
  for spec in front:1 back:2 left:3 right:2; do
    pose=${spec%%:*}; gpu=${spec##*:}
    rm -f "$directory/$pose.json" "$directory/$pose.json.partial" \
      "$directory/$pose.log" "$directory/$pose.pid"
    nohup env PYTHONPATH=/home/zy/old_data/unitree_rl_mjlab \
      CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl \
      "$PY" scripts/decoupled_native_eval.py \
      --checkpoint "$checkpoint" \
      --output "$directory/$pose.json" \
      --poses "$pose" --per-stratum 20 --horizon 600 --device cuda:0 \
      > "$directory/$pose.log" 2>&1 < /dev/null &
    echo $! > "$directory/$pose.pid"
  done
}

wait_for_shards "$PRE"
"$PY" scripts/aggregate_decoupled.py \
  --input-dir "$PRE" --output "$PRE/aggregate.json" \
  > "$PRE/aggregate.log" 2>&1

training_required=$(
  "$PY" - <<'PY'
import json
print(int(json.load(open('run_logs/decoupled_native_20260916/aggregate.json'))['training_required']))
PY
)

if [[ "$training_required" == "0" ]]; then
  printf 'BASELINE_ALREADY_PASSES\n' > "$PRE/pipeline_status.txt"
  exit 0
fi

printf 'TRAINING_STARTED\n' > "$PRE/pipeline_status.txt"
env PYTHONPATH=/home/zy/old_data/unitree_rl_mjlab \
  PYTHON_FRNET_TARGETED_RECOVERY=1 \
  "$PY" scripts/continue_targeted_decoupled.py --iterations 400 --gpus 1 2 \
  > "$PRE/targeted_training.log" 2>&1

run_dir=$(find logs/rsl_rl/go2_recovery_frnet_aligned -maxdepth 1 -type d \
  -name '*decoupled_allfolded_right_edge_replay_400' -print0 | xargs -0 ls -td | head -1)
checkpoint=$(find "$run_dir" -maxdepth 1 -type f -name 'model_*.pt' -print0 \
  | xargs -0 ls -t | head -1)
test -s "$checkpoint"
printf '%s\n' "$checkpoint" > "$PRE/posttrain_checkpoint.txt"

launch_eval "$POST" "$checkpoint"
wait_for_shards "$POST"
"$PY" scripts/aggregate_decoupled.py \
  --input-dir "$POST" --output "$POST/aggregate.json" \
  > "$POST/aggregate.log" 2>&1
printf 'POSTTRAIN_EVALUATION_COMPLETE\n' > "$PRE/pipeline_status.txt"
