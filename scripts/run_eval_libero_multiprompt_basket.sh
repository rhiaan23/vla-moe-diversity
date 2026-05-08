#!/usr/bin/env bash
# Evaluate on LIBERO-10 task 0 (living-room table with basket + seven groceries).
# Uses a richer prompt that asks for all groceries sequentially.
# Benchmark success stays the official BDDL goal: alphabet soup AND tomato sauce in the basket only.
#
# Requires: lerobot deps + libero, MUJOCO_GL=egl (headless), etc.
#
# Usage:
#   ./scripts/run_eval_libero_multiprompt_basket.sh /path/to/pretrained_model [output_dir]
#
# Horizon: add "--env.episode_length=900" before output_dir if evaluating longer behavior needs more steps.
set -uo pipefail

CKPT="${1:?Usage: $0 /path/to/pretrained_model [output_dir]}"
OUT="${2:-./outputs/eval_libero_mp_basket}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec python -m lerobot.scripts.lerobot_eval \
  "--policy.path=$CKPT" \
  "--env.type=libero" \
  "--env.task=libero_10" \
  '--env.task_ids=[0]' \
  "--env.instruction_preset=living_room_scene2_clear_table_to_basket" \
  "--eval.batch_size=1" \
  "--eval.n_episodes=10" \
  "--policy.device=cuda" \
  "--output_dir=$OUT" \
  "--job_name=libero_lr_scene2_clear_prompt"
