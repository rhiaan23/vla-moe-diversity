#!/bin/bash
# CPU-only standard eval of the v2 finetune (router + VLM-LoRA only) on LIBERO tasks 0/1.
# Defaults to the most recent fully-flushed numeric checkpoint to avoid racing
# with the still-running training (which keeps overwriting checkpoints/last/).
set -euo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
# Pick the latest numbered checkpoint (001000, 002000, ...). Override via $CKPT.
CKPT_DEFAULT=$(ls -d "${REPO}"/outputs/finetune_pi0_v5_router_only_libero01/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort | tail -1)
CKPT=${CKPT:-${CKPT_DEFAULT}}
STEP=$(basename "$(dirname "${CKPT}")")
OUT=${REPO}/outputs/eval/ft_pi0_v5_router_only_libero01_step${STEP}_cpu

export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=${HOME}/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Software rendering for camera images so we don't touch the GPU at all.
export MUJOCO_GL=osmesa
mkdir -p /tmp/osmesa_fix && ln -sf /lib64/libOSMesa.so.8 /tmp/osmesa_fix/libOSMesa.so
export LD_LIBRARY_PATH=/tmp/osmesa_fix:${LD_LIBRARY_PATH:-}
# Pin to CPU; the GPU on della-adele is in use by the finetune.
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "${REPO}"

# Robosuite expects this file; create empty if missing.
ROBOSUITE_MACROS_PRIVATE=$(python -c "import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), 'macros_private.py'))" 2>/dev/null || true)
if [ -n "${ROBOSUITE_MACROS_PRIVATE}" ] && [ ! -f "${ROBOSUITE_MACROS_PRIVATE}" ]; then
  echo "FILE_LOGGING_LEVEL = None" > "${ROBOSUITE_MACROS_PRIVATE}"
fi

rm -rf "${OUT}"

echo "[eval] CKPT=${CKPT}"
echo "[eval] OUT=${OUT}"

exec python -m lerobot.scripts.lerobot_eval \
  --policy.path="${CKPT}" \
  --policy.device=cpu \
  --policy.dtype=bfloat16 \
  --policy.use_amp=false \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1]' \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --output_dir="${OUT}" \
  --job_name=eval_ft_pi0_v5_router_only_libero01_step${STEP}_cpu \
  --seed=1000 \
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
