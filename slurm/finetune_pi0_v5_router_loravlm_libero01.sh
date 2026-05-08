#!/bin/bash
# Pi0 v5 finetune: freeze experts, train router + IO heads + LoRA-on-VLM,
# on LIBERO tasks 0/1 with the dataset's original prompts.
#
# Usage on della-fongpu (H100 NVL):
#   ssh della-fongpu
#   module load anaconda3/2024.2 && conda activate lerobot3
#   nohup bash slurm/finetune_pi0_v5_router_loravlm_libero01.sh \
#     > /scratch/gpfs/EYSENBACH/ss5822/logs/ft_pi0_v5_router_loravlm_libero01.out 2>&1 &
set -euo pipefail

REPO=/scratch/gpfs/EYSENBACH/ss5822/vla-moe-diversity
CKPT=${REPO}/checkpoints/pi0_moe_whole_v5_h100/checkpoints/020000/pretrained_model
OUT=${REPO}/outputs/finetune_pi0_v5_router_loravlm_libero01

# lerobot3's user-site .pth already adds ${REPO}/src to sys.path; no override needed.
export HF_LEROBOT_HOME=/scratch/gpfs/EYSENBACH/ss5822/data
export HF_HOME=${HOME}/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
rm -rf "${OUT}"

# Episodes in libero_10 with task_index in {0, 1} (74 episodes total).
EPISODES='[0,1,4,5,11,18,19,21,22,33,37,52,58,62,68,85,88,105,107,108,110,113,114,121,125,129,130,138,146,152,157,167,168,170,176,190,197,206,207,210,211,212,216,220,224,231,233,235,236,242,247,248,249,252,257,264,267,271,295,301,302,307,309,315,322,323,325,343,346,355,356,362,367,369]'

exec python -m lerobot.scripts.lerobot_train \
  --policy.path="${CKPT}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --policy.lora_vlm=true \
  --peft.method_type=LORA \
  --peft.r=8 \
  --dataset.repo_id=libero_10 \
  --dataset.root=/scratch/gpfs/EYSENBACH/ss5822/data/libero_10 \
  --dataset.episodes="${EPISODES}" \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.wrist_image": "observation.images.camera2"}' \
  --output_dir="${OUT}" \
  --job_name=ft_pi0_v5_router_loravlm_libero01 \
  --batch_size=8 \
  --num_workers=4 \
  --steps=5000 \
  --save_freq=1000 \
  --log_freq=50 \
  --seed=1003 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=vla-moe-diversity \
  --wandb.notes="pi0 v5 finetune, experts frozen, router + IO heads + VLM-LoRA, libero tasks 0/1, original prompts"
