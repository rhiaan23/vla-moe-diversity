#!/bin/bash
#SBATCH --job-name=pi0-loop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/FHEIDE/rj2807/logs/pi0_%x_%j.out
#SBATCH --error=/scratch/gpfs/FHEIDE/rj2807/logs/pi0_%x_%j.err

# Usage:
#   sbatch --export=ALL,VARIANT=moe_standard --job-name=pi0_moe_standard slurm/pi0_train_loop.slurm
# Driver: scripts/train_loop.sh handles resume logic + inner retry on crash.
# When SLURM kills the job at wall time, a fresh sbatch will resume from the
# last saved checkpoint (save_freq=2000).

set -u

mkdir -p /scratch/gpfs/FHEIDE/rj2807/logs
cd /scratch/gpfs/FHEIDE/rj2807/vla-moe-diversity

if [ -z "${VARIANT:-}" ]; then
    echo "VARIANT env var required (baseline|moe_single_expert|moe_standard|moe_diversity)" >&2
    exit 2
fi

exec bash scripts/train_loop.sh "$VARIANT" 0
