# Princeton Adroit Cluster — GPU Guide & Common Mistakes

> Sources: [Adroit System](https://researchcomputing.princeton.edu/systems/adroit) · [Getting Started Guide](https://researchcomputing.princeton.edu/get-started/guide-princeton-clusters) · [Mistakes to Avoid](https://researchcomputing.princeton.edu/get-started/mistakes-avoid)

---

## Overview

Adroit is Princeton's smaller training cluster, designed for **developing, debugging, and testing** before scaling to larger clusters like Della or Tiger. It mirrors the architecture of larger clusters, making it an ideal sandbox.

- Login: `ssh <YourNetID>@adroit.princeton.edu` (VPN required off-campus)
- Web portal: [https://myadroit.princeton.edu](https://myadroit.princeton.edu) (supports Jupyter, RStudio, MATLAB, etc.)
- Visualization node: `ssh <YourNetID>@adroit-vis.princeton.edu`

---

## GPU Hardware

### GPU Nodes

| Processor | GPU | Nodes | CPU-Cores/Node | GPUs/Node | CPU Memory | GPU Memory |
|---|---|---|---|---|---|---|
| 2.6 GHz Intel Sapphire Rapids | **A100** | 1 | 48 | 4 | 1 TB | **80 GB** each |
| 2.8 GHz Intel Ice Lake | **A100 (MIG)** | 1 | 48 | 8 | 1 TB | **20 GB** each |
| 2.6 GHz Intel Skylake | **V100** | 1 | 56 | 4 | 770 GB | **32 GB** each |

> **Note on MIG:** The 4× A100 (80 GB) node has been partitioned into 8× MIG instances of 20 GB each using NVIDIA Multi-Instance GPU. Each MIG GPU delivers ~50% of a full A100's performance.

### Visualization Node (`adroit-vis`)

- 64 CPU-cores, 512 GB RAM
- 2× A100 GPUs (80 GB each)
- Has **internet access** (unlike compute nodes)
- No job scheduler — be considerate of other users; check usage with `htop -u $USER`

---

## GPU Job Scheduling (QOS Limits)

| QOS | Time Limit | GPUs per User |
|---|---|---|
| `gpu-test` | 15 minutes | no limit |
| `gpu-short` | 4 hours | 4 |
| `gpu-medium` | 24 hours | 2 |
| `gpu-long` | 2 days | 2 |

> These are minimum limits — actual values may be higher. Run `qos` to see current limits.

Check free GPUs at any time:

```bash
shownodes -p gpu
```

---

## Running GPU Jobs — Slurm Directives

### Request any available GPU

```bash
#SBATCH --gres=gpu:1
```

### Constrain to a specific GPU type

```bash
# A100 GPU (any memory size)
#SBATCH --constraint=a100

# V100 GPU
#SBATCH --constraint=v100

# A100 with 80 GB memory specifically
#SBATCH --constraint=gpu80
```

### Example batch script

```bash
#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out

module purge
module load anaconda3/2024.2

source activate myenv
python train.py
```

> Do **not** specify `--qos` manually — let the scheduler route the job based on the resources and runtime requested.

---

## Filesystem on Adroit

| Path | Purpose | Backed Up? |
|---|---|---|
| `/home/<NetID>` | Small personal files, scripts | Yes |
| `/scratch/network/` | Job output, large datasets (~24 TB NFS) | **No** |
| `/tmp` (local on each node) | Temporary per-node scratch | **No** |
| `/projects` | Long-term, non-volatile storage | Yes |

**Key rule:** Write job output to `/scratch/network/`, then copy important results to `/projects` or `/home` after the job finishes.

Check your quota:

```bash
checkquota
```

---

## CPU Job QOS (for reference)

| QOS | Time Limit | Jobs/User | Cores/User |
|---|---|---|---|
| `test` | 15 min | 2 | 80 cores |
| `short` | 4 hours | 32 | 80 cores |
| `medium` | 24 hours | 4 | 64 cores |
| `long` | 7 days | 2 | 64 cores |

---

## Top 10 Mistakes to Avoid

### 1. Exceeding Storage Quota
Going over quota causes batch job failures, confusing error messages, and broken X11 forwarding. Run `checkquota` regularly. Request a quota increase if needed.

### 2. Running Jobs on the Login Node
The login node is **shared by all users** and is reserved for:
- Submitting jobs
- Compiling code
- Installing software
- Short tests (a few CPU-cores, a few minutes max)

Anything heavier **must** be submitted via Slurm as a batch or interactive job. Violations may result in account suspension.

### 3. Writing Job Output to `/projects`
`/projects` is slow and shared across all clusters. Writing active job output there harms other users and slows your job. Use `/scratch/gpfs/` (or `/scratch/network/` on Adroit) for output, then copy to `/projects` after completion.

### 4. Trying to Access the Internet from Compute Nodes
**Compute nodes have no internet access.** This means running jobs cannot:
- Download files
- Install pip/conda packages
- Connect to GitHub or HuggingFace Hub

Perform all downloads and package installations on the **login node** before submitting the job. For offline clusters, use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

### 5. Allocating Excessive CPU Memory
Overestimating `--mem-per-cpu` or `--mem` wastes resources, raises your queue times, and blocks other users. Profile memory usage and request accurate amounts. See the [Slurm memory guide](https://researchcomputing.princeton.edu/support/knowledge-base/memory).

### 6. Allocating More than One CPU-Core for Serial Jobs
Serial code cannot use multiple cores. Requesting extra cores wastes resources and lowers your job priority. Use [Job Arrays](https://researchcomputing.princeton.edu/support/knowledge-base/slurm#arrays) to run many serial jobs simultaneously instead.

### 7. Not Doing a Scaling Analysis Before Parallel Jobs
Before committing to a large parallel job, run a **scaling analysis** to find the optimal number of nodes, CPU-cores, or GPUs. More resources ≠ faster if communication overhead dominates. See [Choosing the Number of Nodes, CPU-cores and GPUs](https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis).

### 8. Requesting a GPU for CPU-Only Code
**This is one of the most common mistakes.** Only code explicitly written for GPU execution benefits from a GPU. Requesting a GPU for CPU-only code:
- Does **not** speed up the job
- Increases your queue time
- Wastes resources
- Lowers priority for your next submission

Read the documentation of any software you use to verify GPU support.

### 9. Using the System GCC When a Newer Version Is Needed
The default system GCC may be outdated. Check it with:

```bash
gcc --version
```

Load a newer version with:

```bash
module avail gcc-toolset
module load gcc-toolset/<version>
```

### 10. Wasting Time on Solvable Problems Alone
Research Computing staff can help with software issues, job scheduler problems, and debugging. If you're stuck, use the [How to Get Help](https://researchcomputing.princeton.edu/support) page and attend [workshops and training sessions](https://researchcomputing.princeton.edu/learn/workshops-live-training).

---

## Quick Reference Commands

```bash
# Check GPU node availability
shownodes -p gpu

# Check storage quota
checkquota

# Check current QOS limits
qos

# Check node details
snodes

# Submit a batch job
sbatch myjob.sh

# Interactive GPU session
salloc --gres=gpu:1 --time=01:00:00 --mem=16G

# Monitor your running jobs
squeue -u $USER

# Cancel a job
scancel <JOBID>

# Check resource usage on vis node
htop -u $USER
```

---

## Relevant Links

- [Adroit System Page](https://researchcomputing.princeton.edu/systems/adroit)
- [Slurm Job Scheduler](https://researchcomputing.princeton.edu/support/knowledge-base/slurm)
- [GPU Computing](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing)
- [Data Storage](https://researchcomputing.princeton.edu/support/knowledge-base/data-storage)
- [Scaling Analysis Guide](https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis)
- [Memory Allocation](https://researchcomputing.princeton.edu/support/knowledge-base/memory)
- [Environment Modules](https://researchcomputing.princeton.edu/support/knowledge-base/modules)
- [MyAdroit Web Portal](https://myadroit.princeton.edu)
- [How to Get Help](https://researchcomputing.princeton.edu/support)
