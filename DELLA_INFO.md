# Della Cluster — Princeton Research Computing

> Source: https://researchcomputing.princeton.edu/systems/della

## Overview

Della is a general-purpose HPC cluster for running serial and parallel production jobs. It features both CPU and GPU nodes and runs **RHEL 9** (Springdale Linux 9).

---

## Access

### SSH Login Nodes

| Purpose | Hostname |
|---|---|
| CPU or GPU jobs (RHEL 9) | `della.princeton.edu` |
| GPU jobs | `della-gpu.princeton.edu` |
| Grace Hopper (GH200) | `della-gh.princeton.edu` |
| Visualization | `della-vis1.princeton.edu` / `della-vis2.princeton.edu` |

```bash
ssh <YourNetID>@della.princeton.edu
ssh <YourNetID>@della-gpu.princeton.edu
```

> VPN required from off-campus: https://www.princeton.edu/vpn

### Web Portal (MyDella)

Interactive jobs (Jupyter, RStudio, MATLAB, Stata) without the command line:

```
https://mydella.princeton.edu
```

---

## Filesystems

- `/scratch/gpfs` — High-performance parallel scratch (use for job I/O)
- `/projects` — Project storage
- **Globus endpoint**: "Princeton Della /scratch/gpfs"

**Maintenance window:** Second Tuesday of every month, ~6 AM – 2 PM. `/scratch/gpfs` and `/projects` are also down.

---

## Hardware

### CPU Nodes

| Processor | Nodes | Cores/Node | Memory/Node | Max ISA |
|---|---|---|---|---|
| 2.4 GHz AMD EPYC 9654 | 55 | 192 | 1500 GB | AVX-512 |
| 2.8 GHz Intel Cascade Lake | 64 | 32 | 190 GB | AVX-512 |
| 3.1 GHz Intel Cascade Lake | 24 | 40 | 380 GB | AVX-512 |
| 2.1 GHz Intel Sapphire Rapids | 42 | 96 | 1000 GB | AVX-512 |
| 2.5 GHz Intel Emerald Rapids | 18 | 64 | 1500 GB | AVX-512 |

### Large-Memory Nodes (CSML, available to all users)

| Nodes | Memory/Node | Cores/Node |
|---|---|---|
| 1 | 1510 GB | 48 |
| 1 | 2000 GB | 56 |
| 10 | 3080 GB | 96 |
| 3 | 6150 GB | 96 |

Large-memory nodes are allocated automatically when you request more memory than available on regular nodes.

### GPU Nodes

| Processor | Nodes | Cores/Node | CPU Mem | GPUs/Node | GPU Type |
|---|---|---|---|---|---|
| 2.6 GHz AMD EPYC Rome | 20 | 128 | 768 GB | 2 | A100 (40 GB) |
| 2.8 GHz Intel Ice Lake | 59 | 48 | 1000 GB | 4 | A100 (80 GB) |
| 2.8 GHz Intel Ice Lake | 10 | 48 | 1000 GB | 8 | MIG A100 |
| 2.8 GHz Intel Ice Lake | 2 | 48 | 1000 GB | 28 | MIG A100 |
| 2.8 ARM Neoverse-V2 | 1 | 72 | 575 GB | 1 | GH200* |
| 2.1 GHz Intel Sapphire Rapids | 42 | 96 | 1000 GB | 8 | H100 (80 GB)** |
| 2.5 GHz Intel Emerald Rapids | 18 | 64 | 1500 GB | 8 | H200 (141 GB)+ |

\* Grace Hopper — experimental  
\*\* PLI members only  
\+ AI Lab members only

---

## Job Scheduling (Slurm)

All jobs must go through the **Slurm scheduler**. Do not specify `--qos` manually; Slurm assigns it based on requested time.

### CPU QOS Tiers

| QOS | Time Limit | Max Jobs/User | Max Cores/User |
|---|---|---|---|
| test | 61 minutes | 2 | [30 nodes] |
| short | 24 hours | 300 | 300 cores |
| medium | 72 hours | 100 | 250 cores |
| vlong | 144 hours (6 days) | 40 | 160 cores |

### GPU QOS Tiers

| QOS | Time Limit | Max Jobs/User | Max Nodes/User | Max GPUs/User |
|---|---|---|---|---|
| gpu-test | 61 minutes | 2 | no limit | no limit |
| gpu-short | 24 hours | 30 | 30 | 35 |
| gpu-medium | 72 hours | 24 | 24 | 24 |
| gpu-long | 144 hours | 7 | 16 | 16 |

```bash
qos    # show current QOS limits
```

---

## GPU Job Guide

### Selecting GPU Memory

```bash
#SBATCH --constraint=gpu40   # 40 GB A100
#SBATCH --constraint=gpu80   # 80 GB A100
```

### MIG GPUs (10 GB, lowest queue time — use when possible)

Eligible when: single GPU, single CPU-core, <32 GB CPU mem, <10 GB GPU mem.

```bash
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
```

Interactive:
```bash
salloc --nodes=1 --ntasks=1 --time=60:00 --gres=gpu:1 --partition=mig
```

### Excluding MIG GPUs

```bash
#SBATCH --constraint="nomig&gpu40"
#SBATCH --constraint="nomig&gpu80"
```

### Useful GPU Commands

```bash
gfree              # free GPUs by model
gpudash --me       # your GPU utilization (last hour)
shownodes -p gpu   # node hardware specs
shownodes -p mig   # MIG node availability
```

### Compiling CUDA (A100 = sm_80, H100 = sm_90)

```bash
module load cudatoolkit/12.8
nvcc -O3 -arch=sm_80 -o myapp myapp.cu
```

### CUDA MPS (multi-process GPU sharing)

```bash
#SBATCH --gpu-mps
```

---

## CPU Architecture Notes

### Targeting Specific CPU Types

```bash
#SBATCH --constraint=amd    # AMD EPYC 9654 (192 cores, 1.5 TB)
#SBATCH --constraint=intel  # Intel nodes
```

### Illegal Instruction Errors

Compiling with `-march=native` on login nodes can produce binaries that fail on nodes with a lower instruction set. To fix:
- Remove `-xHost` / `-march=native` flags, or
- Constrain to nodes with matching ISA

For GPU jobs, to avoid AMD/Intel CPU mismatch:
```bash
#SBATCH --constraint="intel&gpu40"
#SBATCH --constraint="intel&gpu80"
```

### AMD Compiler Toolchain (RHEL 9 AMD nodes)

```bash
module load aocc/5.0.0
module load aocl/aocc/5.0.0
clang++ -Ofast -march=native -o mycode mycode.cpp
```

---

## Specialized Partitions

### PLI (Princeton Language and Intelligence) — H100 GPUs

- 336 H100 SXM GPUs (42 nodes × 8), 80 GB each, NVLink + NDR Infiniband
- Check membership: `getent group pli`

```bash
#SBATCH --partition=pli-c           # core PLI members
#SBATCH --partition=pli             # campus PLI members (+ --account=<ACCOUNT>)
#SBATCH --partition=pli-lc          # large campus PLI members

shownodes -p pli-c
```

### AI Lab — H200 GPUs

- 144 H200 PCIe GPUs (18 nodes × 8), 141 GB each
- Max 8 CPU-cores per GPU

```bash
#SBATCH --partition=ailab
```

### Grace Hopper (GH200) — Experimental

```bash
ssh <YourNetID>@della-gh.princeton.edu
#SBATCH --partition=grace
```

---

## Visualization Nodes

No Slurm scheduler — connect directly via SSH. Use for GUIs, large data downloads, and tasks unsuitable for login nodes.

| Node | CPU-Cores | Memory | GPU |
|---|---|---|---|
| della-vis1 | 80 | 1 TB | 1× A100 40 GB |
| della-vis2 | 28 | 256 GB | 4× P100 16 GB |

Both nodes have internet access.

```bash
ssh <YourNetID>@della-vis1.princeton.edu
```

---

## Rebuilding Software for RHEL 9

- C/C++/Fortran: recompile from source
- MPI codes (Python, R, Julia): recompile from source
- R: reinstall packages (`module load R/4.4.2`)
- Python (pip): reinstall if built-from-source packages were used
- Python (conda): likely works without changes
- MATLAB: no action needed

---

## Useful Commands

```bash
shownodes          # all node hardware info
shistory -j        # recent job node types (single-node jobs)
qos                # current QOS limits
gfree              # free GPU count by model
gpudash            # GPU utilization dashboard
jobstats           # memory/GPU usage of past jobs
htop -u $USER      # check your processes (vis nodes)
```

---

## Acknowledgement Wording

> "The author(s) are pleased to acknowledge that the work reported on in this paper was substantially performed using the Princeton Research Computing resources at Princeton University. Princeton Research Computing is a consortium of groups including the Princeton Institute for Computational Science and Engineering (PICSciE) and Research Computing at Princeton University."
