# cearlab-phase2  
This repository contains the code used to run the CEAR Lab Phase 2 experiments on history-dependent perspective latents and generate the main figures.  
Related paper: [Same World, Differently Given: History-Dependent Perceptual Reorganization in Artificial Agents](https://arxiv.org/abs/2604.04637)  

The current public version focuses on:
- Phase 1 training of the frozen behavioral backbone
- Phase 2 training under perturbation, mixed-history, and ablation conditions


## Repository layout

```text
cear_pilot/
  envs/          Environment definitions for Phase 1 and Phase 2
  models/        Encoder, world latent, state head, policy, decoder, and agent
  training/      Training entry points for Phase 1 and Phase 2
  analysis/      Probe analysis and final paper figure generation
  experiments/   Optional rollout collection utilities

run_phase2.sh    Main script for Phase 2 experiment sweeps
environment.yml  Conda environment specification
README.md
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate cearlab-phase2
```

Install PyTorch separately if needed:  
(The code has been tested with the cu128 wheel and is expected to be compatible with newer CUDA setups)  

```bash
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify the installation with:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Requirements

This repository assumes access to precomputed **Phase 1 checkpoints**. By default, `run_phase2.sh` expects them at:

```text
outputs/runs/p1_s0/ckpt_final.pt
outputs/runs/p1_s1/ckpt_final.pt
outputs/runs/p1_s2/ckpt_final.pt
outputs/runs/p1_s3/ckpt_final.pt
outputs/runs/p1_s4/ckpt_final.pt
```

## Running Phase 2

The main entry point is:

```bash
bash run_phase2.sh cuda all
```

Supported modes:

* `all` - full sweep + mixed-history + ablation. 
* `sweep` - perturbation-count sweep
* `mixed` - mixed-history schedules
* `ablation` - adaptive vs fixed update regimes
* `smoke` - small smoke-test version (still iterates all experiments)  
*  Parallelism can be controlled with the `MAX_JOBS` environment variable.

Examples:

```bash
bash run_phase2.sh cuda smoke
bash run_phase2.sh cuda all
MAX_JOBS=2 bash run_phase2.sh cuda mixed
```

## Generating paper figures

Final paper figures are generated with:

```bash
python -m cear_pilot.analysis.paper_figures_v3 \
  --sweep_root outputs/phase2_all_YYYYMMDD_HHMMSS \
  --mixed_root outputs/phase2_all_YYYYMMDD_HHMMSS \
  --ablation_root outputs/phase2_all_YYYYMMDD_HHMMSS \
  --outdir outputs/phase2_all_YYYYMMDD_HHMMSS/paper_figures_v3
```

Probe-based representation analysis is generated per run with:

```bash
python -m cear_pilot.analysis.probe_representation \
  --run_dir <run_directory> \
  --device cuda
```

## Data availability

The full raw outputs and checkpoints are not included in this repository because they are large (>10GB).  
This public version includes the code needed to reproduce the experiments, assuming access to the required Phase 1 checkpoints and sufficient compute resources.

