# Recursive SNE: Fast Prototype-Based t-SNE for Large-Scale and Online Data (MATLAB Implementation)

This repository contains the **MATLAB implementation** of the algorithms described in:

> Aghasanli, A., & Angelov, P. (2025). Recursive SNE: Fast prototype-based t-SNE for large-scale and online data. *Transactions on Machine Learning Research (TMLR)*. [OpenReview page](https://openreview.net/forum?id=7wCPAFMDWM)

**Recursive SNE (RSNE)** provides fast, incremental 2-D (or 3-D) visualizations by seeding new points near **cluster centroids** learned from an initial batch, then applying a few light t-SNE-style refinement steps.

This repository includes two variants and ready-to-run benchmarks:

- **i-RSNE** — incremental; embeds **one point at a time** (sequential)
- **Bi-RSNE** — batch-incremental; embeds **batches in parallel** (typically faster)
- **Baseline** — original **Barnes-Hut t-SNE** (MATLAB's built-in `tsne`)

> **Note:** This is a **MATLAB implementation** of the original Python version with a **hybrid workflow** for feature extraction (Python) + algorithm benchmarking (MATLAB).

---

## Key Features

- MATLAB implementation (no Python dependencies for core algorithms)
- Hybrid workflow: Python for CLIP/DINOv2 feature extraction, MATLAB for benchmarking
- Support for both synthetic and real-world datasets (CIFAR-10/100)
- Comprehensive evaluation metrics (Silhouette, Davies-Bouldin)
- Professional visualization tools
- No external MATLAB toolboxes required beyond Statistics and Machine Learning

---

## 📂 Repository Structure
```
Angelov-Aghasanli-Ashwin/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
│
├── matlab/                            # MATLAB implementation
│   ├── README.md                      # MATLAB-specific documentation
│   ├── core/                          # Core algorithm classes
│   │   ├── IRSNE.m                   # i-RSNE (point-by-point) class
│   │   └── BiRSNE.m                  # Bi-RSNE (batch) class
│   ├── utils/                         # Utility functions
│   │   ├── readNPY_simple.m          # Read NumPy .npy files
│   │   ├── writeNPY.m                # Write NumPy .npy files (optional)
│   │   ├── stream_batches.m          # Create batches for streaming
│   │   ├── clustering_quality.m      # Silhouette & Davies-Bouldin metrics
│   │   └── scatter_embedding.m       # 2D scatter plot visualization
│   ├── data/                          # Data generation utilities
│   │   ├── make_blob_dataset.m       # Generate synthetic blob clusters
│   │   └── split_initial_and_stream.m # Split data for streaming
│   ├── benchmarks/                    # Benchmarking scripts
│   │   ├── run_benchmark.m           # Benchmark on synthetic blobs
│   │   └── run_benchmark_features.m  # Benchmark on precomputed features
│   └── examples/                      # Example usage scripts
│       ├── example_synthetic.m        # Synthetic data demo
│       └── example_cifar10.m          # CIFAR-10 demo
```

---

### Prerequisites

**MATLAB Requirements:**
- MATLAB R2020a or later
- Statistics and Machine Learning Toolbox
- (Optional) Deep Learning Toolbox for MATLAB-native feature extraction

**Python Requirements (for feature extraction only):**
```bash
pip install numpy torch torchvision scikit-learn tqdm matplotlib
pip install git+https://github.com/openai/CLIP.git  # For CLIP features
```

> **Note:** Python is only needed for deep feature extraction. All algorithm benchmarking runs in pure MATLAB.

---

### Installation
```bash
# Clone the repository
git clone https://github.com/ashwin0306/Angelov-Aghasanli-Ashwin
cd Angelov-Aghasanli-Ashwin

# Add MATLAB paths (run in MATLAB)
cd matlab
addpath('core');
addpath('utils');
addpath('data');
addpath('benchmarks');
addpath('examples');
```

---

## 🎲 What Recursive SNE Does

1. **Seed with clusters:** Run KMeans on an initial batch and compute a 2-D t-SNE map for that same batch
2. **Store cluster statistics:** For each cluster, keep a high-D centroid, a low-D centroid, and a spread estimate
3. **Embed new data:**
   - **i-RSNE:** for each new point → find nearest high-D cluster → initialize near its low-D mean → apply 1-few (P-Q) nudges → update stats
   - **Bi-RSNE:** same idea, but **vectorized for a whole batch** (parallel updates; faster)
4. **Evaluate:** Silhouette score (higher is better), Davies-Bouldin index (lower is better)

---

## 📊 Benchmark on Synthetic Blobs

Sanity-check the methods without external data.

### MATLAB Command
```matlab
% Navigate to benchmarks directory
cd matlab/benchmarks

% Run benchmark with default parameters
run_benchmark('samples', 8000, ...
              'features', 20, ...
              'centers', 12, ...
              'split', 0.5, ...
              'K', 60, ...
              'batch', 800, ...
              'iters', 2, ...
              'plots', true, ...
              'plot_prefix', '../../results/synthetic_blobs');
```

Optional PNG plots saved to `results/` directory with `--plots true`.

---

## 🖼️ Replicating Paper Results with Deep Features

### Step 1: Extract Features (Python - One-time Operation)

#### Extract DINOv2 features on CIFAR-100 train set
```bash
cd python

python feature_extraction.py \
    --model dinov2_vitl14 \
    --dataset cifar100 \
    --split train \
    --batch-size 128 \
    --output-prefix ../data/cifar100_dino_train
```

#### Extract CLIP features on CIFAR-10 test set
```bash
python feature_extraction.py \
    --model clip_vitl14 \
    --dataset cifar10 \
    --split test \
    --batch-size 128 \
    --output-prefix ../data/cifar10_clip_test
```

**Output files:**
```
data/
├── cifar100_dino_train_features.npy    # Shape: (N, 1024) for DINOv2
├── cifar100_dino_train_labels.npy      # Shape: (N,)
├── cifar10_clip_test_features.npy      # Shape: (N, 768) for CLIP
└── cifar10_clip_test_labels.npy        # Shape: (N,)
```

---

```matlab
cd matlab/benchmarks

% Benchmark on CIFAR-100 with DINOv2 features
run_benchmark_features('prefix', '../../data/cifar100_dino_train', ...
                      'split', 0.5, ...
                      'K', 200, ...
                      'batch', 1000, ...
                      'iters', 2, ...
                      'plots', true, ...
                      'plot_prefix', '../../results/cifar100_dino');
```

#### Using explicit paths
```matlab
% Benchmark on CIFAR-10 with CLIP features
run_benchmark_features('features', '../../data/cifar10_clip_test_features.npy', ...
                      'labels', '../../data/cifar10_clip_test_labels.npy', ...
                      'split', 0.3, ...
                      'K', 100, ...
                      'batch', 800, ...
                      'iters', 2, ...
                      'plots', true, ...
                      'plot_prefix', '../../results/cifar10_clip');
```
## When to Use Which Variant?

- **Bi‑RSNE** — prefer for speed and stability (parallel batch updates).  
- **i‑RSNE** — for truly one‑by‑one streams.
  
Both variants target similar asymptotic costs; **Bi-RSNE is typically faster** due to vectorized processing.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{
aghasanli2025recursive,
title={Recursive {SNE}: Fast Prototype-Based t-{SNE} for Large-Scale and Online Data},
author={Agil Aghasanli and Plamen P Angelov},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=7wCPAFMDWM},
note={}
}
