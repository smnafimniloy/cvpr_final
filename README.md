# Medicinal Leaf Classification

Classification of 16 medicinal plant leaf health conditions across 5 species (Aloe Vera, Azadirachta Indica, Hibiscus Rosa Sinensis, Kalanchoe Pinnata, Piper Betle) using pretrained lightweight backbones, a custom sub-1M-parameter CNN, and a multi-model ensemble.

## Overview

This project compares three approaches for medicinal leaf disease/condition classification:

1. **Pretrained baselines** — MobileNetV3-Small, ShuffleNetV2, EfficientNet-B0, each fine-tuned under full, partial, and classifier-only (linear probe) regimes.
2. **Custom lightweight CNN** — a from-scratch, sub-1M-parameter model (32→64→128 channels) designed for low-memory, low-compute deployment.
3. **Weighted-average ensemble** — combining the four best-performing baseline configurations for maximum accuracy.

The pipeline is built around a **leakage-aware, original-image-first data split**, addressing a common but under-scrutinized risk when working with pre-augmented datasets.

## Dataset

- **Source:** [Medicinal Plant Leaf Image Dataset](https://data.mendeley.com/datasets/89rfgtxbdc/1) (Mendeley Data, CC BY 4.0)
- **Classes:** 16 leaf-condition classes across 5 species
- **Images used:** 1,323 original images (the dataset's 14,677 pre-augmented images are used only in the leakage ablation, not the main pipeline)
- **Class distribution:** 52–153 images per class (moderately imbalanced)

## Key Results

| Model | Train Mode | Accuracy | Macro-F1 | Params | Size (MB) | Inference (ms/img) |
|---|---|---|---|---|---|---|
| **Ensemble (4-model)** | weighted average | **93.50%** | **0.9378** | 8.10M | 30.91 | 1.196 |
| EfficientNet-B0 | full | 92.00% | 0.9197 | 4.03M | 15.37 | 0.367 |
| ShuffleNetV2 | full | 92.00% | 0.9158 | 1.27M | 4.85 | 0.180 |
| MobileNetV3-Small | full | 88.50% | 0.8845 | 1.53M | 5.85 | 0.464 |
| **Custom LightCNN** | full | 81.00% | 0.7637 | **95.8K** | **0.365** | **0.017** |

Full comparison table (all 16 configurations, including ablations) is in [`results/comparison_table_final.csv`](results/comparison_table_final.csv).

## Ablation Studies

| Ablation | Configuration | Accuracy | Macro-F1 |
|---|---|---|---|
| Loss weighting | Weighted CE loss | 79.50% | 0.7458 |
| | Unweighted CE loss | 69.50% | 0.6161 |
| Split strategy | Original-image-first (ours) | 81.00% | 0.7637 |
| | Random split (augmented pool, leakage risk) | 84.73% | 0.8442 |
| Channel width | LightCNN (32→64→128) | 79.50% | 0.7458 |
| | WiderCNN (64→128→256) | 84.00% | 0.7999 |

The split-strategy ablation shows that naive random splitting on pre-augmented data **inflates** apparent accuracy by ~8 macro-F1 points due to near-duplicate images leaking across train/test — validating the need for our leakage-aware protocol.

## Project Structure

Note: the dataset itself is **not included in this repo** — download it separately from Mendeley (link in [Dataset](#dataset) section) and organize it locally as described below before running any notebook.

```
.
├── LightcustomCNN.ipynb                     # custom CNN training
├── ablationLeakage.ipynb                    # split/leakage ablation
├── ablationWeightFalse.ipynb                # unweighted loss ablation
├── ablationWeightTrue.ipynb                 # weighted loss ablation
├── ablationWiderCnn.ipynb                   # channel-width ablation
├── baseline-efficientnet_b0-Full.ipynb
├── baseline-efficientnet_b0-Partial.ipynb
├── baseline-efficientnet_b0-classifierOnly.ipynb
├── baseline-mobileV3-Full.ipynb
├── baseline-mobileV3-Partial.ipynb
├── baseline-mobileV3-classifierOnly.ipynb
├── baseline-shufflenetV2-Full.ipynb
├── baseline-shufflenetV2-Partial.ipynb
├── baseline-shufflenetV2-classifierOnly.ipynb
├── comparison.ipynb                         # final results comparison + visualizations
├── ensemble.ipynb                           # ensemble search + evaluation
├── splitScript.py                           # original-image-first stratified split script
├── splitReport.txt                          # split output log
├── results/
│   └── <run_tag>/
│       ├── best_model.pth
│       ├── last_checkpoint.pth
│       ├── history.json
│       ├── results.json
│       ├── curves.png
│       ├── confusion_matrix.png
│       └── per_class_report.csv
└── README.md
```

## Setup

```bash
conda create -n cvpr python=3.10 -y
conda activate cvpr

# PyTorch (CUDA build — adjust cu121 to match your driver via `nvidia-smi`)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies
conda install -c conda-forge scikit-learn numpy matplotlib pillow pandas seaborn tqdm ipywidgets jupyterlab -y
pip install thop
```

Verify GPU is detected:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Usage

1. **Download the dataset** — get it from [Mendeley](https://data.mendeley.com/datasets/89rfgtxbdc/1) (not included in this repo) and place the original images locally under a folder such as `Medicinal Plant Leaf Health Original Dataset/` (16 class subfolders), matching the path expected in the notebooks' config cells.
2. **Split the dataset** — run `splitScript.py` to generate the stratified, original-image-first `train/val/test` folders (see `splitReport.txt` for an example output log).
3. **Train baselines** — run each `baseline-<model>-<Mode>.ipynb` notebook (efficientnet_b0, mobileV3, shufflenetV2 × Full/Partial/classifierOnly).
4. **Train the custom CNN** — run `LightcustomCNN.ipynb`.
5. **Run ablations** — `ablationWeightTrue.ipynb` / `ablationWeightFalse.ipynb` (loss weighting), `ablationLeakage.ipynb` (split strategy), `ablationWiderCnn.ipynb` (channel width).
6. **Build the ensemble** — run `ensemble.ipynb` to search combinations and evaluate the best one.
7. **Generate final comparison** — run `comparison.ipynb` to produce the consolidated table and plots.

Each training notebook supports **checkpoint resumption** — if interrupted, rerunning picks up from the last completed epoch automatically.

## Model Architecture (Custom LightCNN)

```
Input (3×128×128)
  → Conv(3→32, 3×3) + BatchNorm + ReLU + MaxPool
  → Conv(32→64, 3×3) + BatchNorm + ReLU + MaxPool
  → Conv(64→128, 3×3) + BatchNorm + ReLU + MaxPool
  → AdaptiveAvgPool(1×1)
  → Dropout(0.3) + Linear(128→16)
```

**95,760 parameters | 0.365 MB | 0.017 ms/image inference (RTX 3050)**

## Evaluation Metrics

- Accuracy, macro-averaged precision/recall/F1
- Per-class precision/recall/F1 and confusion matrices
- Model size (MB), parameter count, GFLOPs (via THOP), measured inference latency

## Hardware

Trained and evaluated on an NVIDIA GeForce RTX 3050 (8GB VRAM).

## License

Dataset: CC BY 4.0 ([Mendeley Data](https://data.mendeley.com/datasets/89rfgtxbdc/1)).
Code: add your preferred license here (e.g., MIT).

## Citation

If you use this work, please cite both the dataset and this repository:

**Dataset:**
```
Medicinal Plant Leaf Image Dataset, Mendeley Data, DOI: 10.17632/89rfgtxbdc.1
```

**This work:**
```
S M Nafim Niloy, "Medicinal Leaf Classification", GitHub repository.
https://github.com/smnafimniloy/cvpr_final
```

If this work is published as a paper, please cite instead:
```
[Author(s)]. "[Paper Title]." [Venue/Conference], [Year]. [DOI/URL when available]
```
*(Update this citation block once the associated paper is published.)*
