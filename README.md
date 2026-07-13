# Efficient Multi-Scale Visual Saliency Prediction

Deep Learning course project implemented in PyTorch and Google Colab.

## Research question

Can a compact multi-scale CNN preserve visual-saliency prediction quality while reducing inference cost, and does it remain reliable on off-centre scenes where the SALICON centre prior is less useful?

## Task

The input is an RGB natural image. The output is a dense saliency heatmap representing the predicted spatial distribution of human visual attention.

This project addresses fixation prediction, not salient-object segmentation and not explanation methods such as Grad-CAM.

## Dataset

The project uses the SALICON dataset.

The dataset is not included in this repository because of its size. It is stored separately in Google Drive and loaded from Colab.

## Models

- `MeanMap`: non-learned spatial-prior baseline.
- `Light-S`: MobileNetV2 encoder with a single-scale decoder.
- `Light-M`: MobileNetV2 encoder with a compact multi-scale decoder.
- `Heavy-M`: ResNet-18 encoder with the same multi-scale decoder.

## Notebook-only workflow

The project intentionally uses multiple Google Colab notebooks instead of separate Python modules.

Planned notebooks:

1. `00_dataset_preparation.ipynb`
2. `01_shared_project_code.ipynb`
3. `02_meanmap_and_metrics.ipynb`
4. `03_light_single.ipynb`
5. `04_light_multi.ipynb`
6. `05_heavy_multi.ipynb`
7. `06_final_evaluation.ipynb`

Reusable classes and functions will be placed in `01_shared_project_code.ipynb` and loaded by the experiment notebooks using `%run`.

## Repository contents

```text
notebooks/       Colab notebooks
splits/          Deterministic dataset splits and manifests
documentation/   Project guides and development notes
report/          Final report sources, figures and tables
External storage policy

Google Drive stores:

the SALICON dataset;
model checkpoints;
predictions;
large figures;
experiment outputs;
backups.

GitHub stores:

notebooks;
deterministic split files;
documentation;
lightweight final results;
report material.
Reproducibility

The project uses a fixed random seed:

SEED = 42

All neural models use the same:

dataset splits;
preprocessing;
target normalization;
loss definition;
validation protocol;
evaluation metrics;
checkpoint-selection rule.
Primary metrics
Kullback-Leibler divergence, lower is better.
Correlation coefficient, higher is better.
Similarity metric, higher is better.
Course constraints
PyTorch implementation.
Public dataset.
Complete source code.
Final report of at most six pages.
