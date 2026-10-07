# Efficient Multi-Scale Visual Saliency Prediction

Deep Learning course project implemented in PyTorch and Google Colab.

The project studies whether multi-scale feature fusion can improve visual
saliency prediction while retaining the computational efficiency of a
lightweight encoder.

## Task

Given an RGB natural image, the model predicts a continuous saliency map
representing the spatial distribution of likely human visual attention.

This project addresses free-viewing fixation prediction. It does not address
salient-object segmentation or explanation methods such as Grad-CAM.

## Dataset

The experiments use the SALICON dataset, which contains natural MS COCO images
and continuous attention maps.

The available labelled data are divided as follows:

- 10,000 official training samples used for model training;
- 2,500 samples from the official validation set used for validation;
- 2,500 samples from the official validation set used as an internal test set.

The validation/test division is deterministic and was generated once using
seed 42. The official SALICON test annotations are not used.

The dataset is not included in this repository because of its size.

## Compared methods

Four methods are evaluated:

| Method | Encoder | Decoder | Purpose |
|---|---|---|---|
| MeanMap | None | None | Non-learned training-set spatial-prior baseline |
| Light-S | MobileNetV2 | Single-scale | Lightweight learned baseline |
| Light-M | MobileNetV2 | Multi-scale | Main compact multi-scale model |
| Heavy-M | ResNet-18 | Multi-scale | Larger encoder reference |

Light-S and Light-M use the same MobileNetV2 encoder, isolating the effect of
multi-scale fusion.

Light-M and Heavy-M use the same decoder structure, isolating the effect of
increasing encoder capacity.

## Architecture

The neural models use ImageNet-pretrained encoders.

Light-S decodes only the deepest stride-32 feature. Light-M and Heavy-M use
features at output strides 4, 8, 16, and 32. Each feature is projected to
32 channels and combined through additive top-down fusion.

The refinement blocks contain:

- a depthwise 3x3 convolution;
- a pointwise 1x1 convolution;
- Group Normalization;
- ReLU activation.

The final logits are resized to 169 x 256 and converted into a spatial
probability distribution using softmax over all image positions.

## Training

All neural models use the same:

- training, validation, and internal-test splits;
- input and target resolution;
- target normalization;
- training objective;
- optimizer policy;
- checkpoint-selection rule;
- evaluation implementation.

The common loss is:

```text
KLD(G || P) + 0.5 * (1 - CC(P, G))
```

Main training settings:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Batch size | 8 |
| Encoder learning rate | 1e-4 |
| Decoder learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Maximum epochs | 15 |
| Early-stopping patience | 3 epochs |
| Checkpoint criterion | Highest validation CC |
| Random seed | 42 |

The encoder is frozen during the first epoch. It is then fine-tuned while its
Batch Normalization running statistics remain fixed.

## Evaluation

Prediction quality is evaluated using:

- Kullback--Leibler divergence (KLD), lower is better;
- correlation coefficient (CC), higher is better;
- similarity score (SIM), higher is better.

Efficiency is evaluated using:

- parameter count;
- model state-dictionary size;
- multiply-accumulate operations;
- batch-one CPU latency.

A MeanMap baseline and a ground-truth-defined off-centre subset are also used
to study the influence of the SALICON centre prior.

## Final internal-test results

All results below were obtained on the same fixed 2,500-image internal test
split at a resolution of 169 x 256.

| Method | KLD ↓ | CC ↑ | SIM ↑ |
|---|---:|---:|---:|
| MeanMap | 0.7202 | 0.5570 | 0.5528 |
| Light-S | 0.2371 | 0.8735 | 0.7690 |
| Light-M | **0.2257** | **0.8813** | 0.7733 |
| Heavy-M | 0.2399 | 0.8792 | **0.7737** |

## Efficiency results

CPU measurements use one thread, 30 warm-up forwards, and 200 timed forwards,
repeated three times.

| Model | Parameters | MACs | Median CPU latency |
|---|---:|---:|---:|
| Light-S | 1.826 M | 0.282 G | 22.86 ms |
| Light-M | 1.831 M | 0.286 G | 22.08 ms |
| Heavy-M | 11.211 M | 1.673 G | 55.16 ms |

Light-M provides the best overall accuracy--efficiency trade-off under the
selected experimental protocol.

## Repository structure

```text
efficient-saliency-prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── losses_metrics.py
│   ├── models.py
│   └── train_eval.py
├── notebooks/
│   ├── 00_dataset_preparation.ipynb
│   ├── 01_data_pipeline.ipynb
│   ├── 02_metrics_meanmap.ipynb
│   ├── 03_light_single_pre.ipynb
│   ├── 03_light_single_train.ipynb
│   ├── 04_light_multi_train.ipynb
│   ├── 05_heavy_multi.ipynb
│   └── 06_final_evaluation.ipynb
└── splits/
    ├── dataset_manifest.csv
    ├── split_metadata.json
    ├── train.txt
    ├── val.txt
    ├── test.txt
    ├── train_manifest.csv
    ├── val_manifest.csv
    └── test_manifest.csv
```

Large artefacts are stored externally and are not included in the repository:

- the SALICON dataset;
- local dataset-cache archives;
- complete prediction folders;
- model checkpoints;
- temporary outputs and backups.

## Notebook order

The notebooks should be read or executed in this order:

1. `00_dataset_preparation.ipynb`  
   Verifies the SALICON files and creates the deterministic manifests.

2. `01_data_pipeline.ipynb`  
   Tests dataset loading, preprocessing, and batching.

3. `02_metrics_meanmap.ipynb`  
   Tests KLD, CC, and SIM and evaluates the MeanMap baseline.

4. `03_light_single_pre.ipynb`  
   Performs architecture, gradient, and small-subset checks for Light-S.

5. `03_light_single_train.ipynb`  
   Trains the MobileNetV2 single-scale model.

6. `04_light_multi_train.ipynb`  
   Trains the MobileNetV2 multi-scale model.

7. `05_heavy_multi.ipynb`  
   Trains the ResNet-18 multi-scale model.

8. `06_final_evaluation.ipynb`  
   Loads the selected checkpoints and produces the final accuracy, efficiency,
   centre-bias, and qualitative results.

The executed `06_final_evaluation.ipynb` is the authoritative source for the
final reported results.


## Data and checkpoint paths

The Colab notebooks expect the persistent project workspace at:

```text
/content/drive/MyDrive/saliency_project
```

The principal external artefacts are:

```text
saliency_project/
├── data/SALICON/
├── checkpoints/
│   ├── light_single/best.pt
│   ├── light_multi/best.pt
│   └── heavy_multi_169/best.pt
├── outputs/
└── backups/
```

The dataset is temporarily extracted under `/content` during execution to
avoid repeated small-file reads from mounted Google Drive.


## Author

Riccardo Dal Bianco