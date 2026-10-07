# c-FOS nuclei segmentation — Fluorescent Neuronal Cells v2, Green collection

Segmentation (and counting) of c-FOS-stained neuronal nuclei in the **Green** collection of the
Fluorescent Neuronal Cells v2 dataset (Clissa et al., 2024), with a light c-ResUNet trained from scratch.

## Adopted pipeline

```
image → c-ResUNet-light (runs/cresunet_light/best.pt, 0.50 M parameters) → probability map
      → threshold 0.5 → fill holes → remove objects < 300 px
      → watershed on the distance transform (seeds at least 20 px apart)
      → object classifier (logistic regression on 10 object features, keep if P(true) ≥ 0.34; border objects always kept)
      → labelled cells and their count
```

c-ResUNet-light is the c-ResUNet of Morelli et al. (2021) without the residual block with 5 × 5 kernels at the end of the
encoder. The c-ResUNet is the ResUNet of Zhang et al. (2018) with two additions, a learned colour conversion (1 × 1
convolution) and that block, which enlarges the field of view; the two were evaluated only together, on another collection
(Yellow). c-ResUNet-light keeps the first addition and drops the second: on Green the block brings no measurable gain, for
2.6× the parameters (notebook 02).

## Results on the test set

70 images never used for training or for any choice, 1172 annotated cells; object F1 with matching at IoU > 0.5, 95%
bootstrap interval over the images (notebook 04).

| method | object F1 | precision | recall | pixel Dice | count error (MAE) |
|---|---|---|---|---|---|
| **adopted pipeline** | **0.834** [0.810, 0.858] | 0.807 | 0.864 | 0.758 | 3.9 |
| Cellpose-SAM fine-tuned on the same images | 0.834 | 0.854 | 0.815 | 0.757 | 3.7 |
| Cellpose-SAM zero-shot | 0.574 | 0.412 | 0.944 | 0.579 | 21.6 |
| robust threshold (median + 6.5 MAD) | 0.506 | 0.441 | 0.595 | 0.579 | 9.8 |

The adopted pipeline is at the level of Cellpose-SAM fine-tuned (no demonstrable difference in F1), with ~600× fewer
parameters and runs on a CPU. Most remaining errors are faint cells whose annotation is itself ambiguous (notebooks 03 and 04).

## Notebooks

| notebook | content |
|---|---|
| `01_eda_green.ipynb` | data, mask cleaning, morphology of the Green cells |
| `02_data_and_training.ipynb` | split, dataset and augmentation, model, loss, training, variants tried |
| `03_post_processing.ipynb` | from probabilities to cells: threshold, minimum area, watershed, object classifier |
| `04_test.ipynb` | evaluation on the test set and comparison with other methods |

## Structure

```
notebooks/        the four notebooks above
models/           c-ResUNet (c_res_unet.py), its logit-output version used for training, and c-ResUNet-light
                  (c_res_unet_small_rf.py: no 5×5 residual block in the bottleneck)
train/            dataset, loss, training script, train/val split and 5-fold split
evaluation/       prediction, post-processing, object features and classifier, metrics
experiments/      the experiments that were tried and not adopted: index, results, code (test-time augmentation)
splits/           train/val split and the 5 cross-validation folds
runs/             saved results read by the notebooks: cresunet_light/ (adopted network and classifiers), cresunet/
                  (c-ResUNet, Morelli et al.), cresunet_light_cv*/ and cv_compare/ (cross-validation), cresunet_light_<variant>/
                  (training variants), test_reference.json (expected test numbers, checked by notebook 04)
other_methods/    the methods we compare with (threshold baselines, Cellpose) and how they were obtained
kaggle/           notebooks run on a Kaggle GPU (training, cross-validation, recipe variants) and how to run them
data/             not included (see below)
requirements.txt  versions of the local environment
```

## Data

Download the Green collection (Clissa et al., 2024, AMS Acta, University of Bologna):

```bash
wget -c https://amsacta.unibo.it/7347/28/green.zip
unzip green.zip -d data/      # → data/green/{trainval,test,unlabelled}
```

The cleaned masks (`data/cleaned_masks/{trainval,test}/Green/masks`) are created by `notebooks/01_eda_green.ipynb`.

## Environment

Python 3.12 with the packages in `requirements.txt` (PyTorch 2.14, NumPy 2.4, SciPy 1.17, scikit-image 0.26,
scikit-learn 1.9, pandas 3.0, matplotlib 3.10, joblib 1.6).
The notebooks run on CPU; training needs a GPU and was run on Kaggle (T4): see `kaggle/README.md`.
Run everything from the repository root, so that `models`, `train`, `evaluation` and `experiments` are importable.

## How to run

From the repository root, after downloading the data:

1. `notebooks/01_eda_green.ipynb` → cleaned masks (needed by everything else);
2. `notebooks/02` → `04` in order (CPU; 04 takes about 15 minutes). They load the saved checkpoints and classifiers in
   `runs/` and check every number against the saved references.

To retrain instead of loading: the network with `kaggle/train.ipynb` (GPU, ~1 h), then the probability maps of the training
and validation images (`python -m evaluation.predict --ckpt runs/cresunet_light/best.pt --subset train|val ...`) and the
object classifier (`python -m evaluation.watershed_classifier`).
