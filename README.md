# c-FOS nuclei segmentation — Fluorescent Neuronal Cells v2, Green collection

Segmentation (and counting) of c-FOS-stained neuronal nuclei in the **Green** collection of the
Fluorescent Neuronal Cells v2 dataset (Clissa et al., 2024), with a c-ResUNet trained from scratch.

> Status: the notebooks are being written (see `notebooks/`).

## Adopted pipeline

```
image → c-ResUNet (runs/green_v2/best.pt) → probability map
      → threshold 0.5 → fill holes → remove objects < 120 px
      → watershed on the distance transform (seeds at least 20 px apart)
      → object classifier (logistic regression on 10 object features, keep if P(true) ≥ 0.34; border objects always kept)
      → labelled cells and their count
```

## Notebooks

| notebook | content |
|---|---|
| `01_eda_green.ipynb` | data, mask cleaning, morphology of the Green cells |
| `02_data_and_training.ipynb` | split, dataset and augmentation, model, loss, training, cross-validation |
| `03_from_probabilities_to_cells.ipynb` | threshold, minimum area, watershed, object classifier |
| `04_test.ipynb` | evaluation on the test set and comparison with other methods |

## Structure

```
notebooks/        the four notebooks above
models/           c-ResUNet (c_res_unet.py), its logit-output version used for training, and c-ResUNet-light
                  (c_res_unet_small_rf.py: no 5×5 residual block in the bottleneck)
train/            dataset, loss, training script, train/val split and 5-fold split
evaluation/       prediction, post-processing, object features and classifier, metrics
experiments/      code used only by the experiments (test-time augmentation)
splits/           train/val split and the 5 cross-validation folds
runs/             saved results read by the notebooks (checkpoints, logs, classifiers, cross-validation, test reference)
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
