# Experiments

Everything that was tried on top of the adopted pipeline, with its result. One rule for every change, fixed before the
results: compared with the adopted version, a change is adopted only if the 95% bootstrap interval (2000 resamplings of the
images) of the object F1 difference is entirely above 0, with the pixel Dice and the count error not demonstrably worse.
Post-processing changes first had to gain at least +0.005 of out-of-fold F1 on the training images before being checked on
the validation images.

## Network and training recipe

One training per variant (seed 42, everything else as the adopted recipe), compared on the 43 validation images (network +
post-processing), difference variant − adopted network. Logs and configurations: `runs/cresunet_light_<variant>/`,
`runs/cresunet/`; executed with `kaggle/recipe_variants.ipynb` and `kaggle/train.ipynb`. The c-ResUNet of Morelli et al.
(2021) is the ResUNet of Zhang et al. (2018) with two additions, a learned colour conversion and the 5 × 5 block, evaluated
there only together (on the Yellow collection); the adopted network keeps the first and drops the second.

| variant | object F1 | other demonstrable differences | decision |
|---|---|---|---|
| c-ResUNet of Morelli et al., 2021 (with the 5 × 5 bottleneck block, 1.32 M parameters) | [−0.020, +0.006] | none; training images F1 [−0.006, +0.010] | not adopted (2.6× the parameters for no gain) |
| early stopping (patience 15; stopped at epoch 52) | [−0.034, −0.001] | precision worse | not adopted |
| no gamma augmentation | [−0.013, +0.009] | none | not adopted |
| per-image normalization | [−0.037, +0.005] | count error worse [+0.05, +1.26] | not adopted |
| wider network (32 base channels, 2.0 M parameters) | [−0.011, +0.019] | precision better [+0.006, +0.035]; training images: Dice worse | not adopted |
| 200 epochs | [−0.005, +0.016] | none | not adopted |

## Post-processing

Each variant changes one element of the pipeline, with the object classifier chosen again for it (out-of-fold F1 on the
169 training images, 5 folds by image). Source: `runs/cresunet_light/postprocessing_ablations.json`.

| variant | result | decision |
|---|---|---|
| **minimum area 300 px** instead of 120 | out-of-fold +0.006; validation F1 [+0.001, +0.009]; cross-validation [+0.001, +0.009] | **adopted** |
| minimum area 160 / 200 / 250 px | out-of-fold +0.002 / +0.003 / +0.004 | below the required gain |
| minimum area 350 / 400 px | out-of-fold +0.006; validation F1 [−0.005, +0.014] / [−0.009, +0.014] | not adopted |
| no watershed; seeds 12 / 15 / 25 px apart | out-of-fold −0.002; +0.004 / +0.001 / −0.002 (12 and 15 px cut 3 and 1 cells in two) | 20 px kept |
| test-time augmentation (8 symmetries; `experiments/predict_tta.py`) | network alone, training images: F1 +0.0048 | below the required gain |
| features from the annotation protocol | out-of-fold −0.00004 | not adopted |
| candidates from the robust threshold (k = 5, 4, 3) or Cellpose zero-shot, with a second classifier | out-of-fold ≤ +0.0002 | not adopted |
| morphological closing | pairs within 6 px on the training images: 1 same cell, 18 different cells, 4 with an unannotated object | not adopted |

## Cross-validation

The 212 trainval images in 5 folds stratified by number of cells (`splits/green_cv{0..4}.json`, `train/kfold_split.py`).
For each fold a network is trained on the other four folds and predicts the held-out one with `last.pt` (epoch 100), so every
image is predicted by a network that has not seen it; the object classifier of each fold is chosen and trained on the
objects of the other folds. Two recipes: the adopted one (`runs/cresunet_light_cv{k}/`) and the adopted one + weight maps
(Morelli et al., 2021; `runs/cresunet_light_wm_cv{k}/`). Executed with `kaggle/cross_validation.ipynb`; comparison in
`runs/cv_compare/` (`results.json`, `per_fold.csv`, `classifier_selection.csv`).

| 212 images, 3421 cells | Dice | object F1 | precision | recall | merged | count MAE |
|---|---|---|---|---|---|---|
| network alone | 0.748 | 0.790 | 0.747 | 0.838 | 54 | 4.01 |
| + classifier | 0.744 | 0.808 | 0.796 | 0.819 | 36 | 3.67 |
| + watershed + classifier | 0.746 | 0.813 | 0.797 | 0.830 | 32 | 3.78 |
| + watershed + classifier, minimum area 300 px (adopted) | 0.747 | 0.818 | 0.809 | 0.827 | 32 | 3.62 |
| weight maps: network alone | 0.748 | 0.793 | 0.745 | 0.847 | 52 | 4.12 |
| weight maps: + watershed + classifier | 0.749 | 0.817 | 0.800 | 0.835 | 32 | 3.73 |

- **Weight maps** (+ watershed + classifier, against the same without weight maps): F1 [−0.001, +0.010], better in 5 folds
  of 5 but not demonstrable on the 212 images; merged objects 32 vs 32 → not adopted.
- **Watershed** (20 px): F1 [+0.002, +0.010], better in 4 folds of 5 → kept.
- **Minimum area 300 px**: F1 [+0.001, +0.009], better in 5 folds of 5 → adopted.
- Differences between groups of images (folds: F1 0.77–0.85) are much larger than the differences between recipes.
