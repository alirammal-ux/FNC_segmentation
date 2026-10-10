# Executed notebooks

The Kaggle notebooks as they ran on a T4 GPU, downloaded with their outputs. They document the GPU runs whose results
are stored in `runs/` and `other_methods/cellpose/`. Code cells and outputs are as they ran; only the markdown cells were
rewritten afterwards, in English and consistent with the code. To rerun, use the clean versions (last column).

| notebook | run | run time | results in this repository | clean version |
|---|---|---|---|---|
| `kaggle-training-light.ipynb` | training of c-ResUNet-light (adopted network) | 1.0 h | `runs/cresunet_light/` | `train.ipynb` |
| `Kaggle-training-cresunetMorelli.ipynb` | training of the c-ResUNet of Morelli et al. (2021) | 1.0 h | `runs/cresunet/` | `train.ipynb` without `--small_rf` |
| `kaggle-cv-light.ipynb` | 5-fold cross-validation, adopted recipe | 4.7 h | `runs/cresunet_light_cv{0..4}/` | `cross_validation.ipynb`, `RECIPE = "light"` |
| `kaggle-cv-light_wm.ipynb` | 5-fold cross-validation, adopted recipe + weight maps | 4.6 h | `runs/cresunet_light_wm_cv{0..4}/` | `cross_validation.ipynb`, `RECIPE = "light_weight_maps"` |
| `kaggle-light-variants.ipynb` | five variants of the training recipe; test-time augmentation of the adopted network | 6.5 h | `runs/cresunet_light_<variant>/`, `runs/cresunet_light/tta_check.json` | `recipe_variants.ipynb` |
| `kaggle-cellpose.ipynb` | Cellpose-SAM: zero-shot, fine-tuning on the training images, predictions of the fine-tuned model | 8.1 h | `other_methods/cellpose/` | `other_methods/cellpose/kaggle_cellpose.ipynb` |

- **Check.** Each training prints one line per epoch (learning rate, losses, validation Dice). For all 17 trainings
  (1752 epochs) these lines are identical to `log.csv` in the corresponding `runs/` folder, and the normalization
  statistics printed at the start are those in `config.json`. The Cellpose predictions on the test images and the
  settings in `other_methods/cellpose/` are the files written by the Cellpose run (its zero-shot predictions are identical
  to those of an earlier run).
- **Language and names.** Some code comments and the printed messages are in Italian, and the runs have the names of
  the development repository: `green_v8` → `cresunet_light`, `green_v2` → `cresunet`, `light_cv{k}` →
  `cresunet_light_cv{k}`, `light_wm_cv{k}` → `cresunet_light_wm_cv{k}`, `light_<variant>` → `cresunet_light_<variant>`.
  The code in this repository is the final version of the code they ran, with messages and comments translated into
  English.
- **Not kept.** The probability maps written by these runs (`pred_*.npz`) and the Cellpose predictions on the training
  and validation images are not in this repository. The results computed from them on CPU are: `runs/cv_compare/`
  (cross-validation), `runs/cresunet_light/tta_check.json` (test-time augmentation) and the object classifier of
  Cellpose fine-tuned (`other_methods/cellpose/`, from its predictions on the training images).
