# Kaggle notebooks (GPU)

Every step that needs a GPU was run on Kaggle (T4). The notebooks in `notebooks/` do not train anything: they read
the files that these notebooks produce.

## Inputs: two private Kaggle datasets

Both are uploaded as zip files (Kaggle extracts them). The notebooks locate the folders with a search under
`/kaggle/input`, so how Kaggle nests them does not matter.

- **`cellseg-code`**: the code, zipped from the repository root:

  ```bash
  zip -r ~/cellseg-code.zip models train evaluation experiments splits runs/cresunet_light/best.pt -x "*__pycache__*"
  ```

  `runs/cresunet_light/best.pt` (the adopted network) is needed only by the TTA in `recipe_variants.ipynb`.

- **`fnc-green-data`**: images and cleaned masks, trainval and test. The cleaned masks are written by
  `notebooks/01_eda_green.ipynb`:

  ```bash
  cd data && zip -r ~/fnc-green-data.zip green/trainval/images green/test/images cleaned_masks
  ```

## Notebooks

| notebook | what it runs | output folders (on Kaggle) | T4 time |
|---|---|---|---|
| `train.ipynb` | training of c-ResUNet-light (adopted) or c-ResUNet (Morelli et al., 2021) on the train/validation split | `cresunet_light/`, `cresunet/` | ~1 h each |
| `cross_validation.ipynb` | 5-fold cross-validation of two recipes: adopted, adopted + weight maps | `cresunet_light_cv{k}/`, `cresunet_light_wm_cv{k}/` | ~4.8 h per recipe |
| `recipe_variants.ipynb` | five trainings of the adopted network, each with one change to the training recipe, and test-time augmentation (see below) | `cresunet_light_<variant>/`, `cresunet_light_tta/` | ~6 h |
| `../other_methods/cellpose/kaggle_cellpose.ipynb` | Cellpose: zero-shot, fine-tuning on the training images, predictions of the fine-tuned model (Internet on) | `runs/cellpose/` (the files kept are in `other_methods/cellpose/`) | ~7 h |

This repository keeps only what the notebooks read: in `runs/`, the checkpoints and classifiers of the two networks and,
for the cross-validation and the recipe variants, `config.json` and `log.csv`. The probability maps (`pred_*.npz`, ~70 MB
per run, ~400 MB for the test-time augmentation) are not included.

### What `recipe_variants.ipynb` does

It checks whether the training recipe of the adopted network can be improved. **Part 1:** five trainings of
c-ResUNet-light, identical to the adopted one except for **one change** each; after each training, the probability maps of
the 43 validation images (`best.pt`), compared afterwards with those of the adopted network.

| variant | change from the adopted recipe |
|---|---|
| `early_stop` | stop after 15 epochs without improvement of the validation Dice, instead of always 100 epochs |
| `no_gamma` | brightness augmentation with the gain only, without the gamma (contrast) change |
| `per_image` | each image normalized with its own statistics, instead of the mean and std of the training images |
| `wide` | 32 channels in the first level instead of 16 (a network 4× larger) |
| `200ep` | 200 epochs instead of 100 |

**Part 2: test-time augmentation (TTA)**, no training: the adopted network predicts every image in its 8 orientations
(4 rotations × mirror), each prediction is turned back and the 8 probability maps are averaged
(`experiments/predict_tta.py`).

Results: no variant improves the validation F1 demonstrably (notebook 02, section 7); TTA gains +0.0048 of F1 on the
training images, below the +0.005 required to go on (notebook 03, section 9). Details: `experiments/README.md`.

## Running a notebook

Each notebook has a `PROBE` switch:

- `PROBE = True`: a short check in an interactive session (one epoch on a few images).
- `PROBE = False`: the full run, launched with **Save Version -> Save & Run All (Commit)**. It keeps running with the
  browser closed, up to the 12 h limit of a session.

The results are downloaded from the **Output** tab of the version.

The same commands also run on a CPU, but a
full training then takes many hours.
