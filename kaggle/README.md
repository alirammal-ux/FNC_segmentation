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

| notebook | what it runs | output, copied into `runs/` | T4 time |
|---|---|---|---|
| `train.ipynb` | training of c-ResUNet-light (adopted) or c-ResUNet (Morelli et al., 2021) on the train/validation split | `cresunet_light/`, `cresunet/` | ~1 h each |
| `cross_validation.ipynb` | 5-fold cross-validation of two recipes: adopted, adopted + weight maps | `cresunet_light_cv{k}/`, `cresunet_light_wm_cv{k}/` | ~4.8 h per recipe |
| `recipe_variants.ipynb` | five variants of the training recipe + test-time augmentation of the adopted network | `cresunet_light_<variant>/`, `cresunet_light_tta/` | ~6 h |
| `../other_methods/cellpose/kaggle_cellpose.ipynb` | Cellpose zero-shot and fine-tuning (Internet on) | `other_methods/cellpose/` | ~5 h |
| `../other_methods/cellpose/kaggle_cellpose_ft_predict.ipynb` | fine-tuned Cellpose: masks and probabilities | `other_methods/cellpose/` | ~1.5 h |

Each notebook has a `PROBE` switch:

- `PROBE = True`: a short check in an interactive session (one epoch on a few images).
- `PROBE = False`: the full run, launched with **Save Version → Save & Run All (Commit)**. It keeps running with the
  browser closed, up to the 12 h limit of a session.

The results are downloaded from the **Output** tab of the version.

The same commands also run on a CPU (for example `python -m train.training --small_rf ...` from the repository root), but a
full training then takes many hours.
