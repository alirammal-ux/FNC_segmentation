# Other methods

The methods compared with the adopted pipeline in `notebooks/04_test.ipynb`. Nothing here is trained in the notebooks:
they only load the parameters, weights or predictions stored in this folder.

## Threshold baselines (no learning) — `threshold_baseline.py`

- **Otsu**: one threshold per image on the green channel (no parameters).
- **Robust threshold**: background median + k · MAD on the green channel, then the same post-processing as the network.
  `k = 6.5` was chosen on the **training images** only (grid 1.0–30.0, best pixel Dice): `threshold_val.json`.

## Cellpose (version 4.2.1.1, Cellpose-SAM) — `cellpose/`

Cellpose cannot run on our CPU machine in reasonable time (~290 M parameters); it was run on a Kaggle T4 GPU.

- `kaggle_cellpose.ipynb`, in one session: zero-shot predictions (green channel, diameter 30 px, default settings);
  fine-tuning on the 169 training images (100 epochs, default learning rate and optimizer, `min_train_masks = 0`);
  predictions of the fine-tuned model, masks and cell probabilities, on the training, validation and test images.
- Test predictions: `cellpose_test.npz` (zero-shot masks), `cellpose_ftp_test.npz` (fine-tuned masks + probabilities in 8 bit).
- The stored files come from one run of `kaggle_cellpose.ipynb` (settings in `cellpose_config.json`; the run with its
  outputs: `kaggle/executed/kaggle-cellpose.ipynb`). The zero-shot predictions are deterministic (identical to those of an
  earlier run); the fine-tuning has no fixed seed, so another run gives close but not identical numbers.
- **Fine-tuned + classifier**: the object classifier of our pipeline (logistic regression on the same 10 features; border
  objects never judged), trained on the objects that the fine-tuned Cellpose predicts on the training images
  (`objects_train.csv`), with its threshold chosen for Cellpose by the same 5-fold cross-validation by image:
  keep if P(true) ≥ 0.33 (out-of-fold F1 0.8175; gradient boosting reaches the same, `classifier_results.json`).
  Stored frozen in `cellpose_classifier.joblib` (+ `cellpose_classifier.json`).
