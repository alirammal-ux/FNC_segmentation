# Other methods

The methods compared with the adopted pipeline in `notebooks/04_test.ipynb`. Nothing here is trained in the notebooks:
they only load the parameters, weights or predictions stored in this folder.

## Threshold baselines (no learning) — `threshold_baseline.py`

- **Otsu**: one threshold per image on the green channel (no parameters).
- **Robust threshold**: background median + k · MAD on the green channel, then the same post-processing as the network.
  `k = 6.5` was chosen on the **training images** only (grid 1.0–30.0, best pixel Dice): `threshold_val.json`.

## Cellpose (version 4.2.1.1, Cellpose-SAM) — `cellpose/`

Cellpose cannot run on our CPU machine in reasonable time (~290 M parameters); it was run on a Kaggle T4 GPU.

- `kaggle_cellpose.ipynb`: zero-shot predictions (green channel, diameter 30 px, default settings) and fine-tuning on the
  169 training images (100 epochs, default learning rate and optimizer, `min_train_masks = 0`).
- `kaggle_cellpose_ft_predict.ipynb`: predictions of the fine-tuned model with its cell probabilities.
- Test predictions: `cellpose_test.npz` (zero-shot masks), `cellpose_ftp_test.npz` (fine-tuned masks + probabilities in 8 bit);
  settings in `cellpose_config.json` and `cellpose_ftp_config.json`.
- **Fine-tuned + classifier**: the same object classifier as our pipeline, selected and trained on the objects that the
  fine-tuned Cellpose predicts on the training images (`objects_train.csv`, selection in `classifier_results.json`):
  logistic regression, keep if P(true) ≥ 0.34. Stored frozen in `cellpose_classifier.joblib` (+ `cellpose_classifier.json`).
