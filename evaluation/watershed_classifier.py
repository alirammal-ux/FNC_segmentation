'''Object classifier of the ADOPTED pipeline: v2 → post-processing → watershed (seeds at least 20 px apart)
→ object classifier. Builds the object tables after the watershed (train, val), fits the logistic on the interior train
objects (threshold chosen by cross-validation on train), saves it for the test, and checks on val that the
pipeline gives the numbers already known. delivered_labels() is the pipeline on one image, used as is by notebook 04.'''

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import skimage.io as io
from skimage.segmentation import relabel_sequential

from evaluation.object_classifier import fit_classifier, keep_objects
from evaluation.object_features import build_table, label_objects
from evaluation.postprocess import label_instances, predict_labels
from evaluation.tune import counts_row, metrics_from_counts

MIN_DISTANCE = 20     # minimum distance between watershed seeds (px): no single train cell split (W1)
THRESHOLD = 0.34      # winner of the W1 cross-validation on train for 20 px (logistic; watershed_grid/grid_train.csv)


def delivered_labels(prob, img, gt_lab, i, model):
    '''The delivered pipeline on ONE image → (label image, object table with "keep").
    gt_lab is only used for the "label" column of the table (TP / FP), never for the prediction.'''
    lab = predict_labels(prob, 0.5, min_distance=MIN_DISTANCE)
    t = label_objects(lab, prob, img, gt_lab)
    t = t.assign(image=i, obj=np.arange(1, len(t) + 1))
    keep = keep_objects(model, t, threshold=THRESHOLD) if len(t) else np.zeros(0, dtype=bool)
    kept = t.obj[keep].to_numpy()
    return relabel_sequential(np.where(np.isin(lab, kept), lab, 0))[0], t.assign(keep=keep)   # halves of a split touch


def main():
    run = Path("runs/green_v2")
    images_dir, masks_dir = Path("data/green/trainval/images"), Path("data/cleaned_masks/trainval/Green/masks")
    out = run / "object_classifier_ws20"
    out.mkdir(exist_ok=True)

    # 1) train objects after the watershed → logistic, saved: the test uses exactly this model
    train = build_table(run / "pred_train.npz", images_dir, masks_dir, min_distance=MIN_DISTANCE)
    train.to_csv(out / "objects_train.csv", index=False)
    print(f"train: {len(train)} objects | TP {int(train.label.sum())} (W1: 2913 / 2305)")
    model = fit_classifier(train)
    joblib.dump(model, out / "classifier.joblib")

    # 2) check on val, through delivered_labels (the same function used on the test)
    d = np.load(run / "pred_val.npz")
    counts, tables = [], []
    for i, (name, p, m) in enumerate(zip(d["names"], d["probs"], d["masks"])):
        lab, t = delivered_labels(p.astype(np.float32), io.imread(images_dir / str(name)), label_instances(m > 0), i, model)
        counts.append(counts_row(lab, m > 0))
        tables.append(t)
    pd.concat(tables, ignore_index=True).to_csv(out / "objects_val.csv", index=False)
    val = metrics_from_counts(np.array(counts, dtype=np.int64))
    print("val:", {k: round(v, 4) for k, v in val.items()},
          "\nexpected: dice 0.7629, obj_f1 0.8123, P 0.7806, R 0.8468, merged 8, MAE 4.2326, bias 1.4419")

    with open(out / "results.json", "w") as f:
        json.dump({"min_distance": MIN_DISTANCE, "threshold_classifier": THRESHOLD, "model": "logistic",
                   "train_objects": len(train), "val": val}, f, indent=2, default=float)
    print(f"results in {out}")


if __name__ == "__main__":
    main()
