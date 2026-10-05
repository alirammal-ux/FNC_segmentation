'''The adopted post-processing pipeline and its object classifier: probability map → threshold 0.5 → fill holes →
objects < 300 px removed → watershed (seeds at least 20 px apart; pieces < 300 px removed) → object classifier.
main() builds the object tables of the training and validation images, runs the classifier race on the training objects
(logistic regression vs gradient boosting and the threshold; 5 folds by image; border objects never judged), fits the
winner on the interior training objects, saves it (classifier.joblib) and evaluates the pipeline once on the validation
images. delivered_labels() is the pipeline on one image, used as is by notebooks 03 and 04.'''

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import skimage.io as io
from skimage.segmentation import relabel_sequential

from evaluation.object_classifier import keep_objects
from evaluation.object_features import FEATURES, build_table, label_objects
from evaluation.postprocess import label_instances, predict_labels
from evaluation.select_object_classifier import CANDIDATES, select_classifier
from evaluation.tune import counts_row, metrics_from_counts

MIN_DISTANCE = 20     # minimum distance between watershed seeds (px): no single training cell cut in two (notebook 03)
PRED_MIN_AREA = 300   # smallest predicted object kept, before and after the watershed (px; notebook 03)
THRESHOLD = 0.34      # objects with P(true) below are dropped: winner of the race for the adopted network (notebook 03)


def delivered_labels(prob, img, gt_lab, i, model, threshold=THRESHOLD, min_area=PRED_MIN_AREA):
    '''The adopted pipeline on ONE image → (label image, object table with "keep").
    gt_lab is only used for the "label" column of the table (TP / FP), never for the prediction.'''
    lab = predict_labels(prob, 0.5, min_distance=MIN_DISTANCE, min_area=min_area)
    t = label_objects(lab, prob, img, gt_lab)
    t = t.assign(image=i, obj=np.arange(1, len(t) + 1))
    keep = keep_objects(model, t, threshold) if len(t) else np.zeros(0, dtype=bool)
    kept = t.obj[keep].to_numpy()
    return relabel_sequential(np.where(np.isin(lab, kept), lab, 0))[0], t.assign(keep=keep)   # halves of a split touch


def main():
    parser = argparse.ArgumentParser(description="Object classifier of the adopted pipeline: race on train, fit, one check on val")
    parser.add_argument("--run", type=Path, default=Path("runs/cresunet_light"))   # needs pred_train.npz and pred_val.npz
    parser.add_argument("--min_area", type=int, default=PRED_MIN_AREA)           # smallest predicted object kept (px)
    args = parser.parse_args()
    images_dir, masks_dir = Path("data/green/trainval/images"), Path("data/cleaned_masks/trainval/Green/masks")
    out = args.run / ("object_classifier" if args.min_area == PRED_MIN_AREA else f"object_classifier_area{args.min_area}")
    out.mkdir(exist_ok=True)

    # 1) training objects after the watershed → race → winner fit on the interior objects, saved for the test
    train = build_table(args.run / "pred_train.npz", images_dir, masks_dir, min_distance=MIN_DISTANCE, min_area=args.min_area)
    train.to_csv(out / "objects_train.csv", index=False)
    n_gt = sum(int(label_instances(m > 0).max()) for m in np.load(args.run / "pred_train.npz")["masks"])
    summary, winner = select_classifier(train, n_gt)
    print(f"train: {len(train)} objects | TP {int(train.label.sum())} | true cells {n_gt}")
    print("race (out-of-fold F1 on train):\n" + summary.round(4).to_string(index=False))
    inner = ~train.border.to_numpy(bool)
    model = CANDIDATES[winner.model]().fit(train[FEATURES].to_numpy(np.float64)[inner], train.label.to_numpy()[inner])
    joblib.dump(model, out / "classifier.joblib")
    threshold = float(winner.threshold)

    # 2) ONE evaluation on val, through delivered_labels (the same function used on the test)
    d = np.load(args.run / "pred_val.npz")
    counts, tables = [], []
    for i, (name, p, m) in enumerate(zip(d["names"], d["probs"], d["masks"])):
        lab, t = delivered_labels(p.astype(np.float32), io.imread(images_dir / str(name)), label_instances(m > 0), i, model,
                                  threshold, args.min_area)
        counts.append(counts_row(lab, m > 0))
        tables.append(t)
    pd.concat(tables, ignore_index=True).to_csv(out / "objects_val.csv", index=False)
    val = metrics_from_counts(np.array(counts, dtype=np.int64))
    print(f"winner: {winner.model}, threshold {threshold:.2f} | val:", {k: round(float(v), 4) for k, v in val.items()})

    with open(out / "results.json", "w") as f:
        json.dump({"min_distance": MIN_DISTANCE, "min_area": args.min_area, "model": winner.model,
                   "threshold_classifier": threshold, "oof_f1": float(winner.obj_f1),
                   "race": summary.to_dict(orient="records"), "train_objects": len(train), "val": val},
                  f, indent=2, default=float)
    print(f"results in {out}")


if __name__ == "__main__":
    main()
