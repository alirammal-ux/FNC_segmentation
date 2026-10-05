'''The object classifier chosen by select_object_classifier.py: fit on train, saved, one evaluation on val against v2.
Objects touching the image border are never judged (always kept).
Third of the three object-classifier files: object_features → select_object_classifier → object_classifier.'''

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import confusion_counts, match_objects
from evaluation.object_features import FEATURES
from evaluation.postprocess import label_instances, postprocess
from evaluation.tune import bootstrap_diff, metrics_from_counts

# winner of select_object_classifier.py (cross-validation on train)
THRESHOLD = 0.37      # an object is dropped if P(TP) < THRESHOLD


def fit_classifier(train, features=FEATURES):
    '''Logistic regression on the standardized features, fit on the interior train objects.'''
    inner = ~train.border.to_numpy(bool)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return model.fit(train[features].to_numpy(np.float64)[inner], train.label.to_numpy()[inner])


def keep_objects(model, table, features=FEATURES, threshold=THRESHOLD):
    '''True = the object is kept: P(TP) ≥ threshold, or it touches the border.'''
    p_tp = model.predict_proba(table[features].to_numpy(np.float64))[:, 1]
    return (p_tp >= threshold) | table.border.to_numpy(bool)


def filtered_counts(probs, masks, table, keep, threshold=0.5):
    '''Per-image counts (tune.per_img_counts order) after dropping the objects with keep = False.'''
    rows = []
    for i, (p, m) in enumerate(zip(probs, masks)):
        pred_lab = label_instances(postprocess(p.astype(np.float32), threshold=threshold))
        kept = table.obj[(table.image == i).to_numpy() & keep].to_numpy()
        pred_lab = label_instances(np.isin(pred_lab, kept))   # consecutive labels again (match_objects uses max())
        gt = m > 0
        gt_lab = label_instances(gt)
        px = confusion_counts(torch.from_numpy(pred_lab > 0), torch.from_numpy(gt))
        rows.append([*px, *match_objects(pred_lab, gt_lab), pred_lab.max(), gt_lab.max()])
    return np.array(rows, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Object classifier: fit on train, one evaluation on val against v2")
    parser.add_argument("--run", type=Path, default=Path("runs/green_v2"))
    parser.add_argument("--threshold", type=float, default=0.5)   # CNN probability threshold
    args = parser.parse_args()
    out = args.run / "object_classifier"
    train, val = pd.read_csv(out / "objects_train.csv"), pd.read_csv(out / "objects_val.csv")   # from object_features.py

    # 1) fit on train and save: the test will use exactly this model
    model = fit_classifier(train)
    joblib.dump(model, out / "classifier.joblib")
    weights = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
    print("weights (standardized features):", weights.round(3).to_dict())

    # 2) val: same pipeline without and with the classifier
    keep = keep_objects(model, val)
    d = np.load(args.run / "pred_val.npz")
    counts = {"v2": filtered_counts(d["probs"], d["masks"], val, np.ones(len(val), dtype=bool), args.threshold),
              "v2 + classifier": filtered_counts(d["probs"], d["masks"], val, keep, args.threshold)}
    dropped = val.label.to_numpy()[~keep]
    print(f"val: {len(dropped)} objects dropped ({int((dropped == 0).sum())} FP, {int(dropped.sum())} TP)")
    print(pd.DataFrame([{"pipeline": k, **metrics_from_counts(c)} for k, c in counts.items()]).round(4).to_string(index=False))

    # 3) decision rule fixed before: F1 CI all > 0, Dice and MAE not demonstrably worse
    ci = {m: [float(v) for v in bootstrap_diff(counts["v2 + classifier"], counts["v2"], m)]
          for m in ["dice", "obj_f1", "obj_precision", "obj_recall", "count_mae"]}
    for m, (lo, hi) in ci.items():
        print(f"classifier − v2, {m}: 95% CI [{lo:+.4f}, {hi:+.4f}]")
    adopt = ci["obj_f1"][0] > 0 and ci["dice"][1] >= 0 and ci["count_mae"][0] <= 0
    print("decision:", "ADOPT" if adopt else "do not adopt")

    results = {"config": {"run": str(args.run), "threshold_cnn": args.threshold, "model": "logistic",
                          "threshold_classifier": THRESHOLD, "features": FEATURES, "weights": weights.to_dict()},
               "val": {k: metrics_from_counts(c) for k, c in counts.items()},
               "val_dropped": {"total": int(len(dropped)), "fp": int((dropped == 0).sum()), "tp": int(dropped.sum())},
               "val_ci95_classifier_minus_v2": ci, "adopt": bool(adopt)}
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"results in {out}")


if __name__ == "__main__":
    main()
