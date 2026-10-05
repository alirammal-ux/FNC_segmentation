'''Model selection for the object classifier: logistic regression vs gradient boosting,
cross-validation on TRAIN only. Objects touching the image border are never judged (always kept).
Second of the three object-classifier files: object_features → select_object_classifier → object_classifier.'''

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import prf
from evaluation.object_features import FEATURES
from evaluation.postprocess import label_instances

# the two candidates: linear vs non-linear
CANDIDATES = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "boosting": lambda: HistGradientBoostingClassifier(random_state=0),
}

# an object is dropped if P(TP) < t; t = 0 keeps every object (= v2)
THRESHOLDS = np.round(np.arange(0.0, 0.91, 0.01), 2)


def oof_scores(make_model, table, n_splits=5, features=FEATURES):
    '''Out-of-fold P(TP) of the interior objects; border objects get 1 (always kept).
    Folds by IMAGE: all objects of an image in the same fold.'''
    X, y = table[features].to_numpy(np.float64), table.label.to_numpy()
    inner = np.flatnonzero(~table.border.to_numpy(bool))
    scores = np.ones(len(table))
    for fit_i, held_i in GroupKFold(n_splits).split(inner, groups=table.image.to_numpy()[inner]):
        fit_idx, held_idx = inner[fit_i], inner[held_i]
        model = make_model().fit(X[fit_idx], y[fit_idx])
        scores[held_idx] = model.predict_proba(X[held_idx])[:, 1]
    return scores


def object_prf(table, n_gt, keep):
    '''Object precision, recall, F1 after dropping the objects with keep = False (n_gt = true cells).
    From the table alone: IoU > 0.5 pairs are one-to-one, so dropping an object does not change the others.'''
    tp = int(table.label.to_numpy()[keep].sum())
    fp = int(keep.sum()) - tp
    precision, recall, f1 = prf(tp, fp, n_gt - tp)
    return {"obj_f1": f1, "obj_precision": precision, "obj_recall": recall}


def cv_figure(curves, summary, path):
    '''Out-of-fold F1 on train vs threshold, one line per candidate.'''
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for (name, curve), color in zip(curves.groupby("model", sort=False), ["#2a78d6", "#eb6834"]):
        ax.plot(curve.threshold, curve.obj_f1, color=color, lw=1.6, label=name)
        best = summary.set_index("model").loc[name]
        ax.plot(best.threshold, best.obj_f1, "o", color=color)
    ax.axhline(curves.obj_f1.iloc[0], color="#9a9994", lw=1, ls="--", label="keep all (v2)")
    ax.set_xlabel("threshold on P(TP): objects below are dropped")
    ax.set_ylabel("object F1 (train, out-of-fold)")
    ax.legend(frameon=False)
    ax.grid(color="#e6e5e0", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Object classifier: choose model and threshold with CV on train")
    parser.add_argument("--run", type=Path, default=Path("runs/green_v2"))
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()
    out = args.run / "object_classifier"

    table = pd.read_csv(out / "objects_train.csv")       # from object_features.py
    n_gt = sum(int(label_instances(m > 0).max()) for m in np.load(args.run / "pred_train.npz")["masks"])

    # per candidate: F1 at every threshold, best threshold = highest F1 (ties → lowest t)
    curves, summary = [], []
    for name, make_model in CANDIDATES.items():
        scores = oof_scores(make_model, table, args.n_splits)
        curve = pd.DataFrame([{"model": name, "threshold": t, **object_prf(table, n_gt, scores >= t)}
                              for t in THRESHOLDS])
        best = curve.loc[curve.obj_f1.idxmax()]
        summary.append({**best.to_dict(), "n_dropped": int((scores < best.threshold).sum())})
        curves.append(curve)
    curves, summary = pd.concat(curves, ignore_index=True), pd.DataFrame(summary)

    print(f"train: {len(table)} objects, {n_gt} true cells | keep all (v2): F1 {curves.obj_f1.iloc[0]:.4f}")
    print(summary.round(4).to_string(index=False))
    winner = summary.loc[summary.obj_f1.idxmax()]
    print(f"winner: {winner.model}, threshold {winner.threshold:.2f} (→ constants in object_classifier.py)")

    curves.to_csv(out / "cv_curves.csv", index=False)
    cv_figure(curves, summary, out / "cv_f1.png")


if __name__ == "__main__":
    main()
