'''Model selection for the object classifier: logistic regression vs gradient boosting and the threshold on P(true),
cross-validation on TRAIN only (folds by image). Objects touching the image border are never judged (always kept).'''

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import prf
from evaluation.object_features import FEATURES

# the two candidates: linear vs non-linear
CANDIDATES = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "boosting": lambda: HistGradientBoostingClassifier(random_state=0),
}

# an object is dropped if P(TP) < t; t = 0 keeps every object (= no classifier)
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


def select_classifier(train, n_gt, n_splits=5):
    '''The race on train: per candidate, best threshold and out-of-fold F1 (ties → lowest threshold);
    winner = best F1 (ties → first candidate). n_gt = true cells of the training images.'''
    rows = []
    for name, make_model in CANDIDATES.items():
        scores = oof_scores(make_model, train, n_splits)
        f1 = [object_prf(train, n_gt, scores >= t)["obj_f1"] for t in THRESHOLDS]
        k = int(np.argmax(f1))
        rows.append({"model": name, "threshold": float(THRESHOLDS[k]), "obj_f1": f1[k]})
    summary = pd.DataFrame(rows)
    return summary, summary.loc[summary.obj_f1.idxmax()]
