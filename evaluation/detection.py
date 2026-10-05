'''Cell-counter (detection) view, criterion of Morelli et al. 2021 adapted to Green: a predicted and a true cell are the
same cell if their centres are closer than one cell diameter; matching is one-to-one (Hungarian algorithm).'''

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import measure

from evaluation.metrics import prf

MAX_DIST = 30.0   # px ≈ mean equivalent diameter of the true Green cells in train (30.6 px)


def match_centres(pred_lab, gt_lab, max_dist=MAX_DIST):
    '''Detection TP, FP, FN of one image: one-to-one pairs whose centres are closer than max_dist.'''
    pc = np.array([p.centroid for p in measure.regionprops(pred_lab)]).reshape(-1, 2)
    gc = np.array([p.centroid for p in measure.regionprops(gt_lab)]).reshape(-1, 2)
    if len(pc) == 0 or len(gc) == 0:
        return 0, len(pc), len(gc)
    dist = np.linalg.norm(pc[:, None] - gc[None], axis=2)          # (n_pred, n_true)
    # pairs with the smallest total distance; pairs too far apart get a huge cost (never preferred)
    rows, cols = linear_sum_assignment(np.where(dist < max_dist, dist, 1e6))
    tp = int((dist[rows, cols] < max_dist).sum())
    return tp, len(pc) - tp, len(gc) - tp


def counter_metrics(c):
    '''Detection precision/recall/F1 (from the totals) and count MAE/bias (per image).
    c: one row per image = [TP, FP, FN, predicted count, true count].'''
    tp, fp, fn = c[:, :3].sum(axis=0)
    precision, recall, f1 = prf(tp, fp, fn)
    err = c[:, 3] - c[:, 4]   # predicted − true count
    return {"det_f1": f1, "det_precision": precision, "det_recall": recall,
            "count_mae": float(np.abs(err).mean()), "count_bias": float(err.mean())}
