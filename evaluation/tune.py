'''Threshold × model grid on the validation set, with bootstrap confidence intervals.
Reads pred_val.npz: no pass through the network.'''

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.metrics import confusion_counts,dice_iou,match_objects,prf
from evaluation.postprocess import label_instances,postprocess,predict_labels



def per_img_counts(probs,masks,th):
    '''Post-processing + counts per image → array (n_img, 9):
    px tp/fp/fn, obj tp/fp/fn, merged, n_pred, n_gt.'''

    rows=[]
    for p,m in zip(probs,masks):

        pred=postprocess(p.astype(np.float32),threshold=th)
        gt= m > 0 # bool mask
        px=confusion_counts(torch.from_numpy(pred),torch.from_numpy(gt)) # pixel (tp, fp, fn)

        pred_lab,gt_lab=label_instances(pred),label_instances(gt) 
        obj=match_objects(pred_lab,gt_lab,)  # object (tp, fp, fn, merged)


        rows.append([*px,*obj,pred_lab.max(),gt_lab.max()])

    return np.array(rows,dtype=np.int64)

def per_img_counts_variant(probs, masks, threshold, border, min_distance):
    '''Like per_img_counts, for one post-processing variant (border rule, watershed).'''
    rows = []
    for p, m in zip(probs, masks):
        pred_lab = predict_labels(p.astype(np.float32), threshold, border, min_distance)
        gt = m > 0
        gt_lab = label_instances(gt)
        px = confusion_counts(torch.from_numpy(pred_lab > 0), torch.from_numpy(gt))
        obj = match_objects(pred_lab, gt_lab)
        rows.append([*px, *obj, pred_lab.max(), gt_lab.max()])
    return np.array(rows, dtype=np.int64)

def counts_row(pred_lab, gt):
    '''The 9 per-image counts, in tune.per_img_counts order.'''
    gt_lab = label_instances(gt)
    px = confusion_counts(torch.from_numpy(pred_lab > 0), torch.from_numpy(gt))
    obj = match_objects(pred_lab, gt_lab)
    return [*px, *obj, pred_lab.max(), gt_lab.max()]

def metrics_from_counts(count):
    '''Global metrics from per-image counts (also on resampled rows).'''
    px_tp, px_fp, px_fn, obj_tp, obj_fp, obj_fn, merged, n_pred, n_gt=count.sum(axis=0) # column totals

    dice,iou=dice_iou(px_tp, px_fp, px_fn)
    precision,recall,f1=prf(obj_tp,obj_fp,obj_fn)
    err = count[:, 7] - count[:, 8]   # n_pred − n_gt per image

    return {"dice": dice, "iou": iou, "obj_f1": f1, "obj_precision": precision, "obj_recall": recall,
            "n_merged": int(merged), "count_mae": float(np.abs(err).mean()), "count_bias": float(err.mean())}



def bootstrap_diff(count_a,count_b,metric,n_boot=2000,seed=0):
    '''95% CI of metric(a) − metric(b), resampling the IMAGES.
    Paired: a and b are evaluated on the SAME resampled images.'''

    rng=np.random.default_rng(seed) # local seeded RNG: reproducible
    n=len(count_a)
    diffs=[]
    for _ in range(n_boot):
        idx = rng.integers(0,n,n) # n random images, with repetition
        diffs.append(metrics_from_counts(count_a[idx])[metric] - metrics_from_counts(count_b[idx])[metric]) # metric = dict key
    
    return np.percentile(diffs,[2.5,97.5])



def main():
    parser = argparse.ArgumentParser(description="Threshold × model grid on the validation set, with bootstrap")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)     # e.g. runs/a runs/b
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--out", type=Path, default=Path("runs/tuning_val.csv"))
    args = parser.parse_args()

    counts,rows={},[]
    for run in args.runs:
        d=np.load(run/'pred_val.npz')
        for t in args.thresholds:
            c= per_img_counts(d['probs'],d['masks'],t)
            counts[(run.name,t)]=c # Path("runs/a").name == "a"
            rows.append({"run": run.name, "threshold": t, **metrics_from_counts(c)})

    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    table.to_csv(args.out, index=False)

    a, b = args.runs[-1].name, args.runs[0].name        # last run vs first run
    for metric in ["dice", "obj_f1", "count_mae"]:
        lo, hi = bootstrap_diff(counts[(a, 0.5)], counts[(b, 0.5)], metric)
        print(f"{a} − {b} (threshold 0.5), {metric}: 95% CI [{lo:+.4f}, {hi:+.4f}]")
    for t in args.thresholds:
        if t != 0.5:
            lo, hi = bootstrap_diff(counts[(a, t)], counts[(a, 0.5)], "obj_f1")
            print(f"{a}: threshold {t} − threshold 0.5, obj_f1: 95% CI [{lo:+.4f}, {hi:+.4f}]")

if __name__ == "__main__":
    main()