"""Threshold baselines on the G channel (Otsu; robust = median + k·MAD, k chosen on train),
with the same post-processing and metrics as the CNN."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skimage.filters import threshold_otsu

from evaluation.metrics import ObjectMetrics, PixelMetrics, dice_iou
from evaluation.postprocess import label_instances, postprocess
from train.dataset import load_img_and_masks, load_split


def signal_channel(img):
    """Signal channel for Green: G."""
    return img[..., 1]


def robust_stats(g):
    """Median and MAD: background level and its typical spread."""
    med = float(np.median(g))
    mad = float(np.median(np.abs(g.astype(np.float32) - med)))
    return med, max(mad, 1.0)  # MAD = 0 only in flat images


def robust_threshold(g, k):
    """Background + k typical deviations."""
    med, mad = robust_stats(g)
    return med + k * mad


def fit_k(images, masks, ks):
    """Pick k on TRAIN by global pixel Dice, from per-image G histograms (no re-thresholding)."""
    stats, hist_cell, hist_bg = [], [], []
    for img, m in zip(images, masks):
        g = signal_channel(img)
        stats.append(robust_stats(g))
        hist_cell.append(np.bincount(g[m > 0], minlength=256))
        hist_bg.append(np.bincount(g[m == 0], minlength=256))

    best_k, best_dice = None, -1.0
    for k in ks:
        tp = fp = fn = 0
        for (med, mad), hc, hb in zip(stats, hist_cell, hist_bg):
            start = int(np.clip(np.floor(med + k * mad) + 1, 0, 256))  # first value above the threshold
            tp += hc[start:].sum()
            fp += hb[start:].sum()
            fn += hc[:start].sum()
        dice = dice_iou(tp, fp, fn)[0]
        if dice > best_dice:
            best_k, best_dice = float(k), float(dice)
    return best_k, best_dice


def evaluate(images, masks, threshold_fn):
    """threshold_fn(g) per image, then the CNN's post-processing and metrics."""
    pixel, objects = PixelMetrics(), ObjectMetrics()
    for img, m in zip(images, masks):
        g = signal_channel(img).astype(np.float32)
        pred = postprocess(g, threshold=threshold_fn(g))
        pixel.update(torch.from_numpy(pred), torch.from_numpy(m > 0))
        objects.update(label_instances(pred), label_instances(m > 0))
    return {**pixel.compute(), **objects.compute()}


def main():
    p = argparse.ArgumentParser(description="Baseline a soglia (Otsu e robusta) sulla validation")
    p.add_argument("--images_dir", type=Path, required=True)
    p.add_argument("--masks_dir", type=Path, required=True)
    p.add_argument("--split_json", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/baselines/threshold_val.json"))
    args = p.parse_args()

    train_names, val_names = load_split(args.split_json)
    train_imgs, train_masks = load_img_and_masks(args.images_dir, args.masks_dir, train_names)
    val_imgs, val_masks = load_img_and_masks(args.images_dir, args.masks_dir, val_names)

    k, dice_train = fit_k(train_imgs, train_masks, ks=np.arange(1.0, 30.5, 0.5))
    print(f"k scelto sul train: {k} (Dice di pixel sul train {dice_train:.3f})")

    results = {
        "otsu": evaluate(val_imgs, val_masks, threshold_otsu),
        "robust": evaluate(val_imgs, val_masks, lambda g: robust_threshold(g, k)),
    }
    results["robust"]["k"] = k
    for name, r in results.items():
        print(name, {metric: round(v, 3) for metric, v in r.items()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"salvato in {args.out}")


if __name__ == "__main__":
    main()
