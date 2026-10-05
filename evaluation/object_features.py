'''Features of every object predicted by the CNN (one row per object), saved as objects_<split>.csv.
First of the three object-classifier files: object_features → select_object_classifier → object_classifier.'''

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

from evaluation.metrics import overlap_tables
from evaluation.postprocess import label_instances,postprocess,predict_labels
from train.dataset import load_img_and_masks

# one column per object feature (the classifier input)
FEATURES=["area",
        "mean_prob",
        "contrast",
        "rel_contrast",
        "local_bg",
        "solidity",
        "eccentricity",
        "nn_dist",
        "n_objects",
        'intensity']


def label_objects(pred_lab, prob, img, gt_lab, ring=10):
    '''One row per object of a label image of ONE image: features + label (1 = TP, 0 = FP)
    + border (object touches the image border: metadata, not a feature).
    Works for any model: pred_lab = its objects (1..N), prob = its probability map (for mean_prob).'''

    signal = img.max(axis=2).astype(np.float32)     # axis=2 are the channels (G for Green)
    background = float(np.median(signal))           # cells are <1% of the pixels → median = background

    # no predicted objects → empty table
    n = int(pred_lab.max())
    if n == 0:
        return pd.DataFrame(columns=FEATURES + ["label", "border"])

    # label: a predicted object is a TP if it has a true partner with IoU > 0.5
    _, iou = overlap_tables(pred_lab, gt_lab) # iou[pred, true]
    is_tp = (iou > 0.5).any(axis=1)  #one bool  per PREDICTED cell

    # per-object measures on the signal (area, shape, intensity) and on the CNN probabilities
    sig_props = measure.regionprops(pred_lab, intensity_image=signal)
    prob_props = measure.regionprops(pred_lab, intensity_image=prob)

    # distance from each object to the nearest other object
    centroids = np.array([p.centroid for p in sig_props])
    dist = np.linalg.norm(centroids[:, None] - centroids[None], axis=2)   # (n, n)
    np.fill_diagonal(dist, np.inf)                     # ignore the distance to itself
    nn = dist.min(axis=1) if n > 1 else np.full(n, 2000.0)   # single object: image diagonal (2000 px)

    rows = []
    for k, (sp, pp) in enumerate(zip(sig_props, prob_props)):
        # object touching the image border (original bbox, before enlarging it)
        r0, c0, r1, c1 = sp.bbox
        border = r0 == 0 or c0 == 0 or r1 == pred_lab.shape[0] or c1 == pred_lab.shape[1]

        # bbox enlarged by `ring` px, clipped to the image
        r0, c0 = max(r0 - ring, 0), max(c0 - ring, 0)
        r1, c1 = min(r1 + ring, pred_lab.shape[0]), min(c1 + ring, pred_lab.shape[1])
        crop = pred_lab[r0:r1, c0:c1]

        # ring of `ring` px around the object, only on background pixels (other objects excluded)
        grown = ndimage.binary_dilation(crop == sp.label, iterations=ring)
        ring_px = signal[r0:r1, c0:c1][grown & (crop == 0)]

        rows.append({
            "area": sp.area,
            "mean_prob": pp.intensity_mean,
            "contrast": sp.intensity_mean / background,          # 1 = as bright as the background
            "local_bg": (np.median(ring_px) / background) if ring_px.size else 1.0,
            "solidity": sp.solidity,
            "eccentricity": sp.eccentricity,
            "nn_dist": nn[k],
            "n_objects": n,
            "label": int(is_tp[k]),
            'intensity': sp.intensity_mean / 255,                # absolute intensity
            "border": border,
        })

    # contrast relative to the typical predicted cell of this image
    df = pd.DataFrame(rows)
    df["rel_contrast"] = df["contrast"] / df["contrast"].median()
    return df[FEATURES + ["label", "border"]]


def image_objects(prob, img, gt_mask, threshold=0.5, ring=10, min_distance=None):
    '''One row per object predicted by the CNN in ONE image (label_objects on its post-processed prediction).
    min_distance: watershed between seeds at least this far apart (px), as in the delivered pipeline; None = no watershed.'''
    # predicted objects: same post-processing as the evaluation
    if min_distance is None:
        pred_lab = label_instances(postprocess(prob, threshold=threshold))
    else:
        pred_lab = predict_labels(prob, threshold, min_distance=min_distance)
    return label_objects(pred_lab, prob, img, label_instances(gt_mask > 0), ring)


def build_table(pred_npz, images_dir, masks_dir, threshold=0.5, min_distance=None):
    '''All predicted objects of one split (CNN probabilities and masks from the npz).
    Extra columns: "image" (image index) and "obj" (label id of the object).'''
    d = np.load(pred_npz)
    names = [str(n) for n in d["names"]]
    images, _ = load_img_and_masks(images_dir, masks_dir, names)

    parts = []
    for i, (p, img, m) in enumerate(zip(d["probs"], images, d["masks"])):
        t = image_objects(p.astype(np.float32), img, m, threshold, min_distance=min_distance)
        if len(t):
            # regionprops follows the label order → row k is object k + 1
            parts.append(t.assign(image=i, obj=np.arange(1, len(t) + 1)))
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Object features of the CNN predictions → objects_<split>.csv")
    parser.add_argument("--run", type=Path, default=Path("runs/green_v2"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])   # reads pred_<split>.npz
    parser.add_argument("--images_dir", type=Path, default=Path("data/green/trainval/images"))
    parser.add_argument("--masks_dir", type=Path, default=Path("data/cleaned_masks/trainval/Green/masks"))
    parser.add_argument("--threshold", type=float, default=0.5)   # CNN probability threshold
    args = parser.parse_args()
    out = args.run / "object_classifier"
    out.mkdir(exist_ok=True)

    for split in args.splits:
        table = build_table(args.run / f"pred_{split}.npz", args.images_dir, args.masks_dir, args.threshold)
        table.to_csv(out / f"objects_{split}.csv", index=False)
        print(f"{split}: {len(table)} objects | TP {int(table.label.sum())} | FP {int((table.label == 0).sum())} "
              f"| on the border {int(table.border.sum())}")


if __name__ == "__main__":
    main()
