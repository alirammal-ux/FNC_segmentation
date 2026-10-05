'''5-fold split of the whole trainval, stratified by number of cells (quartiles), for the cross-validation.
One json per fold with the same keys as splits/green_split.json ("train" / "val" = the held-out fold),
so train.training and evaluation.predict work unchanged.'''

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from train.split import count_from_coco


def kfold(names, counts, n_splits, seed):
    '''List of (train names, held-out names), one per fold.'''
    count_quartile = pd.qcut(counts, q=4, labels=False, duplicates='drop')
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(names, count_quartile)
    names = np.array(names)
    return [(sorted(names[tr].tolist()), sorted(names[va].tolist())) for tr, va in folds]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-fold split of trainval, stratified by number of cells")
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--coco_json", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("splits"))
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names, counts, c = count_from_coco(args.images_dir, args.coco_json)
    folds = kfold(names, counts, args.n_splits, args.seed)

    held = [n for _, va in folds for n in va]
    assert sorted(held) == sorted(names), "every image must be held out exactly once"
    print(pd.DataFrame({f"fold {k}": pd.Series([c[n] for n in va]).describe() for k, (_, va) in enumerate(folds)}).round(1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k, (tr, va) in enumerate(folds):
        with open(args.out_dir / f"green_cv{k}.json", "w") as f:
            json.dump({"seed": args.seed, "fold": k, "n_splits": args.n_splits, "train": tr, "val": va}, f, indent=2)
    print(f"saved {args.out_dir}/green_cv0..{args.n_splits - 1}.json")


# usage:
# python -m train.kfold_split \
#     --images_dir data/green/trainval/images \
#     --coco_json data/green/trainval/ground_truths/COCO/annotations_green_trainval.json
