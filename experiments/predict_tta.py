'''Test-time augmentation (TTA) for a checkpoint: the network sees each image in the 8 symmetries of the square
(horizontal flip × rotations by 90°, the same group used as augmentation in training), each output is brought back to
the original orientation and the 8 probabilities are averaged. Same output as evaluation/predict.py (pred_<split>.npz),
so every other script works unchanged. The test split (every image of the folder) is looked at only on the test day.'''

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.predict import load_model
from train.dataset import GreenEvalDataset, load_img_and_masks, load_split

# (flip, k): horizontal flip, then k rotations by 90°; (False, 0) alone = the plain prediction of predict.py
SYMMETRIES = [(flip, k) for flip in (False, True) for k in range(4)]


@torch.no_grad()
def predict_tta(model, dataset, device, symmetries=SYMMETRIES):
    '''Mean probability over the symmetries (N, H, W) float16, and names, one image at a time.'''
    probs, names = [], []
    for x, _, name in DataLoader(dataset):
        x = x.to(device)                                     # (1, 3, H, W)
        total = torch.zeros(x.shape[-2:], device=device)
        for flip, k in symmetries:
            xt = torch.flip(x, dims=[3]) if flip else x      # dim 3 = columns: horizontal flip
            p = torch.sigmoid(model(torch.rot90(xt, k, dims=[2, 3])))   # 1600×1200 after an odd k: still divisible by 8
            p = torch.rot90(p, -k, dims=[2, 3])              # undo in reverse order: rotation first, then flip
            total += (torch.flip(p, dims=[3]) if flip else p)[0, 0]
        probs.append((total / len(symmetries)).cpu().numpy().astype(np.float16))
        names.append(name[0])                                # the DataLoader wraps the name in a list
    return np.stack(probs), names


def main():
    parser = argparse.ArgumentParser(description="TTA probability maps of one checkpoint (8 symmetries) → pred_<split>.npz")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--masks_dir", type=Path, required=True)
    parser.add_argument("--split_json", type=Path, default=Path("splits/green_split.json"))
    parser.add_argument("--subset", choices=["train", "val", "test"], default="val")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, epoch, norm = load_model(args.ckpt, device)
    if args.subset == "test":
        names = sorted(p.name for p in args.images_dir.glob("*.png"))   # every image of the folder, as evaluate.py
    else:
        train_names, val_names = load_split(args.split_json)
        names = val_names if args.subset == "val" else train_names
    images, masks = load_img_and_masks(args.images_dir, args.masks_dir, names)
    dataset = GreenEvalDataset(images, masks, names, mean, std, norm=norm)

    t0 = time.time()
    probs, pred_names = predict_tta(model, dataset, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, probs=probs, masks=masks, names=np.array(pred_names))
    print(f"{args.ckpt} (epoch {epoch}) | TTA, {len(SYMMETRIES)} symmetries | {len(pred_names)} images in "
          f"{time.time() - t0:.0f} s on {device} | saved to {args.out}")


if __name__ == "__main__":
    main()
