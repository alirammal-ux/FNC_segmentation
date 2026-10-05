import argparse
import csv
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from models.c_res_unet_logits import CResUnetLogits
from train.dataset import load_split, load_img_and_masks, compute_norm_stats,GreenTrainDataset, GreenEvalDataset, GAMMA_RANGE, WM_SIGMA
from train.losses import BCEDiceLoss
from evaluation.metrics import PixelMetrics


###########################################################
# Train
###########################################################


def train_one_epoch(model,loader,criterion,optimizer,scaler,device,use_amp):
    '''One training epoch over all the crops.
    Returns loss, BCE and soft Dice averaged over the batches.'''

    model.train() # BatchNorm: batch stats, updates the running averages
    tot_metrics ={
        'loss':0.,
        'bce': 0.,
        'dice':0.
    }
    n_batch=0


    for batch in loader:
        x,label=batch[0],batch[1]
        x=x.to(device,non_blocking=True) # non_blocking: the copy overlaps compute
        label=label.to(device,non_blocking=True)
        weight=batch[2].to(device,non_blocking=True) if len(batch)==3 else None # weight maps (--weight_maps)

        optimizer.zero_grad(set_to_none=True) # gradients accumulate: reset every step

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): # float16 convolutions if use_amp
            logits=model(x)
            loss,bce,dice=criterion(logits,label,weight)

        scaler.scale(loss).backward() # scaled loss → gradients
        scaler.step(optimizer) # unscale + step (skipped if inf)
        scaler.update() # adapt the scale factor

        # enabled=False (CPU): plain backward + step

        # running sums
        tot_metrics['loss']+=loss.item()
        tot_metrics['dice']+=dice.item()
        tot_metrics['bce']+=bce.item() # .item(): tensor → Python number (GPU → CPU)
        n_batch+=1

    return {k: v / n_batch for k, v in tot_metrics.items()} # mean over batches



###########################################################
# Validate
###########################################################

@torch.no_grad()
def validate(model,loader,criterion,device,use_amp):
    '''Validation on full 1200×1600 images (batch_size=1): loss, global and mean Dice/IoU.'''
    
    model.eval() # BatchNorm: running stats, no updates
    metrics=PixelMetrics()
    total_loss=0.

    for x,label,_name in loader:
        assert x.shape[0] == 1, "validate can handle 1 img at a time, but here batch_size > 1 "
        x=x.to(device)
        label=label.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits=model(x)
            loss,bce,dice=criterion(logits,label)
        
        total_loss+=loss.item()
        metrics.update((logits > 0).squeeze(0), (label > 0.5).squeeze(0)) # (1, 1, H, W) → (1, H, W), bool
    
    res=metrics.compute()
    res['loss']=total_loss/len(loader)

    return res

# note:
# val loss uses the soft Dice per full image (then averaged), train loss per batch of 8 crops:
# absolute values are not comparable;
# watch the val-loss trend (rising while the train loss falls = overfitting).




###########################################################
# Main
###########################################################

def parse_args():
    p = argparse.ArgumentParser(description="Training della c-ResUNet (logit) su Green")

    # data and output
    p.add_argument("--images_dir", type=Path, required=True)
    p.add_argument("--masks_dir", type=Path, required=True)
    p.add_argument("--split_json", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crops_per_epoch", type=int, default=1200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--no_gamma", action="store_true") # gain only (green_v3)
    p.add_argument("--norm", choices=["global", "per_image"], default="global") # per_image: green_v4
    p.add_argument("--base_ch", type=int, default=16) # 32: green_v5
    p.add_argument("--weight_maps", action="store_true") # weight maps of Morelli et al. 2021 in the BCE (green_v7)

    # debug only: use the first N images (0 = all)
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--val_limit", type=int, default=0)
    return p.parse_args()


def main():

    # setup
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" # AMP on GPU only
    torch.backends.cudnn.benchmark = True # cuDNN picks the fastest conv algorithm (fixed input sizes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # data loading
    train_names, val_names = load_split(args.split_json) # train/val names

    # debug only
    if args.train_limit:
        train_names = train_names[:args.train_limit]
    if args.val_limit:
        val_names = val_names[:args.val_limit]

    t0 = time.time()


    images, masks = load_img_and_masks(args.images_dir, args.masks_dir, train_names)
    v_images, v_masks = load_img_and_masks(args.images_dir, args.masks_dir, val_names)
    mean, std = compute_norm_stats(images) # train images only
    mean, std = float(mean), float(std)             # np.float32 → Python float (JSON, torch.load)

    print(f"dati: {len(train_names)} train, {len(val_names)} val, caricati in {time.time() - t0:.0f} s | "
          f"mean {mean:.4f} std {std:.4f} | device {device.type}, amp {use_amp}")

    # datasets
    train_ds = GreenTrainDataset(images, masks, mean, std,
                                 crops_per_epoch=args.crops_per_epoch, augment=not args.no_augment,
                                 gamma_range=(1.0, 1.0) if args.no_gamma else GAMMA_RANGE, norm=args.norm,
                                 weight_maps=args.weight_maps)

    val_ds = GreenEvalDataset(v_images, v_masks, val_names, mean, std, norm=args.norm)

    pin = device.type == "cuda" # pinned memory only on GPU

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=pin, persistent_workers=args.num_workers > 0)
    
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=pin)

    # model, optimizer, scheduler, scaler
    model = CResUnetLogits(base_ch=args.base_ch).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # config saved next to the results
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(mean=mean, std=std, device=device.type, amp=use_amp,
                  n_train=len(train_names), n_val=len(val_names))
    if args.weight_maps:
        config["wm_sigma"] = WM_SIGMA
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    #######################################
    # Training loop
    #######################################

    best_dice, best_epoch, no_improve = -1.0, 0, 0
    with open(args.out_dir / "log.csv", "w", newline="") as f_log:

        # created at the first epoch
        writer = None

        for epoch in range(1, args.epochs + 1):

            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"] # lr used in THIS epoch

            # train + validation
            tr = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
            va = validate(model, val_loader, criterion, device, use_amp)
            scheduler.step() # updates the lr, not the weights (those are updated per batch)


            row = {"epoch": epoch, "lr": lr,
                   **{f"train_{k}": v for k, v in tr.items()},
                   **{f"val_{k}": v for k, v in va.items()},
                   "time_s": time.time() - t0}
            if writer is None: # first epoch: header from the row keys
                writer = csv.DictWriter(f_log, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f_log.flush() # write to disk now

            improved = va['dice_global']>best_dice                         
            if improved:
                # new best: reset the early-stopping counter
                best_dice, best_epoch, no_improve = va["dice_global"], epoch, 0

                # save the best checkpoint so far
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": va,
                            "mean": mean, "std": std, "config": config},
                           args.out_dir / "best.pt")
            else:
                no_improve += 1

            print(f"ep {epoch:3d} | lr {lr:.2e} | train loss {tr['loss']:.4f} | "
                  f"val loss {va['loss']:.4f} dice {va['dice_global']:.4f} | {row['time_s']:.0f} s {'*' if improved else ''}")


            # early stopping
            if no_improve>=args.patience:                                 
                print(f"early stopping: nessun miglioramento da {args.patience} epoche")
                break

    # save the final checkpoint
    torch.save({"model": model.state_dict(), "epoch": epoch, "val": va,
                "mean": mean, "std": std, "config": config},
               args.out_dir / "last.pt")
    print(f"fine: miglior Dice val {best_dice:.4f} all'epoca {best_epoch}")


if __name__=='__main__':
    main()


