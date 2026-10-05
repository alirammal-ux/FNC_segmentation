'''Probability maps of a checkpoint on a split, saved to disk,
so thresholds and post-processing can be tried without the network.'''

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


from models.c_res_unet_logits import CResUnetLogits
from models.c_res_unet_small_rf import CResUnetSmallRF
from train.dataset import GreenEvalDataset,load_img_and_masks,load_split


def load_model(ckpt_path,device):
    '''Model with the checkpoint weights, plus the train mean/std, epoch and normalization saved in it.'''
    ckpt=torch.load(ckpt_path,map_location=device)
    cfg=ckpt.get('config',{}) # older checkpoints: no base_ch/norm in the config → defaults

    arch=CResUnetSmallRF if cfg.get('small_rf',False) else CResUnetLogits # small_rf: light network (no 5×5 block)
    model=arch(base_ch=cfg.get('base_ch',16)).to(device) # base channels saved in the config (default 16)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model,ckpt['mean'],ckpt['std'],ckpt['epoch'],cfg.get('norm','global')

@torch.no_grad()
def predict_probs(model,dataset,device):
    '''Probabilities (N, H, W) float16 and names, one image at a time.'''
    probs,names=[],[]
    for x,_,name in DataLoader(dataset):
        logits=model(x.to(device))
        p=torch.sigmoid(logits).squeeze() # (1, 1, H, W) → (H, W)
        probs.append(p.cpu().numpy().astype(np.float16)) # .cpu(): NumPy only reads CPU memory
        names.append(name[0]) # the DataLoader wraps the name in a list

    return np.stack(probs),names # (N, H, W)


def main():
    parser = argparse.ArgumentParser(description="ProbMaps of one ckpt on train or val")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--masks_dir", type=Path, required=True)
    parser.add_argument("--split_json", type=Path, required=True)
    parser.add_argument("--subset", choices=["train", "val", "all"], default="val") # all: every image of images_dir (test)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()


    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,mean,std,epoch,norm=load_model(args.ckpt,device)

    train_names,val_names=load_split(args.split_json)
    names=val_names if args.subset == 'val' else train_names
    if args.subset == 'all':
        names=sorted(p.name for p in args.images_dir.glob('*.png'))
    images,masks= load_img_and_masks(args.images_dir,args.masks_dir,names)
    dataset=GreenEvalDataset(images,masks,names,mean,std,norm=norm)


    t0=time.time()
    probs,pred_names=predict_probs(model,dataset,device)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,probs=probs,masks=masks,names=np.array(pred_names))#Save several arrays into a single file in compressed .npz format.
    print(f"{args.ckpt} (epoch {epoch}) | {len(pred_names)} images in {time.time() - t0:.0f} s | saved to {args.out}")

if __name__=='__main__':
    main()






    
