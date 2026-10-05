import json
from pathlib import Path

import numpy as np
import skimage.io as io
import torch
from scipy import ndimage
from skimage import measure
from torch.utils.data import Dataset



##################################################################
# Load
##################################################################

IMG_H,IMG_W=1200,1600

def load_split(split_json:Path): 
    'Return the train and val name lists.'
    with open(split_json) as file:
        s=json.load(file)
    return s['train'],s['val']


def load_img_and_masks(img_dir:Path,mask_dir:Path,names:list):
    '''Read images and masks into uint8 arrays.'''
    n=len(names)
    # preallocated arrays
    A_img=np.empty((n,IMG_H,IMG_W,3),dtype=np.uint8)
    A_mask=np.empty((n,IMG_H,IMG_W),dtype=np.uint8)

    for i,name in enumerate(names):
        img= io.imread(img_dir/name)
        mask= io.imread(mask_dir/name)
        assert img.shape == (IMG_H,IMG_W,3) , f'{name} img incorrect shape'
        assert mask.shape == (IMG_H,IMG_W) , f'{name} mask incorrect shape'

        A_img[i]=img
        A_mask[i]=mask > 0 # bool → uint8 {0,1}

    return A_img,A_mask


def compute_norm_stats(images:np.ndarray):
    '''Global mean and std over all channels.'''
    s=np.zeros(3) # sum per channel
    s2=np.zeros(3) # sum of squares per channel
    n_px=0

    for img in images:
        x=img.reshape(-1,3).astype(np.float64) / 255.0 #   (H*W,3)
        s+= x.sum(axis=0)
        s2+=(x**2).sum(axis=0) # sum of squares per channel
        n_px+=x.shape[0]

    mean=s.sum()/(3*n_px)
    std=np.sqrt(s2.sum()/(3*n_px)-mean ** 2)
    
    return mean.astype(np.float32),std.astype(np.float32)



##################################################################
# Crop
##################################################################

CROP=512

def random_int(low,high):
    '''Random int in [low, high], torch RNG.'''
    return int(torch.randint(low,high+1,(1,)).item())

def sample_crop(images,masks,crop=CROP,return_index=False,weights=None):
    '''Random crop. Returns views (no copy): do not modify in place.
    weights (optional): weight maps, cropped at the same place and returned last.'''
    i=random_int(0,len(images)-1)
    top=random_int(0,IMG_H-crop)
    left=random_int(0,IMG_W-crop)
    img=images[i,top:top+crop,left:left+crop]
    mask=masks[i,top:top+crop,left:left+crop]

    out=(img,mask,i) if return_index else (img,mask) # i = source image
    if weights is not None:
        out=out+(weights[i,top:top+crop,left:left+crop],)
    return out


##################################################################
# Weight maps (Morelli et al. 2021, Alg. 1)
##################################################################

WM_SIGMA=15. # px ≈ mean radius of the Green cells (mean equivalent diameter 30.6 px)

def weight_map(mask,sigma=WM_SIGMA):
    '''Each cell adds exp(-d²/2σ²), d = distance from the cell (0 inside it); contributions are summed, plus 1.
    Far background → 1, cells ~2, background between close cells higher: errors there cost more.'''
    lab=measure.label(mask>0,connectivity=1)
    w=np.ones(mask.shape,dtype=np.float32)
    r=int(3*sigma) # beyond 3σ the contribution is < 0.012
    for p in measure.regionprops(lab):
        r0,c0,r1,c1=p.bbox
        r0,c0=max(r0-r,0),max(c0-r,0)
        r1,c1=min(r1+r,mask.shape[0]),min(c1+r,mask.shape[1])
        d=ndimage.distance_transform_edt(lab[r0:r1,c0:c1]!=p.label) # distance from this cell, in a window around it
        w[r0:r1,c0:c1]+=np.exp(-d**2/(2*sigma**2))
    return w



##################################################################
# Augmentation (train only)
##################################################################


def random_uniform(low,high):
    return low+(high-low)*torch.rand(1).item()

def augment_geometric(img,mask):
    '''One of the 8 square symmetries (flip + 90° rotation), same for img and mask.'''

    if torch.rand(1).item() < 0.5:
        img=img[:,::-1] # reverse columns → horizontal flip
        mask=mask[:,::-1] 

    k=random_int(0,3) # number of 90° rotations
    img=np.rot90(img,k,axes=(0,1))
    mask=np.rot90(mask,k,axes=(0,1))

    return img,mask

GAMMA_RANGE=(0.8,1.25) # (1.0, 1.0) = no gamma

def augment_photometric(img,gain_range=(0.75,1.5),gamma_range=GAMMA_RANGE,return_params=False):
    '''Random gain (brightness) and gamma (contrast); image only.'''
    a=random_uniform(*gain_range)
    g=random_uniform(*gamma_range)

    img=np.clip(img*a,0.0,1.0) # saturate at 1, like a sensor
    if return_params:
        return img ** g,a,g
    return img ** g


##################################################################
# Per-image normalization
##################################################################

CLIP=(-5.0,30.0) # cells ~3-7, rare artifacts up to ~76

def image_quantiles(images):
    '''Q25, Q50, Q75 of V = max over channels, per image, in [0, 1] (exact order statistics).'''
    q=np.empty((len(images),3))
    for i,img in enumerate(images):
        cdf=np.cumsum(np.bincount(img.max(axis=2).ravel(),minlength=256))
        q[i]=np.searchsorted(cdf,np.array([0.25,0.5,0.75])*cdf[-1])/255.
    return q

def per_image_stats(q,a=1.0,g=1.0):
    '''Center and scale of V after gain a and gamma g (quantiles commute with monotone transforms).'''
    t=np.clip(a*q,0.0,1.0)**g
    return float(t[1]),float(max((t[2]-t[0])/1.349,1e-3)) # Python floats keep float32 tensors


##################################################################
# Dataset
##################################################################


def to_tensor(img,mask,mean,std,clip=None):
    """img float32 (H, W, 3) in [0, 1] and mask (H, W) to tensors:
    normalized img (3, H, W) float32, mask (1, H, W) float32."""

    img= (img-mean)/std
    if clip is not None:
        img=np.clip(img,*clip)

    # transpose: for each new axis, which old axis goes there
    img=np.ascontiguousarray(img.transpose(2,0,1)) # (H, W, C) --> (C, H, W)


    # [None] adds a leading axis of length 1
    mask=np.ascontiguousarray(mask,dtype=np.float32)[None] # (H, W) --> (1, H, W) 

    return torch.from_numpy(img),torch.from_numpy(mask)


class GreenTrainDataset(Dataset):
    '''
    - Random 512 x 512 crop
    - index ignored
    - each call produces a new crop
    '''

    def __init__(self,images,masks,mean,std,crops_per_epoch=1200,crop=CROP,augment=True,gamma_range=GAMMA_RANGE,norm="global",
                 weight_maps=False):
        # 1200 crops ≈ one pass over the train pixels (a 512×512 crop covers ~13.6% of an image)
        assert norm in ("global","per_image"), norm

        self.images=images
        self.masks=masks
        self.mean=mean
        self.std=std
        self.crops_per_epoch=crops_per_epoch
        self.crop=crop
        self.augment=augment
        self.gamma_range=gamma_range
        self.norm=norm
        self.q=image_quantiles(images) if norm=="per_image" else None # whole-image stats, not the crop's
        # weight maps computed once per image (float16 to save RAM: values ~1-4)
        self.weights=np.stack([weight_map(m) for m in masks]).astype(np.float16) if weight_maps else None

    def __len__(self):
        return self.crops_per_epoch

    def __getitem__(self,index):
        out=sample_crop(self.images,self.masks,self.crop,return_index=True,weights=self.weights)
        img,mask,i=out[:3]
        weight=out[3] if self.weights is not None else None
        img= img.astype(np.float32)/255. # float copy: the original array is untouched

        a,g=1.0,1.0 # no photometric change
        if self.augment:
            if weight is None:
                img,mask= augment_geometric(img,mask)
            else: # the weight map follows the same symmetry as the mask (stacked as a second channel)
                img,both= augment_geometric(img,np.stack([mask,weight],axis=-1).astype(np.float32))
                mask,weight=both[...,0],both[...,1]
            img,a,g=augment_photometric(img,gamma_range=self.gamma_range,return_params=True)

        if self.norm=="per_image":
            center,scale=per_image_stats(self.q[i],a,g)
            x,y=to_tensor(img,mask,center,scale,clip=CLIP)
        else:
            x,y=to_tensor(img,mask,self.mean,self.std)
        if weight is None:
            return x,y
        return x,y,torch.from_numpy(np.ascontiguousarray(weight,dtype=np.float32)[None]) # (1, H, W), like the mask



class GreenEvalDataset(Dataset):
    '''Full 1200×1600 images, no augmentation: validation and test.'''

    def __init__(self,images,masks,names,mean,std,norm="global"):
        assert norm in ("global","per_image"), norm
        self.images=images
        self.masks=masks
        self.mean=mean
        self.std=std
        self.names=names
        self.norm=norm
        self.q=image_quantiles(images) if norm=="per_image" else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img=self.images[index].astype(np.float32)/255.
        if self.norm=="per_image":
            center,scale=per_image_stats(self.q[index])
            img,mask=to_tensor(img,self.masks[index],center,scale,clip=CLIP)
        else:
            img,mask=to_tensor(img,self.masks[index],self.mean,self.std)

        return img,mask,self.names[index]










    
    



