import argparse 
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import pprint as pp


def count_from_coco(img_dir:Path,coco_json:Path):
    '''Number of cells per image, from the official COCO count.'''
    
    names=sorted(p.name for p in img_dir.glob('*.png')) # all .png names, alphabetical order (deterministic)
    with open(coco_json) as file_coco:
        coco=json.load(file_coco)
        
    dict_img_count={img['image_id'] + '.png': img['count'] for img in coco['annotations']}
    counts= [dict_img_count[name] for name in names]
    return names,counts,dict_img_count



def split(names,counts,val_size,seed):
    count_quartile=pd.qcut(counts,q=4,labels=False,duplicates='drop')
    train_names,val_names=train_test_split(names,test_size=val_size,stratify=count_quartile,random_state=seed)
    return sorted(train_names),sorted(val_names)



def summarize(names,counts,train_names,val_names,c:dict):
    assert len(train_names) + len(val_names) == len(names)
    assert all([img not in val_names for img in train_names]), 'un img è sia in train che in val'


    df=pd.DataFrame({
        'train':pd.Series([c[n] for n in train_names]).describe(),
        'val':pd.Series([c[n] for n in val_names]).describe(),
    })

    return df.round(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train/val stratificato per numero di cellule")
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--coco_json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names, counts,c = count_from_coco(args.images_dir, args.coco_json)
    train_names, val_names = split(names, counts, args.val_size, args.seed)
    print(summarize(names, counts, train_names, val_names,c))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "seed": args.seed,
            "val_size": args.val_size,
            "train": train_names,
            "val": val_names,
        }, f, indent=2)
    print(f"salvato in {args.out}")



# usage:

#python -m train.split \
    # --images_dir data/green/trainval/images \
    # --coco_json data/green/trainval/ground_truths/COCO/annotations_green_trainval.json \
    # --out splits/green_split.json