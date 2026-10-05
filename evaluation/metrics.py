import numpy as np
import torch

###################################################
# Confusion matrix
###################################################

def confusion_counts(pred:torch.Tensor,target:torch.Tensor):
    '''pred, target: bool tensors of the same shape (one image).'''
    tp=(pred & target).sum().item() # True in both

    fp=(pred & ~target).sum().item() # predicted, not true
    fn=(~pred & target).sum().item() # true, not predicted

    return tp,fp,fn


###################################################
# Dice = 2TP / (2TP + FP + FN) 
#
# IoU = TP / (TP + FP + FN)
###################################################



def dice_iou(tp,fp,fn):
    if tp+fp+fn==0:
        return 1.0,1.0

    dice= 2*tp/(2*tp+fp+fn)
    iou=tp/(tp+fp+fn)

    return dice,iou


####################################################


class PixelMetrics:
    '''Accumulates counts image by image.
    compute(): global and per-image mean Dice/IoU.'''

    def __init__(self):
        self.tp=0
        self.fp=0
        self.fn=0
        self.per_img=[] # (dice, iou) per image

    def update(self,pred,target):
        '''One image at a time, NOT a batch.'''
        tp,fp,fn=confusion_counts(pred,target)
        self.tp+=tp
        self.fp+=fp
        self.fn+=fn
        self.per_img.append(dice_iou(tp,fp,fn)) 

    def compute(self):
        dice_glob,iou_glob=dice_iou(self.tp,self.fp,self.fn) 
        per_img=np.array(self.per_img)
        return{
            'dice_global':dice_glob,
            "iou_global": iou_glob,
            "dice_mean": float(per_img[:, 0].mean()), # mean over images
            "iou_mean": float(per_img[:,1].mean()) # mean over images
        }      


###################################################
# Object-level and counting metrics
###################################################

# Pixel Dice ignores whether cells are separated, and a small missed cell barely moves it,
# but for counting it is a whole error.


def match_objects(pred_lab,gt_lab,iou_thr=0.5):
    '''pred_lab, gt_lab: label images (H, W), 0 = background, 1..N = objects.
    Pair = IoU > iou_thr (one-to-one for 0.5).
    return tp,fp,fn,n_merged'''

    n_p,n_g=int(pred_lab.max()),int(gt_lab.max())

    # intersection matrix m[i, j] = pixels with pred i and true j

    # flatten, then encode each pixel's (pred i, true j) pair as one number
    idx=pred_lab.ravel().astype(np.int64)*(n_g+1)+gt_lab.ravel()

    # bincount = pixels per pair (the intersections);
    # reshape → table m[pred, true]
    m=np.bincount(idx,minlength=(n_p+1)*(n_g+1)).reshape(n_p+1,n_g+1)

    area_p=m.sum(axis=1)[1:] # row sums → area of each pred object ([1:] drops background)
    area_g=m.sum(axis=0)[1:] # column sums → area of each true object
    m=m[1:,1:] # drop background row and column

    # broadcasting → n_p × n_g table of area sums per pair
    union=area_p[:,None]+area_g[None,:] -m # |A ∪ B| = |A| + |B| − |A ∩ B|
    iou=m/np.maximum(union,1) # avoids division by zero
    
    # pairs with IoU > iou_thr
    tp= int((iou>iou_thr).sum())

    # FP = predictions without a partner
    fp= n_p - tp

    # FN = true cells without a partner
    fn=n_g-tp

    # merges: a prediction covering > half of two or more true cells
    # (predictions do not overlap, so > half of a true cell has one owner)
    cover=m/np.maximum(area_g[None,:],1) # fraction of true cell j covered by pred i
    n_merged=int(((cover>0.5).sum(axis=1)>=2).sum())

    return tp,fp,fn,n_merged


def overlap_tables(pred_lab, gt_lab):
    '''Intersections and IoU between all objects (as in match_objects).'''
    n_p, n_g = int(pred_lab.max()), int(gt_lab.max())
    idx = pred_lab.ravel().astype(np.int64) * (n_g + 1) + gt_lab.ravel()
    inter = np.bincount(idx, minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)
    area_p, area_g = inter.sum(axis=1)[1:], inter.sum(axis=0)[1:]
    inter = inter[1:, 1:]
    iou = inter / np.maximum(area_p[:, None] + area_g[None, :] - inter, 1)
    return inter, iou


def prf(tp, fp, fn):
    """Precision, recall, F1 (empty cases → 1)."""

    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    f1 = 2 * tp / (2*tp+fp+fn) if tp + fp + fn > 0 else 1.0
    return precision, recall, f1



class ObjectMetrics:
    """Accumulates object matches and counts, one image at a time."""

    def __init__(self, iou_thr=0.5):
        self.iou_thr = iou_thr
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.merged = 0
        self.counts = []   # (n_pred, n_true) per image

    def update(self, pred_lab, gt_lab):
        """Label images of ONE image (H, W)."""
        tp, fp, fn, merged = match_objects(pred_lab, gt_lab, self.iou_thr)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.merged += merged
        self.counts.append((int(pred_lab.max()), int(gt_lab.max())))

    def compute(self):
        precision, recall, f1 = prf(self.tp, self.fp, self.fn)
        c = np.array(self.counts, dtype=float)
        err = c[:,0]-c[:,1]                   # n_pred − n_true per image
        has_cells = c[:, 1] > 0
        return {
            "obj_precision": precision,
            "obj_recall": recall,
            "obj_f1": f1,
            "n_merged": self.merged,
            "count_mae": float(np.abs(err).mean()),
            "count_bias": float(err.mean()),
            "count_mape": float((np.abs(err[has_cells]) / c[has_cells, 1]).mean() * 100) if has_cells.any() else 0.0,
        }



if __name__ == "__main__":

    
    def square(r0, c0, size, shape=(20, 20)):
        """Bool mask with a filled square."""
        m = torch.zeros(shape, dtype=torch.bool)
        m[r0:r0 + size, c0:c0 + size] = True
        return m

    gt = square(0, 0, 4)                                   # 16 pixels
    empty = torch.zeros((20, 20), dtype=torch.bool)

    # 1) perfect prediction → 1, 1
    assert dice_iou(*confusion_counts(gt, gt)) == (1.0, 1.0)
    # 2) no overlap → 0, 0
    assert dice_iou(*confusion_counts(square(10, 10, 4), gt)) == (0.0, 0.0)
    # 3) both empty → 1, 1
    assert dice_iou(*confusion_counts(empty, empty)) == (1.0, 1.0)
    # 4) empty prediction, non-empty truth → 0
    assert dice_iou(*confusion_counts(empty, gt))[0] == 0.0
    # 5) half overlap: 4×4 square shifted by 2 columns → TP=8, FP=8, FN=8
    tp, fp, fn = confusion_counts(square(0, 2, 4), gt)
    assert (tp, fp, fn) == (8, 8, 8)
    d, i = dice_iou(tp, fp, fn)
    assert abs(d - 0.5) < 1e-9 and abs(i - 1 / 3) < 1e-9     # Dice 16/32, IoU 8/24

    # 6) global ≠ mean: image A perfect, image B empty with 1 wrong pixel
    m = PixelMetrics()
    m.update(gt, gt)                                      # A: Dice 1
    fp_pixel = empty.clone(); fp_pixel[5, 5] = True
    m.update(fp_pixel, empty)                             # B: Dice 0
    r = m.compute()
    print(r)
    assert r["dice_mean"] == 0.5                          # (1 + 0) / 2
    assert abs(r["dice_global"] - 32 / 33) < 1e-9          # 2·16 / (2·16 + 1 + 0)
    
    ########################################################################################################
    
    def squares(shape, boxes):
        """Label image, one rectangle per object: box = (row, col, height, width)."""
        lab = np.zeros(shape, dtype=np.int32)
        for k, (r0, c0, h, w) in enumerate(boxes, start=1):
            lab[r0:r0 + h, c0:c0 + w] = k
        return lab

    gt_lab = squares((40, 40), [(2, 2, 10, 10), (20, 20, 10, 10)])
    # 7) perfect → 2 pairs
    assert match_objects(gt_lab, gt_lab) == (2, 0, 0, 0)
    # 8) shifted by 2 rows (IoU 80/120 = 0.67 > 0.5) → still found
    assert match_objects(squares((40, 40), [(4, 2, 10, 10), (20, 20, 10, 10)]), gt_lab)[0] == 2
    # 9) shifted by 5 columns (IoU 50/150 = 0.33 < 0.5) → 1 FP + 1 FN
    assert match_objects(squares((40, 40), [(2, 7, 10, 10), (20, 20, 10, 10)]), gt_lab) == (1, 1, 1, 0)
    # 10) MERGE: two close true cells predicted as one block
    gt2 = squares((30, 40), [(5, 5, 10, 10), (5, 16, 10, 10)])
    fused = squares((30, 40), [(5, 5, 10, 21)])
    d_px = dice_iou(*confusion_counts(torch.from_numpy(fused > 0), torch.from_numpy(gt2 > 0)))[0]
    print("fusione → (tp, fp, fn, fuse):", match_objects(fused, gt2), "| Dice di pixel:", round(d_px, 4))
    assert match_objects(fused, gt2) == (0, 1, 2, 1)
    # 11) empty image, no prediction → all 1
    empty_lab = np.zeros((10, 10), dtype=np.int32)
    assert match_objects(empty_lab, empty_lab) == (0, 0, 0, 0) and prf(0, 0, 0) == (1.0, 1.0, 1.0)
    # 12) counting: (pred, true) = (3, 2), (1, 2), (5, 5) → errors +1, −1, 0
    om = ObjectMetrics()
    om.counts = [(3, 2), (1, 2), (5, 5)]
    r = om.compute()
    print("conteggio:", {k: round(v, 3) for k, v in r.items() if k.startswith("count")})
    assert abs(r["count_mae"] - 2 / 3) < 1e-9     # (1 + 1 + 0) / 3
    assert r["count_bias"] == 0                     # +1 and −1 cancel out!
    assert abs(r["count_mape"] - 100 / 3) < 1e-9    # (50% + 50% + 0%) / 3
    print('tutti i test superati')


