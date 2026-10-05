import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.feature import peak_local_max
from skimage.segmentation import relabel_sequential, watershed


MIN_AREA=120 # same threshold used to clean the labels (EDA)

def remove_small(mask,min_area=MIN_AREA,connectivity=1):

    # one label k per connected component
    labels=measure.label(mask,connectivity=connectivity)

    # areas[k] = pixels with label k (k = 0 is background)
    areas=np.bincount(labels.ravel())

    # keep[k] = True if object k is large enough
    keep=areas>=min_area 

    # background is never an object
    keep[0]=False

    # lookup → (H, W) mask, True where the pixel's object is kept
    return keep[labels]


def postprocess(prob,threshold=0.5,min_area=MIN_AREA):
    '''Probability map (H, W) → clean binary mask (H, W).'''

    # threshold
    mask= prob>threshold

    # fill holes (background enclosed by a cell)
    mask=ndimage.binary_fill_holes(mask)

    # remove objects smaller than min_area
    mask=remove_small(mask,min_area)

    # clean mask
    return mask


def label_instances(mask,connectivity=1):
    '''bool mask → label image:
    0 = background,
    1..N = cells.'''

    return measure.label(mask,connectivity=connectivity)


def remove_border_objects(mask,connectivity=1):
    '''Step 9c (tested, not adopted): remove objects touching the image border.
    Annotation regularity: only 0.4% of train cells touch it (~4.4% expected).'''
    labels=measure.label(mask,connectivity=connectivity)
    border=np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))
    return mask & ~np.isin(labels,border[border>0])


def split_touching(mask,min_distance=8,min_area=MIN_AREA,connectivity=1):
    '''Step 9c (tested, not adopted): distance-transform watershed to split touching cells.
    bool mask → label image (1..N). Pieces < min_area are removed.'''

    # distance to the edge: one maximum per cell "centre"
    dist=ndimage.distance_transform_edt(mask)
    components=measure.label(mask,connectivity=connectivity)
    peaks=peak_local_max(dist,min_distance=min_distance,labels=components,exclude_border=False)

    markers=np.zeros(mask.shape,dtype=np.int64)
    markers[tuple(peaks.T)]=np.arange(1,len(peaks)+1)
    labels=watershed(-dist,markers,mask=mask)

    # components without peaks stay whole, with a new label
    left=measure.label(mask & (labels==0),connectivity=connectivity)
    labels[left>0]=left[left>0]+labels.max()

    # drop small pieces, then consecutive labels (match_objects uses max())
    areas=np.bincount(labels.ravel())
    keep=areas>=min_area
    keep[0]=False
    labels[~keep[labels]]=0
    return relabel_sequential(labels)[0]


def predict_labels(prob, threshold, border=False, min_distance=None):
    '''Post-processing + optional variants → label image.'''
    mask = postprocess(prob, threshold=threshold)
    if border:
        mask = remove_border_objects(mask)
    # split cells touch: keep the watershed labels
    return label_instances(mask) if min_distance is None else split_touching(mask, min_distance)


def clean_instances(lab,min_area=MIN_AREA):
    '''Instance label image of another model (e.g. Cellpose) → objects smaller than min_area removed, labels 1..N.
    Not label_instances: touching instances must stay separate.'''
    small=np.bincount(lab.ravel())<min_area
    small[0]=False
    return relabel_sequential(np.where(small[lab],0,lab))[0]


if __name__=='__main__':
    prob = np.zeros((60, 60), dtype=np.float32)
    prob[5:20, 5:20] = 0.9        # cell 15×15 = 225 px → kept
    prob[30:35, 30:35] = 0.9      # 25 px → too small, removed
    prob[40:55, 5:20] = 0.9       # cell 15×15 …
    prob[46:49, 11:14] = 0.1      # … with a 3×3 hole → filled
    prob[40:55, 40:55] = 0.4      # below 0.5 → removed; kept at 0.3

    m = postprocess(prob)
    lab = label_instances(m)
    print("oggetti:", lab.max(), "| pixel:", m.sum())
    assert lab.max() == 2                    # small and below-threshold ones removed
    assert m[47, 12]                         # hole filled
    assert m.sum() == 225 + 225              # two filled cells
    assert label_instances(postprocess(prob, threshold=0.3)).max() == 3

    # exact boundary: area 119 removed, 120 kept (same rule as the EDA)
    for area, expected in [(119, 0), (120, 120)]:
        mm = np.zeros((30, 30), dtype=bool)
        r, c = divmod(area, 30)
        mm[:r] = True
        mm[r, :c] = True
        assert remove_small(mm).sum() == expected, f"area {area}"

    # ring: filling BEFORE removing saves the cell
    ring = np.zeros((40, 40), dtype=np.float32)
    ring[5:17, 5:17] = 0.9       # 144 px …
    ring[8:14, 8:14] = 0.1       # … with a 36 px hole → 108 px ring
    print("anello (108 px + buco 36):", postprocess(ring).sum())
    assert postprocess(ring).sum() == 144

    # watershed: two touching discs (r=15) → 2 objects; a single disc stays 1
    yy, xx = np.mgrid[:60, :90]
    two = ((yy - 30) ** 2 + (xx - 30) ** 2 <= 15 ** 2) | ((yy - 30) ** 2 + (xx - 57) ** 2 <= 15 ** 2)
    one = (yy - 30) ** 2 + (xx - 30) ** 2 <= 15 ** 2
    print("watershed, due dischi attaccati:", label_instances(two).max(), "→", split_touching(two).max())
    assert label_instances(two).max() == 1 and split_touching(two).max() == 2
    assert split_touching(one).max() == 1
    assert (split_touching(two) > 0).sum() == two.sum()   # no pixel lost

    # border: only the inner object is kept
    b = np.zeros((30, 30), dtype=bool)
    b[0:5, 10:15] = True       # touches the top border → removed
    b[10:15, 10:15] = True     # inner → kept
    b[20:25, 25:30] = True     # touches the right border → removed
    assert label_instances(remove_border_objects(b)).max() == 1 and remove_border_objects(b)[12, 12]

    print("tutti i test superati")