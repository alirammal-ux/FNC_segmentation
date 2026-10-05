########################################################
# BCE

# Per pixel: how far the predicted probability is from the label:
#    −log(p) for cells,
#    −log(1−p) for background.

# Then averaged over all pixels.


########################################################
# Soft Dice

# Dice without threshold:
#   - uses probabilities p = sigmoid(logit) instead of 0/1 pixels
# 
# so it is differentiable:
#   - soft Dice = (2·Σ p·g + s) / (Σ p + Σ g + s), g = true mask, s = 1 avoids 0/0.
# 
# The loss is 1 − soft Dice.


#########################################################

# BCE + (1 − soft Dice), 1:1: BCE gives stable per-pixel gradients, Dice counters the imbalance.



import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_dice(logits,target,smooth=1.0):
    """Soft Dice over the whole batch."""

    probs=torch.sigmoid(logits)
    inter= (probs*target).sum()
    return (2*inter + smooth) / (probs.sum()+target.sum()+smooth)

class BCEDiceLoss(nn.Module):
    '''BCE + (1 − soft Dice), weighted 1:1.
    forward returns (total loss, bce, dice).
    '''

    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth=smooth


    def forward(self,logits,target,weight=None):
        '''logits and target must have the same shape.
        weight (optional, same shape): per-pixel weights of the BCE (weight maps); the Dice is not weighted.'''
        assert logits.shape == target.shape, f"incorrect shape: {logits.shape} vs {target.shape}"

        logits=logits.float()
        target=target.float()

        if weight is None:
            bce=F.binary_cross_entropy_with_logits(logits,target)
        else: # weighted mean, normalized by the weights: same scale as the plain mean
            weight=weight.float()
            per_px=F.binary_cross_entropy_with_logits(logits,target,reduction="none")

            # weighted mean, normalized by the weights:
            bce=(weight*per_px).sum()/weight.sum()

            
        dice=soft_dice(logits,target,self.smooth)
        loss=(bce+1-dice)
        return loss,bce.detach(),dice.detach()



if __name__ == "__main__":
    crit = BCEDiceLoss()

    # two 64×64 images, one cell rectangle each
    target = torch.zeros(2, 1, 64, 64)
    target[0, 0, 10:20, 10:20] = 1
    target[1, 0, 30:36, 40:52] = 1

    # 1) perfect, confident prediction → loss ≈ 0
    perfect = (target * 2 - 1) * 20        # logit +20 on cells, −20 on background
    loss, bce, dice = crit(perfect, target)
    print(f"perfect:   loss={loss:.4f}  bce={bce:.4f}  dice={dice:.4f}")
    assert loss < 1e-3

    # 2) completely wrong and confident → high loss
    loss, bce, dice = crit(-perfect, target)
    print(f"wrong:     loss={loss:.4f}  bce={bce:.4f}  dice={dice:.4f}")
    assert loss > 10

    # 3) "lazy" net: 0.6% cells, confidently predicts all background
    lazy_target = torch.zeros(1, 1, 100, 100)
    lazy_target[0, 0, :6, :10] = 1                    # 60 pixels out of 10 000
    lazy = torch.full_like(lazy_target, -5.0)         # p = sigmoid(−5) ≈ 0.0067 everywhere
    loss, bce, dice = crit(lazy, lazy_target)
    print(f"lazy:      loss={loss:.4f}  bce={bce:.4f}  dice={dice:.4f}")
    assert bce < 0.05 and dice < 0.1                  # BCE is almost happy, Dice exposes it

    # 4) the gradient pushes cell logits UP and background logits DOWN
    logits = torch.zeros(2, 1, 64, 64, requires_grad=True)   # p = 0.5 everywhere: the net knows nothing
    loss, _, _ = crit(logits, target)
    loss.backward()
    g = logits.grad
    print("mean gradient  cell:", g[target == 1].mean().item(), " background:", g[target == 0].mean().item())
    assert torch.isfinite(g).all()
    assert (g[target == 1] < 0).all() and (g[target == 0] > 0).all()

    # 5) AMP: float16 logits on a realistic batch (8 crops 512×512, ~1% cells).
    #    The Dice must EQUAL the float32 one.
    t = torch.zeros(8, 1, 512, 512)
    t[:, :, :52, :52] = 1
    x = torch.zeros(8, 1, 512, 512)                   # p = 0.5 everywhere
    _, _, dice32 = crit(x, t)
    _, _, dice16 = crit(x.half(), t.half())
    print(f"dice float32={dice32:.6f}  float16={dice16:.6f}")
    assert abs(dice16 - dice32) < 1e-4, "in float16 the sums overflow: is the .float() missing?"

    print("all tests passed")
