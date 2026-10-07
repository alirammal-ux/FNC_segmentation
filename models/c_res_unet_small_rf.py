import torch
import torch.nn as nn

from models.c_res_unet_logits import CResUnetLogits

class CResUnetSmallRF(CResUnetLogits):
    '''Same as CResUnetLogits without the second bottleneck block (the 5×5 residual block):
    receptive field 96 px instead of 160 px, 0.50 M parameters instead of 1.32 M.
    The adopted (light) network; the original model (c_res_unet.py) is not modified.'''

    def __init__(self, n_in=3, base_ch=16):
        super().__init__(n_in, base_ch)
        # bottleneck.block = [MaxPool, ResBlock 3×3, ResBlock 5×5]: the last one becomes a no-op (same channels in and out)
        self.bottleneck.block[2] = nn.Identity()

if __name__=='__main__':
    model=CResUnetSmallRF()
    n=sum(p.numel() for p in model.parameters())
    assert n==504631, n
    print('parameters:', n)

    model.eval()
    x=torch.randn(1,3,1200,1600)
    with torch.no_grad():
        y=model(x)
    assert y.shape==(1,1,1200,1600)
    print('forward ok:', tuple(y.shape))

    # receptive field: input pixels whose gradient is not zero for the central output pixel
    x=torch.randn(1,3,512,512,requires_grad=True)
    model(x)[0,0,256,256].backward()
    rows=torch.nonzero(x.grad.abs().sum(1)[0])[:,0]
    print('receptive field:', int(rows.max()-rows.min()+1), 'px')
