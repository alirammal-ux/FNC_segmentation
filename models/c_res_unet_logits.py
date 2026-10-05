import torch
import torch.nn as nn

from models.c_res_unet import CResUnet

class CResUnetLogits(CResUnet):
    '''Same as CResUnet, but the head returns logits (no sigmoid),
      needed for BCEWithLogitsLoss.
      At inference use torch.sigmoid(model(x)).
      '''
    
    def __init__(self, n_in=3, base_ch=16):
        super().__init__(n_in, base_ch)
        # forward() calls self.head: overriding it is enough
        self.head= nn.Sequential(
            nn.Conv2d(base_ch,1,kernel_size=1) # no sigmoid
        )

if __name__=='__main__':
    Unet_madre=CResUnet()
    model=CResUnetLogits()

    # same number of parameters
    p_madre=sum(p.numel() for p in Unet_madre.parameters()) # numel() = elements of each weight tensor
    p_model=sum(p.numel() for p in model.parameters())
    assert(p_madre==p_model)
    print('stesso numero di params')


    # same weight names and shapes
    model.load_state_dict(Unet_madre.state_dict())
    print('state dict compatibile')

    # same forward: sigmoid(logits) == original output
    Unet_madre.eval()
    model.eval()
    x=torch.randn(1,3,1200,1600)
    with torch.no_grad():
        assert torch.allclose(nn.functional.sigmoid(model(x)), Unet_madre(x), atol=1e-6) # allclose, not ==: floats
        print('forward coerente')




    