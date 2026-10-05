import torch 
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, n_in, n_out,kernel_size=3):
        super().__init__()
        padding=kernel_size//2

        #shortcut branch
        if n_in!=n_out:
            self.shortcut= nn.Conv2d(in_channels=n_in,out_channels=n_out,kernel_size=1,bias=True)
        else:
            self.shortcut= nn.Identity()

        #main branch
        self.main= nn.Sequential(
            nn.BatchNorm2d(num_features=n_in),
            nn.ELU(),
            nn.Conv2d(in_channels=n_in,out_channels=n_out,kernel_size=kernel_size,padding=padding,bias=False),
            nn.BatchNorm2d(n_out),
            nn.ELU(),
            nn.Conv2d(n_out,n_out,kernel_size,padding=padding,bias=True),
        )

    def forward(self,x):
        return self.main(x) + self.shortcut(x)

class Stem(nn.Module):
    def __init__(self, n_in=3, n_out=16):
        super().__init__()

        k = 3
        pad = k // 2

        self.block=nn.Sequential(
            nn.Conv2d(n_in,1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ELU(),
            nn.Conv2d(1,n_out,kernel_size=k,padding=pad),
            nn.BatchNorm2d(n_out),
            nn.ELU(),
            nn.Conv2d(n_out,n_out,kernel_size=k,padding=pad)
        )

    def forward(self,x):
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, n_in, n_out,kernel_size=3):
        super().__init__()

        self.block=nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            ResBlock(n_in,n_out,kernel_size),
        )

    def forward(self,x):
        return self.block(x)


class BottleNeck(nn.Module):

    def __init__(self, n_in,n_out):
        super().__init__()

        self.block=nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            ResBlock(n_in,n_out,kernel_size=3),
            ResBlock(n_out,n_out,kernel_size=5),
        )

    def forward(self,x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self,n_in,skip_size,n_out,kernel_size=3):
        super().__init__()

        self.up=nn.ConvTranspose2d(n_in,n_out,kernel_size=2,stride=2) #raddoppia esattamente H e W
        self.res_block=ResBlock(n_out+skip_size,n_out,kernel_size=kernel_size)

    def forward(self,x,skip_conn):

        #upsampling
        x= self.up(x)

        #concat sul canale
        x=torch.cat([x,skip_conn],dim=1)

        #calcolo il residuo
        x=self.res_block(x)

        #returno il residuo
        return x


class CResUnet(nn.Module):
    def __init__(self, n_in=3,base_ch=16):
        super().__init__()

        #ENCODER
        self.stem= Stem(n_in,base_ch) # 3 -> 16
        self.enc2=EncoderBlock(base_ch,base_ch*2) #16 -> 32
        self.enc3=EncoderBlock(base_ch*2,base_ch*(2**2)) #32 -> 64


        #BOTTLENECK
        self.bottleneck=BottleNeck(base_ch*(2**2),base_ch*(2**3)) #64 ->128

        #DECODER
        self.dec3=DecoderBlock(n_in=base_ch*(2**3),skip_size=base_ch*(2**2),n_out=base_ch*(2**2)) #128 -> 64
        self.dec2=DecoderBlock(n_in=base_ch*(2**2),skip_size=base_ch*2,n_out=base_ch*2)
        self.dec1=DecoderBlock(n_in=base_ch*2,skip_size=base_ch,n_out=base_ch)

        #TESTA
        self.head=nn.Sequential(
            nn.Conv2d(base_ch,1,kernel_size=1),
            nn.Sigmoid(),
        )


    def forward(self,x):

        #ENCODER 
        #salvo gli output per le skip connection

        skip1=self.stem(x)
        skip2=self.enc2(skip1)
        skip3=self.enc3(skip2)


        #BOTTLENECK
        x=self.bottleneck(skip3)


        #DECODER
        #passo x e la skip corrispondete 

        x=self.dec3(x,skip3)
        x=self.dec2(x,skip2)
        x=self.dec1(x,skip1)


        #TESTA
        x=self.head(x)
        return x


