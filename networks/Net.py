import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from .blocks import FocalEnhancedBlock as Block,SymmetricPatchMerge
from .decoders import Decoder
from thop import profile, clever_format


class EncoderStage(nn.Module):
    def __init__(self, dim_in, dim_out, res,scale_factor, split_size,num_heads,depth=1,device='cuda:0') -> None:
        super().__init__()
        self.spm = SymmetricPatchMerge(dim_in=dim_in,dim_out=dim_out,
                                       scale_factor=scale_factor,device=device)
        res = res//scale_factor
        self.blocks = nn.ModuleList(
            [Block(dim_out,res,split_size,num_heads) for _ in range(depth)]
        )
        self.img2token = Rearrange("b c h w -> b (h w) c")
        self.token2img = Rearrange("b (h w) c -> b c h w", h=res,w=res)

    def forward(self,x):
        """
        Args:
            x: B C H W
        Returns:
            att: B C H W
        """
        phi0, phi1 = self.spm(x)

        att0,att1 = self.img2token(phi0),self.img2token(phi1)
        for block in self.blocks:
            att0, att1 = block(att0, att1)
        
        phi0 = self.token2img(att0) + phi0
        phi1 = self.token2img(att1) + phi1

        return torch.cat([phi0,phi1], dim=1)


class Encoder(nn.Module):
    def __init__(self, img_size=224, dim_in=1,device='cuda:0',
                 embed_dim=[96,192,384,768],scale_factor=2,
                 split_size=[1,2,2,7], num_heads=[2,4,8,16],depth=[1,2,4,2]) -> None:
        super().__init__()

        self.stages = []
        assert(len(embed_dim) == len(split_size) == len(num_heads) == len(depth))

        for i in range(len(embed_dim)):
            if i == 0:
                stage = EncoderStage(dim_in=dim_in,dim_out=embed_dim[i], res=img_size,scale_factor=scale_factor,
                                     split_size=split_size[i],num_heads=num_heads[i],depth=depth[i],device=device)
                self.stages.append(stage)
                res = img_size//scale_factor
            else:
                stage = EncoderStage(dim_in=embed_dim[i-1], dim_out=embed_dim[i], res=res,scale_factor=2,
                                     split_size=split_size[i],num_heads=num_heads[i],depth=depth[i],device=device)
                res = res//2
                self.stages.append(stage)

        self.stages = nn.ModuleList(self.stages)
    
    def forward(self,x):
        """
        Args:
            img: B C H W
        Returns:
            skips [(att,x_l)...]:
                att,x_l: B C H W  
        """
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)

        return tuple(skips)

class Net(nn.Module):
    def __init__(self, img_size=224, dim_in=1, dim_out=9,
                 embed_dim=128,scale_factor=2,device='cuda:0',
                 split_size=[1,2,2,7], num_heads=[2,4,8,16],depth=[1,2,4,2]) -> None:
        super().__init__()
        embed_dim = [embed_dim*(2**i) for i in range(len(depth))]
        self.encoder = Encoder(img_size=img_size,dim_in=dim_in,device=device,
                               embed_dim=embed_dim,scale_factor=scale_factor,
                               split_size=split_size,num_heads=num_heads,depth=depth)
        self.decoder = Decoder(channels=embed_dim[::-1], num_classes=dim_out,scale_factor=scale_factor)
        
    def forward(self,img):
        return self.decoder(self.encoder(img))
    

