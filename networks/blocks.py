import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from kornia import morphology as morphy

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class EfficentLePE(nn.Module):
    def __init__(self, dim, res, idx, split_size=7, num_heads=8, qk_scale=None) -> None:
        super().__init__()
        self.res = res
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        if idx == -1:
            H_sp, W_sp = self.res, self.res
        elif idx == 0:
            H_sp, W_sp = self.res, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.res, self.split_size
        else:
            print("ERROR MODE : ",idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.get_v = nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1,groups=dim)

    def im2cswin(self,x):
        B,L,C = x.shape
        H = W = int(np.sqrt(L))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x
    
    def forward(self,qkv):
        q,k,v = qkv[0],qkv[1],qkv[2]
        B,L,C = q.shape
        H = W = self.res
        assert(L == H*W)
        k = self.im2cswin(k)
        v = self.im2cswin(v)

        # get lepe start
        q = q.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        q = q.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        q = q.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = self.get_v(q) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        q = q.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        # get lepe end

        att = k.transpose(-2,-1)@v * self.scale
        att = nn.functional.softmax(att,dim=-1,dtype=att.dtype)

        x = q@att +lepe
        x = x.transpose(1,2).reshape(-1,self.H_sp*self.W_sp,C)

        x = windows2img(x,self.H_sp,self.W_sp,H,W).view(B,-1,C)
        return x
    
class CSWinAttention(nn.Module):
    def __init__(self, dim, res, num_heads, split_size,
                 qkv_bias=False, qk_scale=None,
                 norm_layer=nn.LayerNorm,last_stage=False) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_res = res
        self.split_size = split_size
        self.to_qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.norm = norm_layer(dim)

        if self.patch_res == split_size:
            last_stage = True
        
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2

        if last_stage:
            self.attns = nn.ModuleList([
                EfficentLePE(dim, res, -1, split_size, num_heads, qk_scale)
            for i in range(self.branch_num)
            ])
        else:
            self.attns = nn.ModuleList([
                EfficentLePE(dim//2, res, i, split_size, num_heads, qk_scale)
            for i in range(self.branch_num)
            ])

    def forward(self,x):
        """
        Args:
            x: B H*W C
        Returns:
            x: B H*W C
        """
        H = W = self.patch_res
        B,L,C = x.shape
        assert(H*W == L)
        x = self.norm(x)

        qkv = self.to_qkv(x).reshape(B,-1,3,C).permute(2,0,1,3)
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:,:,:,:C//2])
            x2 = self.attns[1](qkv[:,:,:,C//2:])
            att = torch.cat([x1,x2],dim=2)
        else:
            att = self.attns[0](qkv)
        
        return att


class LocalityEnhancer(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(2*dim)
        self.norm2 = nn.LayerNorm(2*dim)
        self.mixer1 = nn.Sequential(
            nn.Conv1d(2*dim,dim,1,1),
            nn.GELU()
        )
        self.mixer2 = nn.Sequential(
            nn.Conv1d(2*dim,dim,1,1),
            nn.GELU()
        )
    
    def forward(self,att1,x1,att2,x2):
        m1 = self.norm1(torch.cat([att1,x2],dim=-1)).transpose(-1,-2)
        m1 = self.mixer1(m1).transpose(-1,-2) + att1
        m2 = self.norm2(torch.cat([att2,x1],dim=-1)).transpose(-1,-2)
        m2 = self.mixer2(m2).transpose(-1,-2) + att2

        return m1,m2

class AttentionEnhancer(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(2*dim)
        self.mixer1 = nn.Sequential(
            nn.Linear(2*dim,dim),
            nn.GELU()
        )
        self.norm2 = nn.LayerNorm(2*dim)
        self.mixer2 = nn.Sequential(
            nn.Linear(2*dim,dim),
            nn.GELU()
        )

    def forward(self,t1,att1,t2,att2):
        c1 = self.norm1(torch.cat([t1,att2],dim=-1))
        c1 = self.mixer1(c1) + t1
        c2 = self.norm2(torch.cat([t2,att1],dim=-1))
        c2 = self.mixer2(c2) + t2

        return c1,c2


class FocalModulation(nn.Module):
    """ Focal Modulation
    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=9, focal_factor=2, use_postln=False):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.f = nn.Linear(dim, 2*dim+(self.focal_level+1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, 
                        padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )

    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        ctx_all = 0
        for l in range(self.focal_level):                     
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)            
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class FocalEnhancedBlock(nn.Module):
    def __init__(self,dim, res, split_size, num_heads) -> None:
        super().__init__()
        dim= dim//2
        self.block_h = CSWinAttention(dim,res,num_heads,split_size)
        self.block_l = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b (h w) c -> b h w c",h=res,w=res),
            FocalModulation(dim=dim),
            Rearrange("b h w c -> b (h w) c")
        )
        self.le = LocalityEnhancer(dim)
        self.ae = AttentionEnhancer(dim)
        self.res = res

    def forward(self,x1, x2):
        """
        Args:
            x_h: B L C (high resolution features)
            x_l: B L C (low resolution features)
        Returns:
            x_h: B L C
            x_l: B L C 
        or 
            x : B L C (if last stage)
        """
        # print(self.res**2, x_h.shape[1])
        assert(self.res*self.res == x1.shape[1]) # H*W == L
        assert(self.res*self.res == x2.shape[1]) # H*W == L
        att1 = self.block_h(x1)
        att2 = self.block_l(x2)
        t1,t2 = self.le(att1,x1,att2,x2)
        x1,x2 = self.ae(t1,att1,t2,att2)
        
        return x1, x2

class URDS(nn.Module):
    def __init__(self,dim,scale=2,device='cuda:0') -> None:
        super().__init__()
        self.S = S = scale
        self.urds = nn.Sequential(
            nn.Conv2d(in_channels=dim,out_channels=dim,
                      kernel_size=2*S-1,stride=S,padding=S-1),
            nn.GELU()
        )
        self.pool = nn.AvgPool2d(S)
        self.construct_element = torch.ones((3,3),device=device)
        self.up = nn.UpsamplingNearest2d(scale_factor=self.S)

    def forward(self,r):
        r_ = self.urds(r) # first downsample
        if r_.device != self.construct_element.device:
            self.construct_element = self.construct_element.to(r_.device)

        details = r - self.up(r_)
        details = morphy.dilation(details,self.construct_element)
        details = self.pool(details)
        r_ = r_ + details

        return r_

class SymmetricPatchMerge(nn.Module):
    def __init__(self, dim_in,dim_out=None,scale_factor=2,device='cuda:0') -> None:
        super().__init__()
        if dim_out is None:
            dim_out = 2*dim_in
        dim = dim_out//2
        self.conv_expand = nn.Conv2d(in_channels=dim_in,out_channels=dim,
                                     kernel_size=3,stride=1,padding=1)

        self.urds0 = URDS(dim,scale=scale_factor,device=device)
        self.urds1 = URDS(dim,scale=scale_factor,device=device)

    def forward(self,x):
        """
        Args:
            x : B C H W
        """
        x = self.conv_expand(x)
        _,C,H,W = x.shape
        x = nn.functional.layer_norm(x,[C,H,W])

        x1 = self.urds0(x)
        x2 = self.urds1(x)

        return (x1 - x2).detach(), x1 + x2


