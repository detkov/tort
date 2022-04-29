import math
from munch import Munch

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# from drloc import DenseRelativeLoc

import torch
import torch.nn as nn
import torch.nn.functional as F

# from munch import Munch
torch.manual_seed(12345)

def randn_sampling(maxint, sample_size, batch_size):
    return torch.randint(maxint, size=(batch_size, sample_size, 2))

def collect_samples(feats, pxy, batch_size):
    return torch.stack([feats[i, :, pxy[i][:,0], pxy[i][:,1]] for i in range(batch_size)], dim=0)

def collect_samples_faster(feats, pxy, batch_size):
    n,c,h,w = feats.size()
    feats = feats.view(n, c, -1).permute(1,0,2).reshape(c, -1)  # [n, c, h, w] -> [n, c, hw] -> [c, nhw]
    pxy = ((torch.arange(n).long().to(pxy.device) * h * w).view(n, 1) + pxy[:,:,0]*h + pxy[:,:,1]).view(-1)  # [n, m, 2] -> [nm]
    return (feats[:,pxy]).view(c, n, -1).permute(1,0,2)

def collect_positions(batch_size, N):
    all_positions = [[i,j]  for i in range(N) for j in range(N)]
    pts = torch.tensor(all_positions) # [N*N, 2]
    pts_norm = pts.repeat(batch_size,1,1)  # [B, N*N, 2]
    rnd = torch.stack([torch.randperm(N*N) for _ in range(batch_size)], dim=0) # [B, N*N]
    pts_rnd = torch.stack([pts_norm[idx,r] for idx, r in enumerate(rnd)],dim=0) # [B, N*N, 2]
    return pts_norm, pts_rnd

class DenseRelativeLoc(nn.Module):
    def __init__(self, in_dim, out_dim=2, sample_size=32, drloc_mode="l1", use_abs=False):
        super(DenseRelativeLoc, self).__init__()
        self.sample_size = sample_size
        self.in_dim  = in_dim
        self.drloc_mode = drloc_mode
        self.use_abs = use_abs

        if self.drloc_mode == "l1":
            self.out_dim = out_dim
            self.layers = nn.Sequential(
                nn.Linear(in_dim*2, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.out_dim)
            )
        elif self.drloc_mode in ["ce", "cbr"]:
            self.out_dim = out_dim if self.use_abs else out_dim*2 - 1
            self.layers  = nn.Sequential(
                nn.Linear(in_dim*2, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )
            self.unshared = nn.ModuleList()
            for _ in range(2):
                self.unshared.append(nn.Linear(512, self.out_dim))
        else:
            raise NotImplementedError("We only support l1, ce and cbr now.")

    def forward_features(self, x, mode="part"):
        # x, feature map with shape: [B, C, H, W]
        B, C, H, W = x.size()

        if mode == "part":
            pxs = randn_sampling(H, self.sample_size, B).detach()
            pys = randn_sampling(H, self.sample_size, B).detach()
            
            deltaxy = (pxs-pys).float().to(x.device) # [B, sample_size, 2]

            ptsx = collect_samples_faster(x, pxs, B).transpose(1,2).contiguous() # [B, sample_size, C]
            ptsy = collect_samples_faster(x, pys, B).transpose(1,2).contiguous() # [B, sample_size, C]
        else:
            pts_norm, pts_rnd = collect_positions(B, H)
            ptsx = x.view(B,C,-1).transpose(1,2).contiguous() # [B, H*W, C]
            ptsy = collect_samples(x, pts_rnd, B).transpose(1,2).contiguous() # [B, H*W, C]

            deltaxy = (pts_norm - pts_rnd).float().to(x.device) # [B, H*W, 2]

        pred_feats = self.layers(torch.cat([ptsx, ptsy], dim=2))
        return pred_feats, deltaxy, H

    def forward(self, x, normalize=False):
        pred_feats, deltaxy, H = self.forward_features(x)
        deltaxy = deltaxy.view(-1, 2) # [B*sample_size, 2]

        if self.use_abs:
            deltaxy = torch.abs(deltaxy)
            if normalize:
                deltaxy /= float(H-1)
        else:
            deltaxy += (H-1)
            if normalize:
                deltaxy /= float(2*(H - 1))
        
        if self.drloc_mode == "l1":
            predxy = pred_feats.view(-1, self.out_dim) # [B*sample_size, Output_size]
        else: 
            predx, predy = self.unshared[0](pred_feats), self.unshared[1](pred_feats)
            predx = predx.view(-1, self.out_dim) # [B*sample_size, Output_size]
            predy = predy.view(-1, self.out_dim) # [B*sample_size, Output_size]
            predxy = torch.stack([predx, predy], dim=2) # [B*sample_size, Output_size, 2]   
        return predxy, deltaxy

    def flops(self):
        fps =  self.in_dim * 2 * 512 * self.sample_size
        fps += 512 * 512 * self.sample_size
        fps += 512 * self.out_dim * self.sample_size
        if self.drloc_mode in ["ce", "cbr"]:
            fps += 512 * 512 * self.sample_size
            fps += 512 * self.out_dim * self.sample_size
        return fps

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, 
        image_size=224, 
        patch_size=16, 
        num_classes=1000, 
        dim=768, 
        depth=12, 
        heads=12, 
        mlp_dim=3072, 
        pool = 'cls', 
        channels = 3, 
        dim_head = 64, 
        dropout = 0.1, 
        emb_dropout = 0.,
        use_drloc=True,     # relative distance prediction
        drloc_mode="l1",
        sample_size=32,
        use_abs=True,
    ):
        super().__init__()
        self.use_drloc = use_drloc

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        if self.use_drloc:
            self.drloc = DenseRelativeLoc(
                in_dim=dim, 
                out_dim=2 if drloc_mode=="l1" else 14,
                sample_size=sample_size,
                drloc_mode=drloc_mode,
                use_abs=use_abs
            )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        outs = Munch()

        # SSUP
        if self.use_drloc:
            x_last = x[:,1:] # B, L, C 
            x_last = x_last.transpose(1, 2) # [B, C, L]
            B, C, HW = x_last.size()
            H = W = int(math.sqrt(HW))
            x_last = x_last.view(B, C, H, W) # [B, C, H, W]

            drloc_feats, deltaxy = self.drloc(x_last)
            outs.drloc = [drloc_feats]
            outs.deltaxy = [deltaxy]
            outs.plz = [H] # plane size 

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        sup = self.mlp_head(x)
        outs.sup = sup
        return outs
