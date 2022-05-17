import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from tort_utils import trunc_normal_


class TortHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, 
                 hidden_dim=2048, bottleneck_dim=256, num_classes=None, is_rot_head=None):
        super().__init__()
        
        # contrastive part
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.con_head = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.con_head.weight_g.data.fill_(1)
        if norm_last_layer:
            self.con_head.weight_g.requires_grad = False
        
        # supervised learning part
        self.is_sl_head = num_classes is not None
        if self.is_sl_head:
            self.sl_head = nn.Linear(in_dim, num_classes) if num_classes > 0 else nn.Identity()

        # rotation part
        self.is_rot_head = is_rot_head
        if self.is_rot_head:
            self.rot_head = nn.Linear(in_dim, 4) 


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, use_sl_for, use_rot_for):
        con_emb = self.con_head(nn.functional.normalize(self.mlp(x), dim=-1, p=2))
        sl_emb = self.sl_head(x[:use_sl_for]) if self.is_sl_head and use_sl_for is not None else None
        rot_emb = self.rot_head(x[:use_rot_for]) if self.is_rot_head and use_rot_for is not None else None

        return [con_emb, sl_emb, rot_emb]


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x, use_sl_for, use_rot_for):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor(
            [inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            output = torch.cat((output, _out))
            start_idx = end_idx
        return self.head(output, use_sl_for, use_rot_for)
