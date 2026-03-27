import torch
from torch import nn

from src.utils.CompressedViT import CompressedViT


class CompressedDeiT(CompressedViT):
    def __init__(self, search_dict, pruned_model, original_head_dim, img_size=224, patch_size=16):
        super().__init__(search_dict=search_dict, pruned_model=pruned_model, original_head_dim=original_head_dim,
                         img_size=img_size, patch_size=patch_size)
        original_emb_dims = list(range(pruned_model.patch_embed.proj.weight.shape[0]))
        pruned_emb_dims = search_dict["embed_pruned_dims"]
        new_emb_dims = [dim for dim in original_emb_dims if dim not in pruned_emb_dims]

        dist_token_data = pruned_model.dist_token.data[:, :, new_emb_dims]

        with torch.no_grad():
            self.dist_token = nn.Parameter(dist_token_data)

        self.head_dist = nn.Linear(len(new_emb_dims), pruned_model.head.out_features)

        head_dist_weights = pruned_model.head_dist.weight[:, new_emb_dims]
        head_dist_bias = pruned_model.head_dist.bias

        with torch.no_grad():
            self.head_dist.weight.copy_(head_dist_weights)
            self.head_dist.bias.copy_(head_dist_bias)

    def forward(self, x):
        x = self.forward_features(x)
        x_cls = self.head(x[:, 0])
        x_dist = self.head_dist(x[:, 1])
        if self.training:
            return x_cls, x_dist
        else:
            return (x_cls + x_dist) / 2

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token_exp = self.cls_token.expand(x.shape[0], -1, -1)
        dist_token_exp = self.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token_exp, dist_token_exp, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x