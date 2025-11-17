import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, weight_tensor, bias_tensor):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.weight_tensor = weight_tensor
        self.bias_tensor = bias_tensor

        self.patch_embed = nn.Conv2d(in_channels=img_size[0], out_channels=self.bias_tensor.shape[0],
                                     kernel_size=self.patch_size, stride=self.patch_size)

        with torch.no_grad():
            self.patch_embed.weight.data.copy_(self.weight_tensor)
            self.patch_embed.bias.data.copy_(self.bias_tensor)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, pos_encoding_params, cls_token_params):
        super().__init__()
        self.pos_encoding_w = pos_encoding_params
        self.cls_token_w = cls_token_params

        with torch.no_grad():
            self.cls_token = nn.Parameter(self.cls_token_w)
            self.pos_embed = nn.Parameter(self.pos_encoding_w)

    def forward(self, x):
        cls_token_exp = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token_exp, x), dim=1)
        x = x + self.pos_embed
        return x


class MultiHeadSelfAttentionSplit(nn.Module):
    def __init__(self, qk_weights, v_weights, proj_weights):
        super().__init__()
        self.qk_weights = qk_weights
        self.v_weights = v_weights
        self.proj_weights = proj_weights

        self.query_key = nn.Linear(in_features=qk_weights.shape[1], out_features=qk_weights.shape[0])
        self.value = nn.Linear(in_features=v_weights.shape[1], out_features=v_weights.shape[0])
        self.proj = nn.Linear(in_features=proj_weights.shape[1], out_features=proj_weights.shape[0])

        with torch.no_grad():
            self.query_key.weight.data.copy_(self.qk_weights)
            self.value.weight.data.copy_(self.v_weights)
            self.proj.weight.data.copy_(self.proj_weights)

    def forward(self, x):
        query_key = self.query_key(x)
        value = self.value(x)



class MultiHeadSelfAttention(nn.Module):
    pass