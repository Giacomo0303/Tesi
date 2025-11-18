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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, qkv_weights, qkv_bias, proj_weights, proj_bias, qk_dim: int, v_dim: int, num_heads: int):
        super().__init__()
        self.qkv_weights = qkv_weights
        self.qkv_bias = qkv_bias
        self.proj_weights = proj_weights
        self.proj_bias = proj_bias
        self.qk_dim = qk_dim  # dimensione della singola head
        self.v_dim = v_dim  # come per qk
        self.num_heads = num_heads
        self.total_qk = self.num_heads * self.qk_dim
        self.total_v = self.num_heads * self.v_dim
        self.scale = self.qk_dim ** (-0.5)

        self.qkv = nn.Linear(in_features=qkv_weights.shape[1], out_features=qkv_weights.shape[0])
        self.proj = nn.Linear(in_features=proj_weights.shape[1], out_features=proj_weights.shape[0])

        with torch.no_grad():
            self.qkv.weight.data.copy_(self.qkv_weights)
            self.qkv.bias.data.copy_(self.qkv_bias)
            self.proj.weight.data.copy_(self.proj_weights)
            self.proj.bias.data.copy_(self.proj_bias)

    def forward(self, x):
        qkv = self.qkv(x)  # qkv ha shape [B, N, dimQ + dimK + dimV]
        # devo splittare per ottenere Q, K e V proprio sull'ultima dimensione
        Q, K, V = torch.split(qkv, [self.total_qk, self.total_qk, self.total_v], dim=-1)
        # dimQ, dimK e dimV contengono l'output delle varie head concatenate, quindi devo fare il reshape
        # da [B, N, H*dim] -> [B, N, H, dim], ma non va ancora bene perche
        # nel fare il mat_mul ho bisgno di [B, H, N, dim] perche devo moltiplicare tutto il contenuto della testa di Q con quella di K
        Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        K = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        V = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        # [B, H, N, dim] x [B, H, dim, N] -> [B, H, N, N]
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)

        # [B, H, N, N] x [B, H, N, v_dim] -> [B, H, N, v_dim]
        x = attn @ V

        # [B, H, N, v_dim] -> [B, N, H*v_dim]
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.num_heads * self.v_dim)
        x = self.proj(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, mhsa, fc1_weights, fc1_bias, fc2_weights, fc2_bias, norm1_weights, norm1_bias, norm2_weights,
                 norm2_bias):
        super().__init__()
        self.new_emb_dim = fc1_weights.shape[1]
        self.mhsa = mhsa

        self.layerNorm1 = nn.LayerNorm(self.new_emb_dim)
        self.layerNorm2 = nn.LayerNorm(self.new_emb_dim)

        self.fc1 = nn.Linear(fc1_weights.shape[1], fc1_weights.shape[0])
        self.fc2 = nn.Linear(fc2_weights.shape[1], fc2_weights.shape[0])

        self.mlp = nn.Sequential(
            self.fc1,
            nn.GELU(),
            self.fc2
        )

        with torch.no_grad():
            self.fc1.weight.data.copy_(fc1_weights)
            self.fc1.bias.data.copy_(fc1_bias)
            self.fc2.weight.data.copy_(fc2_weights)
            self.fc2.bias.data.copy_(fc2_bias)
            self.layerNorm1.weight.data.copy_(norm1_weights)
            self.layerNorm1.bias.data.copy_(norm1_bias)
            self.layerNorm2.weight.data.copy_(norm2_weights)
            self.layerNorm2.bias.data.copy_(norm2_bias)

    def forward(self, x):
        x = x + self.mhsa(self.layerNorm1(x))
        x = x + self.mlp(self.layerNorm2(x))
        return x


def get_qkv_weights_bias(original_qkv_weights, original_qkv_bias, block_config, original_num_heads, original_head_dim,
                         new_emb_dims):
    head_pruned_idx = block_config["head_pruned_idx"]
    qk_pruned_dims = block_config["qk_pruned_dims"]
    v_proj_pruned_dims = block_config["v_proj_pruned_dims"]

    final_heads = [h for h in range(original_num_heads) if h not in head_pruned_idx]
    final_qk_dims = [d for d in range(original_head_dim) if d not in qk_pruned_dims]
    final_v_dims = [d for d in range(original_head_dim) if d not in v_proj_pruned_dims]

    original_qkv_weights = original_qkv_weights.reshape(3, original_num_heads, original_head_dim, -1)
    original_qkv_bias = original_qkv_bias.reshape(3, original_num_heads, original_head_dim)

    # seleziona Q, k, V, poi la head, poi di tutte le head rimaste prende solo le dimensioni selezionate, infine elimina anche l'embedding prunato
    q_w = original_qkv_weights[0][final_heads][:, final_qk_dims, :][:, :, new_emb_dims].reshape(-1,
                                                                                                len(new_emb_dims))  # l'ultimo reshape è per ottenere [n_heads*final_dim, emb]
    k_w = original_qkv_weights[1][final_heads][:, final_qk_dims, :][:, :, new_emb_dims].reshape(-1, len(new_emb_dims))
    v_w = original_qkv_weights[2][final_heads][:, final_v_dims, :][:, :, new_emb_dims].reshape(-1, len(new_emb_dims))

    q_b = original_qkv_bias[0][final_heads][:, final_qk_dims].flatten()
    k_b = original_qkv_bias[1][final_heads][:, final_qk_dims].flatten()
    v_b = original_qkv_bias[2][final_heads][:, final_v_dims].flatten()

    final_qkv_weights = torch.cat([q_w, k_w, v_w], dim=0)
    final_qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)

    return final_qkv_weights, final_qkv_bias, len(final_qk_dims), len(final_v_dims), len(final_heads)


def get_proj_weights_bias(original_proj_weights, original_proj_bias, block_config, original_num_heads,
                          original_head_dim,
                          new_emb_dims):
    head_pruned_idx = block_config["head_pruned_idx"]
    v_proj_pruned_dims = block_config["v_proj_pruned_dims"]

    final_heads = [h for h in range(original_num_heads) if h not in head_pruned_idx]
    final_v_dims = [d for d in range(original_head_dim) if d not in v_proj_pruned_dims]

    # da [emb, n_heads * head_dim] -> [emb, n_heads, head_dim]
    original_proj_weights = original_proj_weights.reshape(-1, original_num_heads, original_head_dim)

    final_proj_weights = original_proj_weights[new_emb_dims][:, final_heads, :][:, :, final_v_dims].reshape(
        len(new_emb_dims), -1)
    final_proj_bias = original_proj_bias[new_emb_dims]

    return final_proj_weights, final_proj_bias


class CompressedViT(nn.Module):
    def __init__(self, search_dict, pruned_model, img_size=224, patch_size=16):
        super().__init__()

        original_emb_dims = list(range(pruned_model.patch_embed.proj.weight.shape[0]))
        pruned_emb_dims = search_dict["embed_pruned_dims"]
        new_emb_dims = [dim for dim in original_emb_dims if dim not in pruned_emb_dims]
        patch_embed_weight = pruned_model.patch_embed.proj.weight[new_emb_dims, :, :, :]
        patch_embed_bias = pruned_model.patch_embed.proj.bias[new_emb_dims]
        self.patch_embed = PatchEmbed((img_size, img_size), (patch_size, patch_size), patch_embed_weight,
                                      patch_embed_bias)

        self.pos_enc = PositionalEncoding(pos_encoding_params=pruned_model.pos_embed.data[:, :, new_emb_dims],
                                          cls_token_params=pruned_model.cls_token.data[:, :, new_emb_dims])

        self.blocks_list = []
        blocks_dict = search_dict["blocks"]
        for i, block in enumerate(blocks_dict):
            original_num_heads = pruned_model.blocks[0].attn.num_heads
            original_head_dim = pruned_model.blocks[0].attn.head_dim

            qkv_weights, qkv_bias, qk_dim, v_dim, num_heads = get_qkv_weights_bias(
                pruned_model.blocks[i].attn.qkv.weight,
                pruned_model.blocks[i].attn.qkv.bias, block,
                original_num_heads, original_head_dim, new_emb_dims)

            proj_weights, proj_bias = get_proj_weights_bias(pruned_model.blocks[i].attn.proj.weight,
                                                            pruned_model.blocks[i].attn.proj.bias,
                                                            block, original_num_heads, original_head_dim, new_emb_dims)

            mhsa = MultiHeadSelfAttention(qkv_weights, qkv_bias, proj_weights, proj_bias, qk_dim, v_dim, num_heads)


def forward(self, x):
    x = self.patch_embed(x)
    x = self.pos_enc(x)
    return x
