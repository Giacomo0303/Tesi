import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, weight_tensor, bias_tensor):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        in_channels = weight_tensor.shape[1]
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=bias_tensor.shape[0],
                              kernel_size=self.patch_size, stride=self.patch_size)

        with torch.no_grad():
            self.proj.weight.data.copy_(weight_tensor)
            self.proj.bias.data.copy_(bias_tensor)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, qkv_weights, qkv_bias, proj_weights, proj_bias, original_qk_dim: int, qk_dim: int, v_dim: int,
                 num_heads: int):
        super().__init__()
        self.head_dim = qk_dim  # dimensione della singola head
        self.qk_dim = qk_dim
        self.v_dim = v_dim  # come per qk
        self.num_heads = num_heads
        self.total_qk = self.num_heads * self.qk_dim
        self.total_v = self.num_heads * self.v_dim
        self.scale = original_qk_dim ** (-0.5)

        self.qkv = nn.Linear(in_features=qkv_weights.shape[1], out_features=qkv_weights.shape[0])
        self.proj = nn.Linear(in_features=proj_weights.shape[1], out_features=proj_weights.shape[0])

        with torch.no_grad():
            self.qkv.weight.data.copy_(qkv_weights)
            self.qkv.bias.data.copy_(qkv_bias)
            self.proj.weight.data.copy_(proj_weights)
            self.proj.bias.data.copy_(proj_bias)

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
        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, mhsa, fc1_weights, fc1_bias, fc2_weights, fc2_bias, norm1_weights, norm1_bias, norm2_weights,
                 norm2_bias):
        super().__init__()
        self.new_emb_dim = fc1_weights.shape[1]
        self.attn = mhsa

        self.norm1 = nn.LayerNorm(self.new_emb_dim)
        self.norm2 = nn.LayerNorm(self.new_emb_dim)

        in_features = fc1_weights.shape[1]
        hidden_features = fc1_weights.shape[0]
        out_features = fc2_weights.shape[0]

        self.mlp = Mlp(in_features, hidden_features, out_features)

        with torch.no_grad():
            self.mlp.fc1.weight.data.copy_(fc1_weights)
            self.mlp.fc1.bias.data.copy_(fc1_bias)
            self.mlp.fc2.weight.data.copy_(fc2_weights)
            self.mlp.fc2.bias.data.copy_(fc2_bias)
            self.norm1.weight.data.copy_(norm1_weights)
            self.norm1.bias.data.copy_(norm1_bias)
            self.norm2.weight.data.copy_(norm2_weights)
            self.norm2.bias.data.copy_(norm2_bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def get_qkv_weights_bias(original_qkv_weights, original_qkv_bias, block_config, original_num_heads, original_qk_dim,
                         original_v_dim, new_emb_dims):
    head_pruned_idx = block_config["head_pruned_idx"]
    qk_pruned_dims = block_config["qk_pruned_dims"]
    v_proj_pruned_dims = block_config["v_proj_pruned_dims"]

    final_heads = [h for h in range(original_num_heads) if h not in head_pruned_idx]
    final_qk_dims = [d for d in range(original_qk_dim) if d not in qk_pruned_dims]
    final_v_dims = [d for d in range(original_v_dim) if d not in v_proj_pruned_dims]

    # calcolo delle dimensioni attuali
    total_qk_size = original_qk_dim * original_num_heads
    total_v_size = original_v_dim * original_num_heads

    # splitting delle Q, K, V con tutte le head
    q_w, k_w, v_w = torch.split(original_qkv_weights, [total_qk_size, total_qk_size, total_v_size], dim=0)
    q_b, k_b, v_b = torch.split(original_qkv_bias, [total_qk_size, total_qk_size, total_v_size], dim=0)

    # selezione dei pesi originali delle matrici Q e K
    q_w = q_w.reshape(original_num_heads, original_qk_dim, -1)
    k_w = k_w.reshape(original_num_heads, original_qk_dim, -1)
    q_b = q_b.reshape(original_num_heads, original_qk_dim)
    k_b = k_b.reshape(original_num_heads, original_qk_dim)

    # V usa original_v_dim
    v_w = v_w.reshape(original_num_heads, original_v_dim, -1)
    v_b = v_b.reshape(original_num_heads, original_v_dim)

    # seleziona Q, k, V, poi la head, poi di tutte le head rimaste prende solo le dimensioni selezionate, infine elimina anche l'embedding prunato
    q_w = q_w[final_heads][:, final_qk_dims, :][:, :, new_emb_dims].reshape(-1,
                                                                            len(new_emb_dims))  # l'ultimo reshape è per ottenere [n_heads*final_dim, emb]
    k_w = k_w[final_heads][:, final_qk_dims, :][:, :, new_emb_dims].reshape(-1, len(new_emb_dims))
    q_b = q_b[final_heads][:, final_qk_dims].flatten()
    k_b = k_b[final_heads][:, final_qk_dims].flatten()

    v_w = v_w[final_heads][:, final_v_dims, :][:, :, new_emb_dims].reshape(-1, len(new_emb_dims))
    v_b = v_b[final_heads][:, final_v_dims].flatten()

    final_qkv_weights = torch.cat([q_w, k_w, v_w], dim=0)
    final_qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)

    return final_qkv_weights, final_qkv_bias, len(final_qk_dims), len(final_v_dims), len(final_heads)


def get_proj_weights_bias(original_proj_weights, original_proj_bias, block_config, original_num_heads,
                          original_v_dim,
                          new_emb_dims):
    head_pruned_idx = block_config["head_pruned_idx"]
    v_proj_pruned_dims = block_config["v_proj_pruned_dims"]

    final_heads = [h for h in range(original_num_heads) if h not in head_pruned_idx]
    final_v_dims = [d for d in range(original_v_dim) if d not in v_proj_pruned_dims]

    # da [emb, n_heads * head_dim] -> [emb, n_heads, head_dim]
    original_proj_weights = original_proj_weights.reshape(-1, original_num_heads, original_v_dim)

    final_proj_weights = original_proj_weights[new_emb_dims][:, final_heads, :][:, :, final_v_dims].reshape(
        len(new_emb_dims), -1)
    final_proj_bias = original_proj_bias[new_emb_dims]

    return final_proj_weights, final_proj_bias


def get_mlp_weights_bias(original_mlp, block_config, new_emb_dims):
    fc1_weights = original_mlp.fc1.weight
    fc1_bias = original_mlp.fc1.bias
    fc2_weights = original_mlp.fc2.weight
    fc2_bias = original_mlp.fc2.bias

    mlp_pruned_dims = block_config["mlp_pruned_dims"]
    original_mlp_dims = fc1_weights.shape[0]
    final_fc1_dims = [d for d in range(original_mlp_dims) if d not in mlp_pruned_dims]

    final_fc1_weights = fc1_weights[final_fc1_dims, :][:, new_emb_dims]
    final_fc1_bias = fc1_bias[final_fc1_dims]

    final_fc2_weights = fc2_weights[new_emb_dims, :][:, final_fc1_dims]
    final_fc2_bias = fc2_bias[new_emb_dims]

    return final_fc1_weights, final_fc1_bias, final_fc2_weights, final_fc2_bias


class CompressedViT(nn.Module):
    def __init__(self, search_dict, pruned_model, original_head_dim, img_size=224, patch_size=16):
        super().__init__()

        original_emb_dims = list(range(pruned_model.patch_embed.proj.weight.shape[0]))
        pruned_emb_dims = search_dict["embed_pruned_dims"]
        new_emb_dims = [dim for dim in original_emb_dims if dim not in pruned_emb_dims]
        patch_embed_weight = pruned_model.patch_embed.proj.weight[new_emb_dims, :, :, :]
        patch_embed_bias = pruned_model.patch_embed.proj.bias[new_emb_dims]
        self.patch_embed = PatchEmbed((img_size, img_size), (patch_size, patch_size), patch_embed_weight,
                                      patch_embed_bias)

        cls_token_data = pruned_model.cls_token.data[:, :, new_emb_dims]
        pos_embed_data = pruned_model.pos_embed.data[:, :, new_emb_dims]

        with torch.no_grad():
            self.cls_token = nn.Parameter(cls_token_data)
            self.pos_embed = nn.Parameter(pos_embed_data)

        self.blocks_list = []
        blocks_dict = search_dict["blocks"]

        for i, block in enumerate(blocks_dict):
            current_block_attn = pruned_model.blocks[i].attn
            original_num_heads = current_block_attn.num_heads

            if hasattr(current_block_attn, 'qk_dim'):
                original_qk_dim = current_block_attn.qk_dim
                original_v_dim = current_block_attn.v_dim
            else:
                original_qk_dim = current_block_attn.head_dim
                original_v_dim = current_block_attn.head_dim

            original_eps = pruned_model.blocks[i].norm1.eps

            qkv_weights, qkv_bias, qk_dim, v_dim, num_heads = get_qkv_weights_bias(
                pruned_model.blocks[i].attn.qkv.weight,
                pruned_model.blocks[i].attn.qkv.bias, block,
                original_num_heads, original_qk_dim, original_v_dim, new_emb_dims)

            proj_weights, proj_bias = get_proj_weights_bias(pruned_model.blocks[i].attn.proj.weight,
                                                            pruned_model.blocks[i].attn.proj.bias,
                                                            block, original_num_heads, original_v_dim, new_emb_dims)

            mhsa = MultiHeadSelfAttention(qkv_weights, qkv_bias, proj_weights, proj_bias, original_head_dim, qk_dim,
                                          v_dim, num_heads)

            fc1_weights, fc1_bias, fc2_weights, fc2_bias = get_mlp_weights_bias(pruned_model.blocks[i].mlp, block,
                                                                                new_emb_dims)

            norm1_weights = pruned_model.blocks[i].norm1.weight.data[new_emb_dims]
            norm1_bias = pruned_model.blocks[i].norm1.bias.data[new_emb_dims]
            norm2_weights = pruned_model.blocks[i].norm2.weight.data[new_emb_dims]
            norm2_bias = pruned_model.blocks[i].norm2.bias.data[new_emb_dims]

            enc_block = EncoderBlock(mhsa, fc1_weights, fc1_bias, fc2_weights, fc2_bias, norm1_weights, norm1_bias,
                                     norm2_weights, norm2_bias)

            enc_block.norm1.eps = original_eps
            enc_block.norm2.eps = original_eps

            self.blocks_list.append(enc_block)

        self.blocks = nn.Sequential(*self.blocks_list)

        last_norm_weights = pruned_model.norm.weight.data[new_emb_dims]
        last_norm_bias = pruned_model.norm.bias.data[new_emb_dims]

        head_weights = pruned_model.head.weight[:, new_emb_dims]
        head_bias = pruned_model.head.bias

        self.norm = nn.LayerNorm(len(new_emb_dims), eps=original_eps)
        self.head = nn.Linear(len(new_emb_dims), pruned_model.head.out_features)

        with torch.no_grad():
            self.norm.weight.copy_(last_norm_weights)
            self.norm.bias.copy_(last_norm_bias)
            self.head.weight.copy_(head_weights)
            self.head.bias.copy_(head_bias)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token_exp = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token_exp, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x
