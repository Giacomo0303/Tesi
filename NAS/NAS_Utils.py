import torch
import torch.nn.utils.prune as pruning


def set_initial_masks(model):
    # patch embedding mask
    patch_emb = model.patch_embed
    pruning.identity(patch_emb.proj, name="weight")
    pruning.identity(patch_emb.proj, name="bias")

    # transformer blocks masks
    for block in model.blocks:
        # first layer norm mask
        pruning.identity(block.norm1, name="weight")
        pruning.identity(block.norm1, name="bias")

        # multi head self attention masks
        attn = block.attn
        pruning.identity(attn.qkv, name="weight")
        pruning.identity(attn.qkv, name="bias")
        pruning.identity(attn.proj, name="weight")
        pruning.identity(attn.proj, name="bias")

        # second layer norm mask
        pruning.identity(block.norm2, name="weight")
        pruning.identity(block.norm2, name="bias")

        # mlp masks
        mlp = block.mlp
        pruning.identity(mlp.fc1, name="weight")
        pruning.identity(mlp.fc1, name="bias")
        pruning.identity(mlp.fc2, name="weight")
        pruning.identity(mlp.fc2, name="bias")

    #final norm mask
    pruning.identity(model.norm, name="weight")
    pruning.identity(model.norm, name="bias")

    # mlp head mask
    pruning.identity(model.head, name="weight")
    pruning.identity(model.head, name="bias")

    #masking of external parameters (cls token e position embed)
    model.register_buffer("cls_token_mask", torch.ones_like(model.cls_token))
    model.register_buffer("pos_embed_mask", torch.ones_like(model.pos_embed))