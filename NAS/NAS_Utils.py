import torch
import torch.nn.utils.prune as pruning
from Pruning.PruneUtils import compute_grads, importance_score, head_alignment
from math import log2


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

    # final norm mask
    pruning.identity(model.norm, name="weight")
    pruning.identity(model.norm, name="bias")

    # mlp head mask
    pruning.identity(model.head, name="weight")
    pruning.identity(model.head, name="bias")

    # masking of external parameters (cls token e position embed)
    model.register_buffer("cls_token_mask", torch.ones_like(model.cls_token))
    model.register_buffer("pos_embed_mask", torch.ones_like(model.pos_embed))


def count_parameters(model):
    n_params = 0

    # patch_embedding params
    n_params += torch.sum(model.patch_embed.proj.weight_mask)
    n_params += torch.sum(model.patch_embed.proj.bias_mask)

    # transformer block
    for block in model.blocks:
        # first layer norm
        n_params += torch.sum(block.norm1.weight_mask)
        n_params += torch.sum(block.norm1.bias_mask)

        # mhsa
        n_params += torch.sum(block.attn.qkv.weight_mask)
        n_params += torch.sum(block.attn.qkv.bias_mask)
        n_params += torch.sum(block.attn.proj.weight_mask)
        n_params += torch.sum(block.attn.proj.bias_mask)

        # second layer norm
        n_params += torch.sum(block.norm2.weight_mask)
        n_params += torch.sum(block.norm2.bias_mask)

        # mlp
        n_params += torch.sum(block.mlp.fc1.weight_mask)
        n_params += torch.sum(block.mlp.fc1.bias_mask)
        n_params += torch.sum(block.mlp.fc2.weight_mask)
        n_params += torch.sum(block.mlp.fc2.bias_mask)

    # last layer norm
    n_params += torch.sum(model.norm.weight_mask)
    n_params += torch.sum(model.norm.bias_mask)

    # head
    n_params += torch.sum(model.head.weight_mask)
    n_params += torch.sum(model.head.bias_mask)

    # cls token and position embedding
    n_params += torch.sum(model.cls_token_mask)
    n_params += torch.sum(model.pos_embed_mask)

    return float(n_params / 1e6)


def compute_obj(model, loss_fn, device, dataloader):
    loss = compute_grads(model, loss_fn, device, dataloader)
    params = count_parameters(model)
    # attualmente uso soltanto loss e param

    return -log2(loss) - log2(params)


def find_target_QK(model):
    target = {
        "position": (0, 0),  # (num_block, dim)
        "importance": float("inf")
    }

    # calculate importances of every Q_i, K_i in every mhsa
    for b, block in enumerate(model.blocks):
        head_aligned = head_alignment(block.attn)
        Q = head_aligned.Q
        K = head_aligned.K

        # get dimensions to prune
        n_dims = Q.weight.shape[1]

        for dim in range(n_dims):
            # check if dim is already pruned
            if Q.weight_mask[0, dim, 0].item() == 0.0:
                continue

            weights = [
                Q.weight[:, dim, :],
                Q.bias[:, dim],
                K.weight[:, dim, :],
                K.bias[:, dim]
            ]

            grads = [
                Q.weight_grad[:, dim, :],
                Q.bias_grad[:, dim],
                K.weight_grad[:, dim, :],
                K.bias_grad[:, dim]
            ]

            imp = importance_score(weights, grads)

            if imp < target["importance"]:
                target["importance"] = imp
                target["position"] = (b, dim)

    return target

def apply_QK_Prune(model, target:dict) -> None:
    block = target["position"][0]
    dim = target["position"][1]

    head_aligned = head_alignment(model.blocks[block].attn)
    Q = head_aligned.Q
    K = head_aligned.K

    #apply pruning
    Q.weight_mask[:, dim, :] = 0.0
    Q.bias_mask[:, dim] = 0.0

    K.weight_mask[:, dim, :] = 0.0
    K.bias_mask[:, dim] = 0.0

