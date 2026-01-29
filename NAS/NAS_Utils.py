import torch
import torch.nn.utils.prune as pruning
from Pruning.PruneUtils import compute_grads, importance_score, head_alignment, importance_score_no_square
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
    pruning.identity(model, name="cls_token")
    pruning.identity(model, name="pos_embed")


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


def count_params_no_mask(model):
    return float(sum(p.numel() for p in model.parameters()) / 1e6)


def compute_obj(model, loss_fn, device, dataloader, original_params):
    _, accuracy = compute_grads(model, loss_fn, device, dataloader)
    params = count_parameters(model)

    return log2(accuracy) - 2.0 * log2(params / original_params), accuracy, params


# QK PRUNING
# rimuoviamo chunk_size dimensioni da ogni testa, in totale chunk_size * num_heads dimensioni

def find_target_QK(model, chunk_size=8) -> tuple:
    best_target = {
        "position": (0, []),  # (block_index, [lista_di_8_indici])
        "importance": float("inf")
    }

    # calculate importances of every Q_i, K_i in every mhsa
    for b, block in enumerate(model.blocks):
        head_aligned = head_alignment(block.attn)
        Q = head_aligned.Q
        K = head_aligned.K

        # get dimensions to prune
        n_dims = Q.weight.shape[1]

        candidates = []

        for dim in range(n_dims):
            # check if dim is already pruned using the bias
            if torch.all(Q.bias_mask[:, dim] == 0.0):
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

            signed_imp = importance_score_no_square(weights, grads)
            candidates.append((dim, signed_imp))

        if len(candidates) < chunk_size:
            continue

        # sort in ascending order by abs of signed importances
        candidates.sort(key=lambda x: abs(x[1]))

        best_dims = candidates[:chunk_size]
        best_indices = [x[0] for x in best_dims]
        best_values = [x[1] for x in best_dims]

        chunk_importance = (sum(best_values)) ** 2

        if chunk_importance < best_target["importance"]:
            best_target["position"] = (b, best_indices)
            best_target["importance"] = chunk_importance

    return best_target["position"]


# V/PROJ PRUNING

def find_target_V_proj(model, chunk_size=8) -> tuple:
    target = {
        "position": (0, []),  # (num_block, dims)
        "importance": float("inf")
    }

    # calculate importances of every V_i and proj_i in every mhsa
    for b, block in enumerate(model.blocks):
        # head alignment
        head_aligned = head_alignment(block.attn)
        V = head_aligned.V
        proj = head_aligned.proj

        n_dims = V.weight.shape[1]

        candidates = []

        for dim in range(n_dims):
            # checking already pruned dimensions
            if torch.all(V.bias_mask[:, dim] == 0.0):
                continue

            weights = [
                V.weight[:, dim, :],
                V.bias[:, dim],
                proj.weight[:, :, dim]  # prune input_size
                # no need to prune proj bias because is applied on output, not on input
            ]

            grads = [
                V.weight_grad[:, dim, :],
                V.bias_grad[:, dim],
                proj.weight_grad[:, :, dim]
            ]

            signed_imp = importance_score_no_square(weights, grads)
            candidates.append((dim, signed_imp))

        if len(candidates) < chunk_size:
            continue

        # sort in ascending order by abs of signed importances
        candidates.sort(key=lambda x: abs(x[1]))

        best_dims = candidates[:chunk_size]
        best_indices = [x[0] for x in best_dims]
        best_values = [x[1] for x in best_dims]

        chunk_importance = (sum(best_values)) ** 2

        if chunk_importance < target["importance"]:
            target["position"] = (b, best_indices)
            target["importance"] = chunk_importance

    return target["position"]


# HEAD PRUNING

def find_target_head(model) -> tuple:
    target = {
        "position": (0, 0),  # (block, head_n)
        "importance": float("inf")
    }

    for b, block in enumerate(model.blocks):
        head_aligned = head_alignment(block.attn)
        Q = head_aligned.Q
        K = head_aligned.K
        V = head_aligned.V
        proj = head_aligned.proj

        num_heads = Q.weight.shape[0]

        for head in range(num_heads):
            # if all Q and V bias of a head are zero then it is pruned
            if torch.all(Q.bias_mask[head, :] == 0.0) and torch.all(V.bias_mask[head, :] == 0.0):
                continue

            weights = [
                Q.weight[head, :, :],
                Q.bias[head, :],
                K.weight[head, :, :],
                K.bias[head, :],
                V.weight[head, :, :],
                V.bias[head, :],
                proj.weight[:, head, :]
            ]

            grads = [
                Q.weight_grad[head, :, :],
                Q.bias_grad[head, :],
                K.weight_grad[head, :, :],
                K.bias_grad[head, :],
                V.weight_grad[head, :, :],
                V.bias_grad[head, :],
                proj.weight_grad[:, head, :]
            ]

            imp = importance_score(weights, grads)

            if imp < target["importance"]:
                target["importance"] = imp
                target["position"] = (b, head)

    return target["position"]


# MLP PRUNING

def find_target_mlp(model, chunk_size=32) -> tuple:
    target = {
        "position": (0, []),  # (block, dims)
        "importance": float("inf")
    }

    # importance_score takes too much time, so parallelize to speed up
    for b, block in enumerate(model.blocks):
        mlp = block.mlp

        # importance of fc1 neurons
        imp_fc1_w = torch.sum(mlp.fc1.weight * mlp.fc1.weight_orig.grad, dim=1)
        imp_fc1_b = mlp.fc1.bias * mlp.fc1.bias_orig.grad

        # importance of fc2 input dims
        imp_fc2_w = torch.sum(mlp.fc2.weight * mlp.fc2.weight_orig.grad, dim=0)

        # final scores
        signed_imp_per_neuron = imp_fc1_w + imp_fc1_b + imp_fc2_w

        abs_imp_per_neuron = torch.abs(signed_imp_per_neuron)

        # masking already pruned dims
        abs_imp_per_neuron[mlp.fc1.bias_mask == 0.0] = float('inf')

        if (mlp.fc1.bias_mask > 0).sum() < chunk_size:
            continue

        # take the dims with the smallest abs importances
        _, indices = torch.topk(abs_imp_per_neuron, k=chunk_size, largest=False)

        best_dims = signed_imp_per_neuron[indices]

        # Se hai selezionato dei valori 'inf' (già prunati), il chunk è da scartare
        if torch.isinf(best_dims).any():
            continue

        chunk_importance = (torch.sum(best_dims)) ** 2
        if chunk_importance < target["importance"]:
            target["position"] = (b, indices.tolist())
            target["importance"] = chunk_importance.item()

    return target["position"]


# EMB PRUNING

def find_target_emb(model, chunk_size=8) -> list[int]:
    # like for mlp calculation takes too long so we got to parallelize

    # contribution of cls token
    pruned_cls_token = model.cls_token * model.cls_token_mask
    total_dim_importances = torch.sum(pruned_cls_token * model.cls_token_orig.grad, dim=(0, 1))

    # contribution of positional embedding
    pruned_pos_embed = model.pos_embed * model.pos_embed_mask
    total_dim_importances += torch.sum(pruned_pos_embed * model.pos_embed_orig.grad, dim=(0, 1))

    # contribution of patch_embedding
    total_dim_importances += torch.sum(model.patch_embed.proj.weight * model.patch_embed.proj.weight_orig.grad,
                                       dim=(1, 2, 3))  # sum importances for every dim of conv
    total_dim_importances += model.patch_embed.proj.bias * model.patch_embed.proj.bias_orig.grad  # bias are [384]

    for block in model.blocks:
        # contribution of LayerNorm1
        total_dim_importances += block.norm1.weight * block.norm1.weight_orig.grad
        total_dim_importances += block.norm1.bias * block.norm1.bias_orig.grad

        # contribution of QKV
        total_dim_importances += torch.sum(block.attn.qkv.weight * block.attn.qkv.weight_orig.grad, dim=0)

        # contribution of Proj
        total_dim_importances += torch.sum(block.attn.proj.weight * block.attn.proj.weight_orig.grad, dim=1)
        total_dim_importances += block.attn.proj.bias * block.attn.proj.bias_orig.grad

        # contribution of LayerNorm2
        total_dim_importances += block.norm2.weight * block.norm2.weight_orig.grad
        total_dim_importances += block.norm2.bias * block.norm2.bias_orig.grad

        # contribution of FC
        total_dim_importances += torch.sum(block.mlp.fc1.weight * block.mlp.fc1.weight_orig.grad, dim=0)
        total_dim_importances += torch.sum(block.mlp.fc2.weight * block.mlp.fc2.weight_orig.grad, dim=1)
        total_dim_importances += block.mlp.fc2.bias * block.mlp.fc2.bias_orig.grad

    # contribution of last LayerNorm
    total_dim_importances += model.norm.weight * model.norm.weight_orig.grad
    total_dim_importances += model.norm.bias * model.norm.bias_orig.grad

    # contribution of classification head
    total_dim_importances += torch.sum(model.head.weight * model.head.weight_orig.grad, dim=0)

    # --- SELEZIONE (Chunk Strategy) ---

    # Usiamo il valore ASSOLUTO per decidere chi tagliare
    abs_imp = torch.abs(total_dim_importances)

    # Mascheriamo le dimensioni già prunate
    # cls_token_mask shape: [1, 1, dim] -> prendiamo [0,0,:] per avere il vettore [dim]
    current_mask = model.cls_token_mask[0, 0, :]
    abs_imp[current_mask == 0.0] = float('inf')

    # Controllo di sicurezza
    if (current_mask > 0).sum() < chunk_size:
        return []

    # Troviamo gli indici con magnitudine minore (i più vicini a zero)
    _, indices = torch.topk(abs_imp, k=chunk_size, largest=False)

    # Restituiamo direttamente la lista degli indici
    return indices.tolist()
