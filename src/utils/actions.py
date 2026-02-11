from abc import abstractmethod, ABC
from copy import deepcopy
import torch
from src.utils.PruneUtils import head_alignment, importance_score


class SearchAction(ABC):
    def __init__(self, action_name, state_key, chunk_size):
        self.action_name = action_name
        self.chunk_size = chunk_size
        self.state_key = state_key

    @abstractmethod
    def find_target(self, model):
        pass

    @abstractmethod
    def apply(self, state, target):
        pass


class QKPruning(SearchAction):
    def __init__(self, chunk_size=8):
        super().__init__(action_name="QK pruning", state_key="qk_pruned_dims", chunk_size=chunk_size)

    def find_target(self, model):
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

                importances = [
                    Q.weight_imp[:, dim, :],
                    Q.bias_imp[:, dim],
                    K.weight_imp[:, dim, :],
                    K.bias_imp[:, dim]
                ]

                dim_imp = importance_score(importances)
                candidates.append((dim, dim_imp))

            if len(candidates) < self.chunk_size:
                continue

            # sort in ascending order
            candidates.sort(key=lambda x: x[1])

            best_dims = candidates[:self.chunk_size]
            best_indices = [x[0] for x in best_dims]
            best_values = [x[1] for x in best_dims]

            chunk_importance = sum(best_values)

            if chunk_importance < best_target["importance"]:
                best_target["position"] = (b, best_indices)
                best_target["importance"] = chunk_importance

        return best_target["position"]

    def apply(self, state, target):
        block, dims = target
        if not dims:
            return None
        next_state = deepcopy(state)
        next_state["blocks"][block][self.state_key].extend(dims)
        next_state["last_act"] = self.action_name
        return next_state


class VProjPruning(SearchAction):
    def __init__(self, chunk_size=8):
        super().__init__(action_name="V/Proj pruning", state_key="v_proj_pruned_dims", chunk_size=chunk_size)

    def find_target(self, model):
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

                importances = [
                    V.weight_imp[:, dim, :],
                    V.bias_imp[:, dim],
                    proj.weight_imp[:, :, dim]  # prune input_size
                    # no need to prune proj bias because is applied on output, not on input
                ]

                dim_imp = importance_score(importances)
                candidates.append((dim, dim_imp))

            if len(candidates) < self.chunk_size:
                continue

            # sort in ascending order
            candidates.sort(key=lambda x: x[1])

            best_dims = candidates[:self.chunk_size]
            best_indices = [x[0] for x in best_dims]
            best_values = [x[1] for x in best_dims]

            chunk_importance = sum(best_values)

            if chunk_importance < target["importance"]:
                target["position"] = (b, best_indices)
                target["importance"] = chunk_importance

        return target["position"]

    def apply(self, state, target):
        block, dims = target
        if not dims:
            return None
        next_state = deepcopy(state)
        next_state["blocks"][block][self.state_key].extend(dims)
        next_state["last_act"] = self.action_name
        return next_state


class HeadPruning(SearchAction):
    def __init__(self):
        super().__init__(action_name="Head pruning", state_key="head_pruned_idx", chunk_size=1)

    def find_target(self, model):
        target = {
            "position": (0, -1),  # (block, head_n)
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

                importances = [
                    Q.weight_imp[head, :, :],
                    Q.bias_imp[head, :],
                    K.weight_imp[head, :, :],
                    K.bias_imp[head, :],
                    V.weight_imp[head, :, :],
                    V.bias_imp[head, :],
                    proj.weight_imp[:, head, :]
                ]

                imp = importance_score(importances)

                if imp < target["importance"]:
                    target["importance"] = imp
                    target["position"] = (b, head)

        return target["position"]

    def apply(self, state, target):
        block, dim = target
        if dim == -1:
            return None
        next_state = deepcopy(state)
        next_state["blocks"][block][self.state_key].append(dim)
        next_state["last_act"] = self.action_name
        return next_state


class MLPPruning(SearchAction):
    def __init__(self, chunk_size=32):
        super().__init__(action_name="MLP pruning", state_key="mlp_pruned_dims", chunk_size=chunk_size)

    def find_target(self, model):
        target = {
            "position": (0, []),  # (block, dims)
            "importance": float("inf")
        }

        # importance_score takes too much time, so parallelize to speed up
        for b, block in enumerate(model.blocks):
            mlp = block.mlp

            # importance of fc1 neurons
            imp_fc1_w = torch.sum(mlp.fc1.weight_orig.imp, dim=1)
            imp_fc1_b = mlp.fc1.bias_orig.imp

            # importance of fc2 input dims
            imp_fc2_w = torch.sum(mlp.fc2.weight_orig.imp, dim=0)

            # final scores
            imp_per_neuron = imp_fc1_w + imp_fc1_b + imp_fc2_w

            # masking already pruned dims
            imp_per_neuron[mlp.fc1.bias_mask == 0.0] = float('inf')

            if (mlp.fc1.bias_mask > 0).sum() < self.chunk_size:
                continue

            # take the dims with the smallest abs importances
            _, indices = torch.topk(imp_per_neuron, k=self.chunk_size, largest=False)

            best_dims = imp_per_neuron[indices]

            # Se hai selezionato dei valori 'inf' (già prunati), il chunk è da scartare
            if torch.isinf(best_dims).any():
                continue

            chunk_importance = torch.sum(best_dims)
            if chunk_importance < target["importance"]:
                target["position"] = (b, indices.tolist())
                target["importance"] = chunk_importance.item()

        return target["position"]

    def apply(self, state, target):
        block, dims = target
        if not dims:
            return None
        next_state = deepcopy(state)
        next_state["blocks"][block][self.state_key].extend(dims)
        next_state["last_act"] = self.action_name
        return next_state


class EmbPruning(SearchAction):
    def __init__(self, chunk_size=8):
        super().__init__(action_name="Emb pruning", state_key="embed_pruned_dims", chunk_size=chunk_size)

    def find_target(self, model):
        # like for mlp calculation takes too long so we got to parallelize

        # contribution of cls token
        total_dim_importances = torch.sum(model.cls_token_orig.imp, dim=(0, 1))

        # contribution of positional embedding
        total_dim_importances += torch.sum(model.pos_embed_orig.imp, dim=(0, 1))

        # contribution of patch_embedding
        total_dim_importances += torch.sum(model.patch_embed.proj.weight_orig.imp,
                                           dim=(1, 2, 3))  # sum importances for every dim of conv
        total_dim_importances += model.patch_embed.proj.bias_orig.imp  # bias are [384]

        for block in model.blocks:
            # contribution of LayerNorm1
            total_dim_importances += block.norm1.weight_orig.imp
            total_dim_importances += block.norm1.bias_orig.imp

            # contribution of QKV
            total_dim_importances += torch.sum(block.attn.qkv.weight_orig.imp, dim=0)

            # contribution of Proj
            total_dim_importances += torch.sum(block.attn.proj.weight_orig.imp, dim=1)
            total_dim_importances += block.attn.proj.bias_orig.imp

            # contribution of LayerNorm2
            total_dim_importances += block.norm2.weight_orig.imp
            total_dim_importances += block.norm2.bias_orig.imp

            # contribution of FC
            total_dim_importances += torch.sum(block.mlp.fc1.weight_orig.imp, dim=0)
            total_dim_importances += torch.sum(block.mlp.fc2.weight_orig.imp, dim=1)
            total_dim_importances += block.mlp.fc2.bias_orig.imp

        # contribution of last LayerNorm
        total_dim_importances += model.norm.weight_orig.imp
        total_dim_importances += model.norm.bias_orig.imp

        # contribution of classification head
        total_dim_importances += torch.sum(model.head.weight_orig.imp, dim=0)

        # Mascheriamo le dimensioni già prunate
        # cls_token_mask shape: [1, 1, dim] -> prendiamo [0,0,:] per avere il vettore [dim]
        current_mask = model.cls_token_mask[0, 0, :]
        total_dim_importances[current_mask == 0.0] = float('inf')

        # Controllo di sicurezza
        if (current_mask > 0).sum() < self.chunk_size:
            return []

        # Troviamo gli indici con magnitudine minore (i più vicini a zero)
        _, indices = torch.topk(total_dim_importances, k=self.chunk_size, largest=False)

        # Restituiamo direttamente la lista degli indici
        return indices.tolist()

    def apply(self, state, target):
        if not target:
            return None
        next_state = deepcopy(state)
        next_state[self.state_key].extend(target)
        next_state["last_act"] = self.action_name
        return next_state
