import torch
from random import choice, sample
from src.utils.PruneUtils import head_alignment, importance_score
from src.utils.actions import QKPruning, VProjPruning, HeadPruning, MLPPruning, EmbPruning


class RandomQKPruning(QKPruning):
    def __init__(self, chunk_size=8):
        super().__init__(chunk_size=chunk_size)

    def find_target(self, model):
        blocks_idx = list(range(len(model.blocks)))

        while len(blocks_idx) > 0:
            chosen_block_idx = choice(blocks_idx)
            chosen_block = model.blocks[chosen_block_idx]

            head_aligned = head_alignment(chosen_block.attn)
            Q = head_aligned.Q
            K = head_aligned.K

            n_dims = Q.weight.shape[1]

            candidates = []

            for dim in range(n_dims):
                # check if dim is already pruned using the bias
                if torch.all(Q.bias_mask[:, dim] == 0.0):
                    continue

                candidates.append(dim)

            if len(candidates) < self.chunk_size:
                blocks_idx.remove(chosen_block_idx)
                continue

            selected_dims = sample(candidates, k=self.chunk_size)
            return chosen_block_idx, selected_dims

        return 0, []


class RandomVProjPruning(VProjPruning):
    def __init__(self, chunk_size=8):
        super().__init__(chunk_size=chunk_size)

    def find_target(self, model):
        blocks_idx = list(range(len(model.blocks)))

        while len(blocks_idx) > 0:
            chosen_block_idx = choice(blocks_idx)
            chosen_block = model.blocks[chosen_block_idx]

            head_aligned = head_alignment(chosen_block.attn)
            V = head_aligned.V

            n_dims = V.weight.shape[1]

            candidates = []

            for dim in range(n_dims):
                # check if dim is already pruned using the bias
                if torch.all(V.bias_mask[:, dim] == 0.0):
                    continue

                candidates.append(dim)

            if len(candidates) < self.chunk_size:
                blocks_idx.remove(chosen_block_idx)
                continue

            selected_dims = sample(candidates, k=self.chunk_size)
            return chosen_block_idx, selected_dims

        return 0, []


class RandomHeadPruning(HeadPruning):
    def __init__(self):
        super().__init__()

    def find_target(self, model):
        blocks_idx = list(range(len(model.blocks)))

        while len(blocks_idx) > 0:
            chosen_block_idx = choice(blocks_idx)
            chosen_block = model.blocks[chosen_block_idx]

            head_aligned = head_alignment(chosen_block.attn)
            num_heads = head_aligned.Q.weight.shape[0]

            candidates = []

            for head in range(num_heads):
                # check if dim is already pruned using the bias
                if torch.all(head_aligned.Q.bias_mask[head, :] == 0.0) and torch.all(
                        head_aligned.V.bias_mask[head, :] == 0.0):
                    continue

                candidates.append(head)

            if len(candidates) < self.chunk_size:
                blocks_idx.remove(chosen_block_idx)
                continue

            selected_dims = choice(candidates)
            return chosen_block_idx, selected_dims

        return 0, -1


class RandomMLPPruning(MLPPruning):
    def __init__(self, chunk_size=32):
        super().__init__(chunk_size=chunk_size)

    def find_target(self, model):
        blocks_idx = list(range(len(model.blocks)))

        while len(blocks_idx) > 0:
            chosen_block_idx = choice(blocks_idx)
            chosen_block = model.blocks[chosen_block_idx]

            n_dims = chosen_block.mlp.fc1.weight.shape[0]

            candidates = []

            for dim in range(n_dims):
                # check if dim is already pruned using the bias
                if chosen_block.mlp.fc1.bias_mask[dim] == 0.0:
                    continue

                candidates.append(dim)

            if len(candidates) < self.chunk_size:
                blocks_idx.remove(chosen_block_idx)
                continue

            selected_dims = sample(candidates, k=self.chunk_size)
            return chosen_block_idx, selected_dims

        return 0, []


class RandomEmbPruning(EmbPruning):
    def __init__(self, chunk_size=8):
        super().__init__(chunk_size=chunk_size)

    def find_target(self, model):

        n_dims = model.cls_token_mask[0, 0, :].shape[0]
        candidates = []

        for dim in range(n_dims):
            if model.cls_token_mask[0, 0, dim] == 0.0:
                continue

            candidates.append(dim)

        if len(candidates) < self.chunk_size:
            return []

        return sample(candidates, k=self.chunk_size)
