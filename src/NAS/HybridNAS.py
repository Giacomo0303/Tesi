from random import choice

from src.utils.PruneUtils import set_initial_masks
from src.utils.actions import EmbPruning, HeadPruning, QKPruning, VProjPruning, MLPPruning
from src.utils.random_actions import RandomEmbPruning, RandomHeadPruning, RandomQKPruning, RandomVProjPruning, \
    RandomMLPPruning
from src.utils.PruneUtils import head_alignment, compute_obj, reset_masks
from copy import deepcopy


class HybridNAS:
    def __init__(self, model, loss_fn, search_loader, device, original_params, threshold, actions="guided"):
        self.base_model = model
        self.original_params = original_params
        self.loss_fn = loss_fn
        self.device = device
        self.dataloader = search_loader
        self.best_value = -float("inf")
        self.best_state = None
        self.threshold = threshold
        self.action_mode = actions
        if actions == "guided":
            self.actions = [
                EmbPruning(),
                HeadPruning(),
                QKPruning(),
                VProjPruning(),
                MLPPruning()
            ]
        elif actions == "random":
            self.actions = [
                RandomEmbPruning(),
                RandomHeadPruning(),
                RandomQKPruning(),
                RandomVProjPruning(),
                RandomMLPPruning()
            ]

    def build_initial_state(self) -> dict:
        start_state = {}
        start_state["embed_pruned_dims"] = []
        start_state["blocks"] = []

        n_blocks = len(self.base_model.blocks)

        for block in range(n_blocks):
            block_state = {
                "head_pruned_idx": [],
                "qk_pruned_dims": [],
                "v_proj_pruned_dims": [],
                "mlp_pruned_dims": []
            }
            start_state["blocks"].append(block_state)

        start_state["obj_val"] = -float("inf")
        start_state["depth"] = 0
        start_state["last_act"] = "Start"

        return start_state

    def bound(self, state):
        if (state["obj_val"] + self.threshold) < self.best_value:
            return True
        return False

    def branch(self, state, model) -> list[dict]:
        next_states = []
        for action in self.actions:
            target = action.find_target(model)
            next_state = action.apply(state, target)
            if next_state is not None:
                next_states.append(next_state)

        return next_states

    @staticmethod
    def apply_pruning(state, model):
        # prune cls token and position embedding
        model.cls_token_mask[:, :, state["embed_pruned_dims"]] = 0.0
        model.pos_embed_mask[:, :, state["embed_pruned_dims"]] = 0.0

        # pruning of patch embedding
        model.patch_embed.proj.weight_mask[state["embed_pruned_dims"], :, :, :] = 0.0
        model.patch_embed.proj.bias_mask[state["embed_pruned_dims"]] = 0.0

        for block, block_state in zip(model.blocks, state["blocks"]):
            # pruning of layerNorm1
            block.norm1.weight_mask[state["embed_pruned_dims"]] = 0.0
            block.norm1.bias_mask[state["embed_pruned_dims"]] = 0.0

            if not getattr(block.attn, 'is_empty', False):
                # pruning of QKV and proj
                # emb dimension first
                block.attn.qkv.weight_mask[:, state["embed_pruned_dims"]] = 0.0
                block.attn.proj.weight_mask[state["embed_pruned_dims"], :] = 0.0
                block.attn.proj.bias_mask[state["embed_pruned_dims"]] = 0.0

                aligned_head = head_alignment(block.attn)
                Q = aligned_head.Q
                K = aligned_head.K
                V = aligned_head.V
                proj = aligned_head.proj

                # head pruning
                Q.weight_mask[block_state["head_pruned_idx"], :, :] = 0.0
                Q.bias_mask[block_state["head_pruned_idx"], :] = 0.0
                K.weight_mask[block_state["head_pruned_idx"], :, :] = 0.0
                K.bias_mask[block_state["head_pruned_idx"], :] = 0.0
                V.weight_mask[block_state["head_pruned_idx"], :, :] = 0.0
                V.bias_mask[block_state["head_pruned_idx"], :] = 0.0
                proj.weight_mask[:, block_state["head_pruned_idx"], :] = 0.0

                # QK pruning
                Q.weight_mask[:, block_state["qk_pruned_dims"], :] = 0.0
                Q.bias_mask[:, block_state["qk_pruned_dims"]] = 0.0
                K.weight_mask[:, block_state["qk_pruned_dims"], :] = 0.0
                K.bias_mask[:, block_state["qk_pruned_dims"]] = 0.0

                # V/Proj pruning
                V.weight_mask[:, block_state["v_proj_pruned_dims"], :] = 0.0
                V.bias_mask[:, block_state["v_proj_pruned_dims"]] = 0.0
                proj.weight_mask[:, :, block_state["v_proj_pruned_dims"]] = 0.0

            # pruning of layerNorm2
            block.norm2.weight_mask[state["embed_pruned_dims"]] = 0.0
            block.norm2.bias_mask[state["embed_pruned_dims"]] = 0.0

            if not getattr(block.mlp, 'is_empty', False):
                # pruning of MLP
                # embed dim first
                block.mlp.fc1.weight_mask[:, state["embed_pruned_dims"]] = 0.0
                block.mlp.fc2.weight_mask[state["embed_pruned_dims"], :] = 0.0
                block.mlp.fc2.bias_mask[state["embed_pruned_dims"]] = 0.0

                # mlp pruning
                block.mlp.fc1.weight_mask[block_state["mlp_pruned_dims"], :] = 0.0
                block.mlp.fc1.bias_mask[block_state["mlp_pruned_dims"]] = 0.0
                block.mlp.fc2.weight_mask[:, block_state["mlp_pruned_dims"]] = 0.0

        # pruning last layer_norm and head
        model.norm.weight_mask[state["embed_pruned_dims"]] = 0.0
        model.norm.bias_mask[state["embed_pruned_dims"]] = 0.0

        model.head.weight_mask[:, state["embed_pruned_dims"]] = 0.0

        return model

    def eval_model(self, model, state, search_iter):
        obj_val, accuracy, params = compute_obj(model, self.loss_fn, device=self.device, dataloader=self.dataloader,
                                                original_params=self.original_params, imp=self.action_mode == "guided")
        state["obj_val"] = obj_val
        if search_iter > 0 and obj_val > self.best_value:
            print(f"--- NUOVO BEST TROVATO! --- Valore: {obj_val:.4f} (Precedente: {self.best_value:.4f})")
            self.best_value = obj_val
            self.best_state = deepcopy(state)
        return accuracy, params

    def search(self, depth_limit=None):
        start_state = self.build_initial_state()
        stack = [start_state]

        search_iterations = 0
        pruned_branches = 0

        working_model = deepcopy(self.base_model)
        set_initial_masks(working_model)

        print("--- Inizio Ricerca NAS ---")

        while len(stack) > 0:
            current_state = stack.pop()
            reset_masks(working_model)
            self.apply_pruning(state=current_state, model=working_model)
            acc, params = self.eval_model(working_model, current_state, search_iterations)

            search_iterations += 1

            print(
                f"Iter: {search_iterations} | Stack: {len(stack)} | Pruned: {pruned_branches} | Curr Val: {current_state['obj_val']:.4f} | Last Action: {current_state['last_act']} |Acc: {acc:.4f} | Params: {params:.4f}M")

            # depth limit
            if depth_limit is not None and current_state["depth"] >= depth_limit:
                continue

            # è fondamentale che il pruning venga fatto sullo stato corrente e non sui figli, così da garantire che il confronto sia fatto con il best value attuale
            if not (self.bound(current_state)):
                next_states = self.branch(current_state, working_model)
                for state in next_states:
                    state["depth"] = current_state["depth"] + 1
                    stack.append(state)
            else:
                pruned_branches += 1

        print("--- Ricerca Completata ---")
        print(f"Iterazioni totali: {search_iterations}")
        print(f"Rami potati (pruned): {pruned_branches}")
        print(f"Miglior Valore Trovato: {self.best_value:.4f}")

        return self.best_state, self.best_value

    def random_search(self, depth_limit=6):
        current_state = self.build_initial_state()
        working_model = deepcopy(self.base_model)
        set_initial_masks(working_model)

        self.eval_model(working_model, current_state, 0)

        n_actions = 0

        while n_actions < depth_limit:
            action = choice(self.actions)

            target = action.find_target(working_model)
            next_state = action.apply(current_state, target)

            if next_state is None:
                continue

            n_actions += 1
            current_state = next_state

            reset_masks(working_model)
            self.apply_pruning(state=current_state, model=working_model)
            acc, params = self.eval_model(working_model, current_state, search_iter=0)

            print(
                f"Step {n_actions}/{depth_limit} | Last Action: {current_state['last_act']} | Curr Val: {current_state['obj_val']:.4f} | Acc: {acc:.4f} | Params: {params:.4f}M")

        self.best_state = deepcopy(current_state)
        self.best_value = current_state['obj_val']

        print("--- Selezione Random Completata ---")
        print(f"Miglior Valore Trovato: {self.best_value:.4f}")

        return self.best_state, self.best_value
