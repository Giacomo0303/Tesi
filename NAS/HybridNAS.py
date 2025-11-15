import torch, timm
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import json
from NAS.NAS_Utils import count_parameters
from NAS_Utils import find_target_emb, find_target_QK, find_target_V_proj, find_target_head, find_target_mlp, \
    set_initial_masks, compute_obj
from Pruning.PruneUtils import head_alignment, compute_grads
from copy import deepcopy


class HybridNAS:
    def __init__(self, model, loss_fn, search_loader, device):
        self.base_model = model
        self.loss_fn = loss_fn
        self.device = device
        self.dataloader = search_loader
        self.best_value = -float("inf")
        self.best_state = None
        self.actions = [find_target_QK, find_target_V_proj, find_target_head, find_target_mlp, find_target_emb]

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

        return start_state

    def bound(self, state):
        if state["obj_val"] + 0.0025 < self.best_value:
            return True
        return False

    def branch(self, state, model) -> list[dict]:
        targets = []
        for action in self.actions:
            targets.append(action(model))

        next_states = [deepcopy(state) for i in range(len(self.actions))]
        # pruning QK
        block, dim = targets[0]
        next_states[0]["blocks"][block]["qk_pruned_dims"].append(dim)
        # pruning V/proj
        block, dim = targets[1]
        next_states[1]["blocks"][block]["v_proj_pruned_dims"].append(dim)
        # head pruning
        block, dim = targets[2]
        next_states[2]["blocks"][block]["head_pruned_idx"].append(dim)
        # mlp pruning
        block, dim = targets[3]
        next_states[3]["blocks"][block]["mlp_pruned_dims"].append(dim)
        # embed pruning
        dim = targets[4]
        next_states[4]["embed_pruned_dims"].append(dim)

        return next_states

    def apply_pruning(self, state):
        model = deepcopy(self.base_model)
        set_initial_masks(model)

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
        obj_val = compute_obj(model, self.loss_fn, device=self.device, dataloader=self.dataloader)
        state["obj_val"] = obj_val
        if search_iter > 0 and obj_val > self.best_value:
            print(f"--- NUOVO BEST TROVATO! --- Valore: {obj_val:.4f} (Precedente: {self.best_value:.4f})")
            self.best_value = obj_val
            self.best_state = deepcopy(state)

    def search(self):
        start_state = self.build_initial_state()
        stack = [start_state]

        search_iterations = 0
        pruned_branches = 0

        print("--- Inizio Ricerca NAS ---")

        while len(stack) > 0:
            current_state = stack.pop()
            model = self.apply_pruning(current_state)
            self.eval_model(model, current_state, search_iterations)

            search_iterations += 1

            print(
                f"Iter: {search_iterations} | Stack: {len(stack)} | Pruned: {pruned_branches} | Curr Val: {current_state['obj_val']:.4f} | Best Val: {self.best_value:.4f}")

            if not (self.bound(current_state)):
                next_states = self.branch(current_state, model)
                for state in next_states:
                    stack.append(state)
            else:

                pruned_branches += 1

        print("--- Ricerca Completata ---")
        print(f"Iterazioni totali: {search_iterations}")
        print(f"Rami potati (pruned): {pruned_branches}")
        print(f"Miglior Valore Trovato: {self.best_value:.4f}")

        return self.best_state, self.best_value


# TESTING
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 204
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=num_classes).to(device)
    checkpoint = torch.load("D:\\Tesi\\FirstFineTuning\\best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    data_config = timm.data.resolve_model_data_config(model)
    imagenet_mean, imagenet_std = data_config["mean"], data_config["std"]

    search_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])

    path = "D:\\Tesi\\Sets\\Set1\\search"
    batch_size = 128

    search_set = ImageFolder(root=path, transform=search_transform)
    search_loader = torch.utils.data.DataLoader(search_set, batch_size=batch_size, shuffle=False, num_workers=1)

    nas = HybridNAS(model, loss_fn=nn.CrossEntropyLoss(), search_loader=search_loader, device=device)
    # 1. Cattura lo stato finale e il valore
    state, best_val = nas.search()

    # 2. Controlla se la ricerca ha prodotto un risultato valido
    if state is not None:
        print(f"\n--- Statistiche del Modello Migliore (Valore: {best_val:.4f}) ---")

        # 3. Calcola le statistiche finali (è più pulito farlo fuori dal print)
        final_pruned_model = nas.apply_pruning(state)
        _, final_accuracy = compute_grads(final_pruned_model, nas.loss_fn, device, search_loader)
        final_params = count_parameters(final_pruned_model)

        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"Parametri (Milioni): {final_params:.2f}M")

        # 4. Salva il dizionario 'state' nel file JSON
        output_filename = "D:\\Tesi\\NAS\\best_architecture.json"
        print(f"Salvataggio dell'architettura in {output_filename}...")

        with open(output_filename, "w") as f:
            json.dump(state, f, indent=4)

        print("Salvataggio completato.")

    else:
        print("Ricerca completata senza trovare un best_state valido.")

