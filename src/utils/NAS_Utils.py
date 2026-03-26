import os
import re
import torch
import time
import timm
import json
from src.utils.FineTuneUtils import EarlyStopping, train_model
from src.utils.Compression import CompressedViT, CompressedDeiT
from src.NAS.HybridNAS import HybridNAS
from src.utils.PruneUtils import set_initial_masks
from src.utils.XAIutils import analize_mlp, analize_qk, analize_vproj, analize_head


def load_model(model_name, num_classes, path):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def pruningNAS(model, loss_fn, search_loader, device, initial_params_count, depth_limit, original_head_dim, threshold,
               actions="guided", search=True):
    nas_start = time.time()
    nas = HybridNAS(model, loss_fn=loss_fn, search_loader=search_loader, device=device,
                    original_params=initial_params_count, threshold=threshold, actions=actions)
    if search:
        state, best_val = nas.search(depth_limit=depth_limit)
    else:
        state, best_val = nas.random_search(depth_limit=depth_limit)
    nas_duration = time.time() - nas_start
    set_initial_masks(model)
    model = nas.apply_pruning(state, model)
    if hasattr(model, 'head_dist') and model.head_dist is not None:
        comp_model = CompressedDeiT(state, model, original_head_dim=original_head_dim).to(device)
    else:
        comp_model = CompressedViT(state, model, original_head_dim=original_head_dim).to(device)

    return comp_model, nas_duration, state


def recoveryFineTune(model, lr, weight_decay, max_epochs, early_stop_path, patience, min_delta, device, train_loader,
                     val_loader, loss_fn, teacher_model=None, T=3.0):
    ft_start = time.time()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

    earlystop = EarlyStopping(path=early_stop_path, patience=patience, min_delta=min_delta)

    # Passiamo val_dataloader per vedere i log nel training, ma ignoriamo il return qui
    _, _, _ = train_model(
        model, max_epochs, optimizer=optim, device=device,
        train_dataloader=train_loader, loss_fn=loss_fn,
        scheduler=scheduler, val_dataloader=val_loader,
        early_stopping=earlystop, teacher_model=teacher_model, T=T
    )
    ft_duration = time.time() - ft_start

    return ft_duration


def save_model(model, path):
    try:
        torch.save(model, path)
        print(f"✅ Modello completo salvato in: {path}")
    except Exception as e:
        print(f"❌ Errore durante il salvataggio del modello: {e}")


class PruningReport:
    def __init__(self, model):
        self.n_blocks = len(model.blocks)
        self.model = model
        self.Embedding = None
        self.blocks = []
        self.buildPruningReport()

    def buildPruningReport(self):
        self.Embedding = {
            "kept": list(range(self.model.patch_embed.proj.weight.shape[0])),
            "pruned": []
        }

        for i in range(self.n_blocks):
            Heads = {
                "kept": list(range(self.model.blocks[i].attn.num_heads)),
                "pruned": []
            }
            QK = {
                "kept": list(range(self.model.blocks[i].attn.head_dim)),
                "pruned": []
            }
            VProj = {
                "kept": list(range(self.model.blocks[i].attn.head_dim)),
                "pruned": []
            }
            MLP = {
                "kept": list(range(self.model.blocks[i].mlp.fc1.weight.shape[0])),
                "pruned": []
            }
            self.blocks.append({"Heads": Heads, "QK": QK, "VProj": VProj, "MLP": MLP})

    def updateDim(self, pruned_dims, destinationDim):
        pruned_dims.sort(reverse=True)
        for dim in pruned_dims:
            original_pruned_idx = destinationDim["kept"].pop(dim)
            destinationDim["pruned"].append(original_pruned_idx)

    def updatePruningReport(self, state):
        self.updateDim(state["embed_pruned_dims"], self.Embedding)

        for block in range(self.n_blocks):
            self.updateDim(state["blocks"][block]["head_pruned_idx"], self.blocks[block]["Heads"])
            self.updateDim(state["blocks"][block]["qk_pruned_dims"], self.blocks[block]["QK"])
            self.updateDim(state["blocks"][block]["v_proj_pruned_dims"], self.blocks[block]["VProj"])
            self.updateDim(state["blocks"][block]["mlp_pruned_dims"], self.blocks[block]["MLP"])

    def savePruningReport(self, path):
        export_data = {}
        export_data["Embedding"] = {
            "kept": self.Embedding["kept"],
            "pruned": self.Embedding["pruned"],
            "num_pruned": len(self.Embedding["pruned"])
        }
        export_data["blocks"] = []
        for block in self.blocks:
            block_export = {}
            for key in ["Heads", "QK", "VProj", "MLP"]:
                block_export[key] = {
                    "kept": block[key]["kept"],
                    "pruned": block[key]["pruned"],
                    "num_pruned": len(block[key]["pruned"])
                }
            export_data["blocks"].append(block_export)

        json_str = json.dumps(export_data, indent=4)

        def collapse_list(match):
            inner_content = re.sub(r'\s+', ' ', match.group(1)).strip()
            return f"[{inner_content}]"

        compact_json_str = re.sub(r'\[([\d\s,]*)\]', collapse_list, json_str)

        with open(path, 'w') as f:
            f.write(compact_json_str)


def save_plots(original_model, finetuned_model, pruning_report, save_path, iter):
    mlp_dir = os.path.join(save_path, "MLP")
    qk_dir = os.path.join(save_path, "QK")
    vproj_dir = os.path.join(save_path, "VProj")
    head_dir = os.path.join(save_path, "Head")

    os.makedirs(mlp_dir, exist_ok=True)
    os.makedirs(qk_dir, exist_ok=True)
    os.makedirs(vproj_dir, exist_ok=True)
    os.makedirs(head_dir, exist_ok=True)

    analize_mlp(original_model, finetuned_model, pruning_report, pruned=True,
                save_path=os.path.join(mlp_dir, f"mlp_iter_{iter}.png"))

    analize_qk(original_model, finetuned_model, pruning_report, pruned=True,
               save_path=os.path.join(qk_dir, f"qk_iter_{iter}.png"))

    analize_vproj(original_model, finetuned_model, pruning_report, pruned=True,
                  save_path=os.path.join(vproj_dir, f"vproj_iter_{iter}.png"))

    analize_head(original_model, finetuned_model, pruning_report, pruned=True,
                 save_path=os.path.join(head_dir, f"head_iter_{iter}.png"))
