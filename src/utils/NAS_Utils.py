import torch
import time
import timm
import json
from src.utils.FineTuneUtils import EarlyStopping, train_model
from src.utils.CompressedViT import CompressedViT
from src.NAS.HybridNAS import HybridNAS
from src.utils.PruneUtils import set_initial_masks


def load_model(model_name, num_classes, path):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def pruningNAS(model, loss_fn, search_loader, device, initial_params_count, depth_limit, original_head_dim, threshold):
    nas_start = time.time()
    nas = HybridNAS(model, loss_fn=loss_fn, search_loader=search_loader, device=device,
                    original_params=initial_params_count, threshold=threshold)
    state, best_val = nas.search(depth_limit=depth_limit)
    nas_duration = time.time() - nas_start
    set_initial_masks(model)
    model = nas.apply_pruning(state, model)
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


def createPruningReport(model):
    start_state = {}
    start_state["embed_pruned_dims"] = 0
    start_state["blocks"] = []

    n_blocks = len(model.blocks)

    for block in range(n_blocks):
        block_state = {
            "head_pruned_idx": 0,
            "qk_pruned_dims": 0,
            "v_proj_pruned_dims": 0,
            "mlp_pruned_dims": 0
        }
        start_state["blocks"].append(block_state)

    start_state["obj_val"] = -float("inf")
    start_state["depth"] = 0

    return start_state


def updatePruningReport(pruningReport, state):
    pruningReport["obj_val"] = state["obj_val"]
    pruningReport["depth"] = state["depth"]

    pruningReport["embed_pruned_dims"] += len(state["embed_pruned_dims"])
    n_blocks = len(pruningReport["blocks"])

    for i, block in enumerate(pruningReport["blocks"]):
        block["head_pruned_idx"] += len(state["blocks"][i]["head_pruned_idx"])
        block["qk_pruned_dims"] += len(state["blocks"][i]["qk_pruned_dims"])
        block["v_proj_pruned_dims"] += len(state["blocks"][i]["v_proj_pruned_dims"])
        block["mlp_pruned_dims"] += len(state["blocks"][i]["mlp_pruned_dims"])

    return pruningReport


def savePruningReport(report, path):
    # Salva il file JSON
    with open(path, 'w') as f:
        json.dump(report, f, indent=4)
