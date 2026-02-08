import time
import numpy as np
import timm
import torch
import json
from torch.utils.data import Subset, random_split
from torchvision.datasets import CIFAR100
from FirstFineTuning.FineTuneUtils import EarlyStopping, train_model
from NAS.CompressedViT import CompressedViT
from NAS.HybridNAS import HybridNAS
from NAS.NAS_Utils import set_initial_masks


def load_model(model_name, num_classes, path):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def get_search_set(train_set, n_per_classes, n_classes):
    train_indices = np.arange(len(train_set))

    targets = np.array(train_set.dataset.targets)
    train_labels = targets[train_set.indices]

    gen = np.random.default_rng()
    final_indices = []

    for cls in range(n_classes):
        cls_indices = train_indices[train_labels == cls]

        if len(cls_indices) >= n_per_classes:
            selected_indices = gen.choice(cls_indices, size=n_per_classes, replace=False)
        else:
            selected_indices = cls_indices

        final_indices.extend(selected_indices.tolist())

    return Subset(train_set, final_indices)


def split_dataset(transforms, seed):
    total_size = 50000
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    train_set = CIFAR100(root="D:\\Tesi\\CIFAR100", train=True, download=True, transform=transforms[0])
    val_set = CIFAR100(root="D:\\Tesi\\CIFAR100", train=True, download=True, transform=transforms[1])
    test_set = CIFAR100(root="D:\\Tesi\\CIFAR100", train=False, download=True, transform=transforms[2])

    generator = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(train_set, [train_size, val_size], generator=generator)

    train_set = Subset(train_set, train_split.indices)
    val_set = Subset(val_set, val_split.indices)

    return train_set, val_set, test_set


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
                     val_loader, loss_fn, teacher_model = None):
    ft_start = time.time()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

    earlystop = EarlyStopping(path=early_stop_path, patience=patience, min_delta=min_delta)

    # Passiamo val_dataloader per vedere i log nel training, ma ignoriamo il return qui
    _, _, _ = train_model(
        model, max_epochs, optimizer=optim, device=device,
        train_dataloader=train_loader, loss_fn=loss_fn,
        scheduler=scheduler, val_dataloader=val_loader,
        early_stopping=earlystop, teacher_model=teacher_model, T=2.0
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
