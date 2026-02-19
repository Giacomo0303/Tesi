import torch
from torch import nn
from src.utils.NAS_Utils import load_model
from src.Datasets.Cifar100 import Cifar100
import matplotlib.pyplot as plt
import numpy as np  # Necessario per gestire le posizioni delle barre

model_name = "vit_small_patch16_224"
num_classes = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    # --- 1. SETUP E CALCOLO GRADIENTI (INVARIATO) ---
    model = load_model(model_name, num_classes=num_classes,
                       path="D:\\Tesi\\src\\FineTuning\\vit_small_cifar100.pth").to(device)
    dataset = Cifar100(root_path="D:\\Tesi\\Data\\CIFAR100", img_size=224, batch_size=128, mean_std="imagenet",
                       model_name=model_name)
    search_loader = dataset.get_search_loader(n_per_classes=25)

    for p in model.parameters():
        if p.requires_grad:
            p.abs_grad = torch.zeros_like(p.data)

    model.train()
    scaler = torch.amp.GradScaler()

    print("Calcolo dei gradienti in corso...")
    for X, y in search_loader:
        X, y = X.to(device), y.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(X)
            batch_loss = loss_fn(logits, y)

        model.zero_grad()
        scaler.scale(batch_loss).backward()

        current_scale = scaler.get_scale()
        for p in model.parameters():
            if p.grad is not None:
                # Accumuliamo il gradiente assoluto reale (unscaled)
                p.abs_grad += abs(p.grad.data.detach()) / current_scale

    # Media finale dei gradienti
    for p in model.parameters():
        if p.grad is not None:
            p.abs_grad /= len(search_loader)

    # --- 2. AGGREGAZIONE DATI (GRADIENTI E PESI) ---
    block_grads_sum = {}
    block_weights_sum = {}  # NUOVO: Accumulatore per i pesi
    block_param_count = {}

    print("Aggregazione statistiche per blocco...")

    for name, p in model.named_parameters():
        if "blocks." in name and hasattr(p, "abs_grad"):
            parts = name.split('.')
            block_idx = int(parts[1])

            if block_idx not in block_grads_sum:
                block_grads_sum[block_idx] = 0.0
                block_weights_sum[block_idx] = 0.0
                block_param_count[block_idx] = 0

            # Somma Gradienti Assoluti
            block_grads_sum[block_idx] += p.abs_grad.sum().item()

            # NUOVO: Somma Pesi Assoluti (|w|)
            # Usiamo p.data perché contiene i pesi attuali
            block_weights_sum[block_idx] += p.data.abs().sum().item()

            block_param_count[block_idx] += p.abs_grad.numel()

    # --- 3. CALCOLO MEDIE ---
    blocks = sorted(block_grads_sum.keys())
    avg_grads = []
    avg_weights = []  # NUOVO

    for b in blocks:
        avg_g = block_grads_sum[b] / block_param_count[b]
        avg_w = block_weights_sum[b] / block_param_count[b]

        avg_grads.append(avg_g)
        avg_weights.append(avg_w)

        print(f"Blocco {b}: |Grad|={avg_g:.2e}, |Weight|={avg_w:.4f}")

    # --- 4. PLOTTING (DOPPIO ASSE Y) ---
    x = np.arange(len(blocks))  # Posizioni delle etichette
    width = 0.35  # Larghezza delle barre

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Barre Gradienti (Asse Sinistro - Blu)
    rects1 = ax1.bar(x - width / 2, avg_grads, width, label='Media |Gradiente|', color='royalblue', alpha=0.8,
                     edgecolor='black')

    # Setup Asse Sinistro
    ax1.set_xlabel('Indice del Blocco (Input -> Output)', fontsize=12)
    ax1.set_ylabel('Media |Gradiente|', fontsize=12, color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(blocks)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Creazione Secondo Asse Y
    ax2 = ax1.twinx()  # Condivide lo stesso asse X

    # Barre Pesi (Asse Destro - Arancione)
    rects2 = ax2.bar(x + width / 2, avg_weights, width, label='Media |Pesi|', color='darkorange', alpha=0.8,
                     edgecolor='black')

    # Setup Asse Destro
    ax2.set_ylabel('Media |Pesi|', fontsize=12, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    # Limiti assi (opzionale: per dare "respiro" alle barre)
    # ax1.set_ylim(0, max(avg_grads) * 1.2)
    # ax2.set_ylim(0, max(avg_weights) * 1.2)

    plt.title(f'Confronto Magnitudine Gradienti vs Pesi per Blocco\n{model_name}', fontsize=14)

    # Legenda Unificata
    # Raccoglie le maniglie (handles) e le etichette (labels) da entrambi gli assi
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=11)

    plt.tight_layout()
    plt.show()