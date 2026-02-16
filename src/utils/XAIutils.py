import torch
import matplotlib.pyplot as plt
import numpy as np

from src.utils.PruneUtils import head_alignment


# RELATIVE L2 WEIGHT CHANGE
def RWC(initial_weight: torch.Tensor, final_weight: torch.Tensor, eps=1e-7):
    if initial_weight.shape != final_weight.shape:
        return None
    return torch.norm(input=final_weight - initial_weight, p=2) / (torch.norm(input=initial_weight, p=2) + eps)


def analize_mlp(original_model, finetuned_model, pruning_report):
    block_indices = []
    pruning_percentages = []
    rwc_means = []

    for i in range(len(original_model.blocks)):
        pruned_dims = pruning_report["blocks"][i]["mlp_pruned_dims"]
        total_dims = original_model.blocks[i].mlp.fc1.weight.shape[0]
        pruning_percentage = (pruned_dims / total_dims) * 100

        block_indices.append(i)
        pruning_percentages.append(pruning_percentage)

        rwcs = []
        for j in range(total_dims):
            original_neuron = torch.cat([
                original_model.blocks[i].mlp.fc1.weight[j, :].flatten(),
                original_model.blocks[i].mlp.fc1.bias[j:j + 1],
                original_model.blocks[i].mlp.fc2.weight[:, j].flatten()
            ])

            tuned_neuron = torch.cat([
                finetuned_model.blocks[i].mlp.fc1.weight[j, :].flatten(),
                finetuned_model.blocks[i].mlp.fc1.bias[j:j + 1],
                finetuned_model.blocks[i].mlp.fc2.weight[:, j].flatten()
            ])

            rwc = RWC(original_neuron, tuned_neuron)
            rwcs.append(rwc.item())

        rwc_avg = sum(rwcs) / len(rwcs)
        rwc_means.append(rwc_avg)

    print("Generazione Grafico...")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(block_indices))
    width = 0.35

    # Barre Pruning
    rects1 = ax1.bar(x - width / 2, pruning_percentages, width, label='% MLP Pruned', color='#d62728', alpha=0.7)
    ax1.set_xlabel('Transformer Block Index', fontsize=12)
    ax1.set_ylabel('% Neuroni Eliminati', color='#d62728', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(0, 110)  # Aumentato leggermente per far spazio alle label

    # Barre RWC
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, rwc_means, width, label='Avg RWC', color='#1f77b4', alpha=0.7)
    ax2.set_ylabel('Average Relative Weight Change (RWC)', color='#1f77b4', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    # --- AGGIUNTA ETICHETTE NUMERICHE ---
    # fmt='%.1f' mette 1 decimale per le percentuali
    ax1.bar_label(rects1, padding=3, fmt='%.1f', fontsize=9, color='#d62728')
    # fmt='%.4f' mette 4 decimali per RWC
    ax2.bar_label(rects2, padding=3, fmt='%.4f', fontsize=9, color='#1f77b4')

    plt.title('Confronto per Blocco: Pruning (Hessiano) vs Cambiamento Pesi (RWC)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(block_indices)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.show()


def analize_qk(original_model, finetuned_model, pruning_report):
    block_indices = []
    pruning_percentages = []
    rwc_means = []

    for i in range(len(original_model.blocks)):
        original_attn = head_alignment(original_model.blocks[i].attn)
        finetuned_attn = head_alignment(finetuned_model.blocks[i].attn)

        pruned_dims = pruning_report["blocks"][i]["qk_pruned_dims"]
        total_dims = original_attn.Q.weight.shape[1]
        pruning_percentage = (pruned_dims / total_dims) * 100

        block_indices.append(i)
        pruning_percentages.append(pruning_percentage)

        rwcs = []
        for j in range(total_dims):
            original_qk = torch.cat([
                original_attn.Q.weight[:, j, :].flatten(),
                original_attn.Q.bias[:, j],
                original_attn.K.weight[:, j, :].flatten(),
                original_attn.K.bias[:, j]
            ])

            tuned_qk = torch.cat([
                finetuned_attn.Q.weight[:, j, :].flatten(),
                finetuned_attn.Q.bias[:, j],
                finetuned_attn.K.weight[:, j, :].flatten(),
                finetuned_attn.K.bias[:, j]
            ])

            rwc = RWC(original_qk, tuned_qk)
            rwcs.append(rwc.item())

        rwc_avg = sum(rwcs) / len(rwcs)
        rwc_means.append(rwc_avg)

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(block_indices))
    width = 0.35

    # Barre Pruning (Asse SX - Rosso)
    rects1 = ax1.bar(x - width / 2, pruning_percentages, width, label='% QK Pruned', color='#d62728', alpha=0.7)
    ax1.set_xlabel('Transformer Block Index', fontsize=12)
    ax1.set_ylabel('% QK Dimensions Pruned', color='#d62728', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(0, 100)

    # Barre RWC (Asse DX - Blu)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, rwc_means, width, label='Avg RWC (QK)', color='#1f77b4', alpha=0.7)
    ax2.set_ylabel('Avg RWC (QK)', color='#1f77b4', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    # Etichette valori
    ax1.bar_label(rects1, padding=3, fmt='%.1f', fontsize=9, color='#d62728')
    ax2.bar_label(rects2, padding=3, fmt='%.4f', fontsize=9, color='#1f77b4')

    plt.title('Comparison per Block: QK Pruning vs Weight Change (RWC)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(block_indices)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.show()


def analize_vproj(original_model, finetuned_model, pruning_report):
    block_indices = []
    pruning_percentages = []
    rwc_means = []

    for i in range(len(original_model.blocks)):
        original_attn = head_alignment(original_model.blocks[i].attn)
        finetuned_attn = head_alignment(finetuned_model.blocks[i].attn)

        pruned_dims = pruning_report["blocks"][i]["v_proj_pruned_dims"]
        total_dims = original_attn.V.weight.shape[1]
        pruning_percentage = (pruned_dims / total_dims) * 100

        block_indices.append(i)
        pruning_percentages.append(pruning_percentage)

        rwcs = []
        for j in range(total_dims):
            original_vproj = torch.cat([
                original_attn.V.weight[:, j, :].flatten(),
                original_attn.V.bias[:, j],
                original_attn.proj.weight[:, :, j].flatten()
            ])

            tuned_vproj = torch.cat([
                finetuned_attn.V.weight[:, j, :].flatten(),
                finetuned_attn.V.bias[:, j],
                finetuned_attn.proj.weight[:, :, j].flatten(),
            ])

            rwc = RWC(original_vproj, tuned_vproj)
            rwcs.append(rwc.item())

        rwc_avg = sum(rwcs) / len(rwcs)
        rwc_means.append(rwc_avg)

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(block_indices))
    width = 0.35

    # Barre Pruning (Asse SX - Rosso) - LABELS CORRETTE
    rects1 = ax1.bar(x - width / 2, pruning_percentages, width, label='% V-Proj Pruned', color='#d62728', alpha=0.7)
    ax1.set_xlabel('Transformer Block Index', fontsize=12)
    ax1.set_ylabel('% V-Proj Dimensions Pruned', color='#d62728', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(0, 100)

    # Barre RWC (Asse DX - Blu) - LABELS CORRETTE
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, rwc_means, width, label='Avg RWC (V-Proj)', color='#1f77b4', alpha=0.7)
    ax2.set_ylabel('Avg RWC (V-Proj)', color='#1f77b4', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    # Etichette valori
    ax1.bar_label(rects1, padding=3, fmt='%.1f', fontsize=9, color='#d62728')
    ax2.bar_label(rects2, padding=3, fmt='%.4f', fontsize=9, color='#1f77b4')

    plt.title('Comparison per Block: V-Proj Pruning vs Weight Change (RWC)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(block_indices)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.show()


def analize_head(original_model, finetuned_model, pruning_report):
    block_indices = []
    pruning_percentages = []
    rwc_means = []

    for i in range(len(original_model.blocks)):
        original_attn = head_alignment(original_model.blocks[i].attn)
        finetuned_attn = head_alignment(finetuned_model.blocks[i].attn)

        pruned_dims = pruning_report["blocks"][i]["head_pruned_idx"]
        total_dims = original_attn.Q.weight.shape[0]
        pruning_percentage = (pruned_dims / total_dims) * 100

        block_indices.append(i)
        pruning_percentages.append(pruning_percentage)

        rwcs = []
        for j in range(total_dims):
            original_head = torch.cat([
                original_attn.Q.weight[j, :, :].flatten(),
                original_attn.Q.bias[j, :],
                original_attn.K.weight[j, :, :].flatten(),
                original_attn.K.bias[j, :],
                original_attn.V.weight[j, :, :].flatten(),
                original_attn.V.bias[j, :],
                original_attn.proj.weight[:, j, :].flatten()
            ])

            tuned_head = torch.cat([
                finetuned_attn.Q.weight[j, :, :].flatten(),
                finetuned_attn.Q.bias[j, :],
                finetuned_attn.K.weight[j, :, :].flatten(),
                finetuned_attn.K.bias[j, :],
                finetuned_attn.V.weight[j, :, :].flatten(),
                finetuned_attn.V.bias[j, :],
                finetuned_attn.proj.weight[:, j, :].flatten()
            ])

            rwc = RWC(original_head, tuned_head)
            rwcs.append(rwc.item())

        rwc_avg = sum(rwcs) / len(rwcs)
        rwc_means.append(rwc_avg)

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(block_indices))
    width = 0.35

    # Barre Pruning (Rosso)
    rects1 = ax1.bar(x - width / 2, pruning_percentages, width, label='% Heads Pruned', color='#d62728', alpha=0.7)
    ax1.set_xlabel('Transformer Block Index', fontsize=12)
    ax1.set_ylabel('% Heads Pruned', color='#d62728', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(0, 105)  # Max 100%

    # Barre RWC (Blu)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, rwc_means, width, label='Avg RWC (Heads)', color='#1f77b4', alpha=0.7)
    ax2.set_ylabel('Avg RWC (Heads)', color='#1f77b4', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    # Etichette
    ax1.bar_label(rects1, padding=3, fmt='%.0f', fontsize=9, color='#d62728')
    ax2.bar_label(rects2, padding=3, fmt='%.4f', fontsize=9, color='#1f77b4')

    plt.title('Comparison per Block: Head Pruning vs Weight Change (RWC)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(block_indices)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.show()
