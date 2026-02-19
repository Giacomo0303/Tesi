import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils.PruneUtils import head_alignment


# RELATIVE L2 WEIGHT CHANGE
def RWC(initial_weight: torch.Tensor, final_weight: torch.Tensor, eps=1e-7):
    if initial_weight.shape != final_weight.shape:
        return None
    return torch.norm(input=final_weight - initial_weight, p=2) / (torch.norm(input=initial_weight, p=2) + eps)


def analize_mlp(original_model, finetuned_model, pruning_report, pruned=False, save_path=None):
    block_indices = []
    pruning_percentages = []
    rwc_means = []

    kept_emb = pruning_report.Embedding["kept"]

    for i in range(len(original_model.blocks)):
        pruned_dims = len(pruning_report.blocks[i]["MLP"]["pruned"])
        kept_idxs = pruning_report.blocks[i]["MLP"]["kept"]
        total_dims = original_model.blocks[i].mlp.fc1.weight.shape[0]
        pruning_percentage = (pruned_dims / total_dims) * 100

        block_indices.append(i)
        pruning_percentages.append(pruning_percentage)

        rwcs = []

        num_dims = total_dims if pruned == False else total_dims - pruned_dims

        for j in range(num_dims):
            if pruned == False:
                original_neuron = torch.cat([
                    original_model.blocks[i].mlp.fc1.weight[j, :].flatten(),
                    original_model.blocks[i].mlp.fc1.bias[j:j + 1],
                    original_model.blocks[i].mlp.fc2.weight[:, j].flatten()
                ])
            else:
                original_neuron = torch.cat([
                    original_model.blocks[i].mlp.fc1.weight[kept_idxs[j], kept_emb].flatten(),
                    original_model.blocks[i].mlp.fc1.bias[kept_idxs[j]:kept_idxs[j] + 1],
                    original_model.blocks[i].mlp.fc2.weight[kept_emb, kept_idxs[j]].flatten()
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
    if save_path is not None:
        # dpi=300 garantisce un'alta risoluzione, ottima per le tesi o paper!
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def analize_qk(original_model, finetuned_model, pruning_report, pruned=False, save_path=None):
    block_indices = []
    pruning_percentages = []
    rwc_means = []

    kept_emb = pruning_report.Embedding["kept"]

    for i in range(len(original_model.blocks)):
        original_attn = head_alignment(original_model.blocks[i].attn)
        finetuned_attn = head_alignment(finetuned_model.blocks[i].attn)

        pruned_dims = len(pruning_report.blocks[i]["QK"]["pruned"])
        kept_idxs = pruning_report.blocks[i]["QK"]["kept"]
        kept_heads = pruning_report.blocks[i]["Heads"]["kept"]

        total_dims = original_attn.Q.weight.shape[1]
        pruning_percentage = (pruned_dims / total_dims) * 100

        block_indices.append(i)
        pruning_percentages.append(pruning_percentage)

        rwcs = []

        num_dims = total_dims if pruned == False else total_dims - pruned_dims

        for j in range(num_dims):
            if pruned == False:
                orig_Q_w = original_attn.Q.weight[:, j, :]
                orig_K_w = original_attn.K.weight[:, j, :]
                orig_Q_b = original_attn.Q.bias[:, j]
                orig_K_b = original_attn.K.bias[:, j]
            else:
                orig_Q_w = original_attn.Q.weight[kept_heads][:, kept_idxs[j], :][:, kept_emb]
                orig_K_w = original_attn.K.weight[kept_heads][:, kept_idxs[j], :][:, kept_emb]

                orig_Q_b = original_attn.Q.bias[kept_heads][:, kept_idxs[j]]
                orig_K_b = original_attn.K.bias[kept_heads][:, kept_idxs[j]]

            original_qk = torch.cat([
                orig_Q_w.flatten(),
                orig_Q_b.flatten(),
                orig_K_w.flatten(),
                orig_K_b.flatten()
            ])

            tuned_qk = torch.cat([
                finetuned_attn.Q.weight[:, j, :].flatten(),
                finetuned_attn.Q.bias[:, j].flatten(),
                finetuned_attn.K.weight[:, j, :].flatten(),
                finetuned_attn.K.bias[:, j].flatten()
            ])

            rwc = RWC(original_qk, tuned_qk)
            rwcs.append(rwc.item())

        rwc_avg = sum(rwcs) / len(rwcs) if len(rwcs) > 0 else 0.0
        rwc_means.append(rwc_avg)

    # --- PLOTTING ---
    print("Generazione Grafico...")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(block_indices))
    width = 0.35

    # Barre Pruning (Asse SX - Rosso)
    rects1 = ax1.bar(x - width / 2, pruning_percentages, width, label='% QK Pruned', color='#d62728', alpha=0.7)
    ax1.set_xlabel('Transformer Block Index', fontsize=12)
    ax1.set_ylabel('% QK Dimensions Pruned', color='#d62728', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(0, 110)

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
    if save_path is not None:
        # dpi=300 garantisce un'alta risoluzione, ottima per le tesi o paper!
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def analize_vproj(original_model, finetuned_model, pruning_report, pruned=False, save_path=None):
    block_indices = []
    pruning_percentages = []
    rwc_means = []

    kept_embed = pruning_report.Embedding["kept"]

    for i in range(len(original_model.blocks)):
        original_attn = head_alignment(original_model.blocks[i].attn)
        finetuned_attn = head_alignment(finetuned_model.blocks[i].attn)

        pruned_dims = len(pruning_report.blocks[i]["VProj"]["pruned"])
        kept_idxs = pruning_report.blocks[i]["VProj"]["kept"]
        kept_heads = pruning_report.blocks[i]["Heads"]["kept"]

        total_dims = original_attn.V.weight.shape[1]
        pruning_percentage = (pruned_dims / total_dims) * 100

        block_indices.append(i)
        pruning_percentages.append(pruning_percentage)

        n_dims = total_dims if pruned == False else total_dims - pruned_dims

        rwcs = []
        for j in range(n_dims):

            if pruned == False:
                original_vproj = torch.cat([
                    original_attn.V.weight[:, j, :].flatten(),
                    original_attn.V.bias[:, j],
                    original_attn.proj.weight[:, :, j].flatten()
                ])
            else:
                original_vproj = torch.cat([
                    original_attn.V.weight[kept_heads][:, kept_idxs[j], :][:, kept_embed].flatten(),
                    original_attn.V.bias[kept_heads][:, kept_idxs[j]],
                    original_attn.proj.weight[kept_embed][:, kept_heads, :][:, :, kept_idxs[j]].flatten()
                ])

            tuned_vproj = torch.cat([
                finetuned_attn.V.weight[:, j, :].flatten(),
                finetuned_attn.V.bias[:, j],
                finetuned_attn.proj.weight[:, :, j].flatten(),
            ])

            rwc = RWC(original_vproj, tuned_vproj)
            rwcs.append(rwc.item())

        rwc_avg = sum(rwcs) / len(rwcs) if len(rwcs) > 0 else 0.0
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
    if save_path is not None:
        # dpi=300 garantisce un'alta risoluzione, ottima per le tesi o paper!
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def analize_head(original_model, finetuned_model, pruning_report, pruned=False, save_path=None):
    block_indices = []
    pruning_percentages = []
    rwc_means = []

    kept_embed = pruning_report.Embedding["kept"]

    for i in range(len(original_model.blocks)):
        original_attn = head_alignment(original_model.blocks[i].attn)
        finetuned_attn = head_alignment(finetuned_model.blocks[i].attn)

        # Corretto il refuso "Pr" e aggiunti gli indici necessari per QK e VProj
        pruned_dims = len(pruning_report.blocks[i]["Heads"]["pruned"])
        kept_heads = pruning_report.blocks[i]["Heads"]["kept"]
        kept_qk = pruning_report.blocks[i]["QK"]["kept"]
        kept_v = pruning_report.blocks[i]["VProj"]["kept"]

        total_dims = original_attn.Q.weight.shape[0]
        pruning_percentage = (pruned_dims / total_dims) * 100

        block_indices.append(i)
        pruning_percentages.append(pruning_percentage)

        num_dims = total_dims if pruned == False else len(kept_heads)

        rwcs = []
        for j in range(num_dims):
            if pruned == False:
                original_head = torch.cat([
                    original_attn.Q.weight[j, :, :].flatten(),
                    original_attn.Q.bias[j, :].flatten(),
                    original_attn.K.weight[j, :, :].flatten(),
                    original_attn.K.bias[j, :].flatten(),
                    original_attn.V.weight[j, :, :].flatten(),
                    original_attn.V.bias[j, :].flatten(),
                    original_attn.proj.weight[:, j, :].flatten()
                ])
            else:
                # Slicing a 3 livelli:
                # 1. Seleziono la testa singola `kept_heads[j]` (abbassando la dim a 2D)
                # 2. Filtro i neuroni sopravvissuti per quella testa (QK o V)
                # 3. Filtro l'embedding in ingresso
                orig_Q_w = original_attn.Q.weight[kept_heads[j]][kept_qk, :][:, kept_embed]
                orig_Q_b = original_attn.Q.bias[kept_heads[j]][kept_qk]

                orig_K_w = original_attn.K.weight[kept_heads[j]][kept_qk, :][:, kept_embed]
                orig_K_b = original_attn.K.bias[kept_heads[j]][kept_qk]

                orig_V_w = original_attn.V.weight[kept_heads[j]][kept_v, :][:, kept_embed]
                orig_V_b = original_attn.V.bias[kept_heads[j]][kept_v]

                # Per la proiezione l'ordine è [emb_out, n_heads, v_dim]
                # 1. Filtro l'embedding in uscita
                # 2. Seleziono la testa singola (abbassando la dim a 2D)
                # 3. Filtro i neuroni VProj sopravvissuti in ingresso
                orig_proj_w = original_attn.proj.weight[kept_embed][:, kept_heads[j], :][:, kept_v]

                original_head = torch.cat([
                    orig_Q_w.flatten(),
                    orig_Q_b.flatten(),
                    orig_K_w.flatten(),
                    orig_K_b.flatten(),
                    orig_V_w.flatten(),
                    orig_V_b.flatten(),
                    orig_proj_w.flatten()
                ])

            # Il modello tuned è già completamente compresso in tutte le sue dimensioni
            tuned_head = torch.cat([
                finetuned_attn.Q.weight[j, :, :].flatten(),
                finetuned_attn.Q.bias[j, :].flatten(),
                finetuned_attn.K.weight[j, :, :].flatten(),
                finetuned_attn.K.bias[j, :].flatten(),
                finetuned_attn.V.weight[j, :, :].flatten(),
                finetuned_attn.V.bias[j, :].flatten(),
                finetuned_attn.proj.weight[:, j, :].flatten()
            ])

            rwc = RWC(original_head, tuned_head)
            rwcs.append(rwc.item())

        rwc_avg = sum(rwcs) / len(rwcs) if len(rwcs) > 0 else 0.0
        rwc_means.append(rwc_avg)

    # --- PLOTTING ---
    print("Generazione Grafico...")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(block_indices))
    width = 0.35

    # Barre Pruning (Rosso)
    rects1 = ax1.bar(x - width / 2, pruning_percentages, width, label='% Heads Pruned', color='#d62728', alpha=0.7)
    ax1.set_xlabel('Transformer Block Index', fontsize=12)
    ax1.set_ylabel('% Heads Pruned', color='#d62728', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(0, 110)

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

    if save_path is not None:
        # dpi=300 garantisce un'alta risoluzione, ottima per le tesi o paper!
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
