import timm
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.NAS_Utils import load_model
import json
from src.utils.XAIutils import RWC

model_name = "vit_small_patch16_224"
save_path = "D:\\Tesi\\src\\FineTuning"
report_path = "D:\\Tesi\\src\\NAS\\ResultsFinal\\Distil\\pruning_report_89.json"
num_classes = 100
img_size = 224
batch_size = 128
N_epochs = 30

original_model = timm.create_model(model_name=model_name, pretrained=True)

finetuned_model = load_model(model_name="vit_small_patch16_224", num_classes=num_classes,
                             path="D:\\Tesi\\src\\FineTuning\\best_model.pth")

pruning_report = json.load(open(report_path))

block_indices = []
pruning_percentages = []
rwc_means = []

for i in range(len(original_model.blocks)):
    pruned_dims = pruning_report["blocks"][i]["mlp_pruned_dims"]
    total_dims = original_model.blocks[i].mlp.fc1.weight.shape[0]
    pruning_percentage = (pruned_dims / total_dims) * 100

    block_indices.append(i)
    pruning_percentages.append(pruning_percentage)

    hidden_dim = original_model.blocks[i].mlp.fc1.weight.shape[0]
    rwcs = []
    for j in range(hidden_dim):
        original_neuron = torch.cat([
            original_model.blocks[i].mlp.fc1.weight[j, :].flatten(),
            original_model.blocks[i].mlp.fc1.bias[j:j + 1],
            original_model.blocks[i].mlp.fc2.weight[:, j].flatten()
        ])

        pruned_neuron = torch.cat([
            finetuned_model.blocks[i].mlp.fc1.weight[j, :].flatten(),
            finetuned_model.blocks[i].mlp.fc1.bias[j:j + 1],
            finetuned_model.blocks[i].mlp.fc2.weight[:, j].flatten()
        ])

        rwc = RWC(pruned_neuron, original_neuron)
        rwcs.append(rwc.item())

    rwc_avg = sum(rwcs) / len(rwcs)
    rwc_means.append(rwc_avg)

print("Generazione Grafico...")

fig, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(block_indices))
width = 0.35

# Barre Pruning
rects1 = ax1.bar(x - width/2, pruning_percentages, width, label='% MLP Pruned', color='#d62728', alpha=0.7)
ax1.set_xlabel('Transformer Block Index', fontsize=12)
ax1.set_ylabel('% Neuroni Eliminati', color='#d62728', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#d62728')
ax1.set_ylim(0, 110) # Aumentato leggermente per far spazio alle label

# Barre RWC
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, rwc_means, width, label='Avg RWC', color='#1f77b4', alpha=0.7)
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