import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tkinter as tk
from tkinter import filedialog
import os


def load_json_from_disk():
    # Inizializza tkinter e nascondi la finestra principale
    root = tk.Tk()
    root.withdraw()

    # Apri la finestra di dialogo per selezionare il file
    file_path = filedialog.askopenfilename(
        title="Seleziona il tuo Pruning Report (es. pruning_report.json)",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )

    if not file_path:
        print("Nessun file selezionato. Chiusura script.")
        exit()

    print(f"File caricato: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data, os.path.dirname(file_path)


def plot_individual_barcodes(data, save_dir):
    blocks = data['blocks']
    n_blocks = len(blocks)

    # Palette colori: 0 = Potato (Rosso), 1 = Mantenuto (Nero)
    cmap_red_black = ListedColormap(['#d62728', '#202020'])

    print("\nGenerazione grafici 'Codice a Barre' separati...")

    # ==========================================
    # 1. EMBEDDING (Immagine singola)
    # ==========================================
    emb_kept = data['Embedding']['kept']
    emb_pruned = data['Embedding']['pruned']
    emb_orig_dim = len(emb_kept) + len(emb_pruned)

    emb_matrix = np.zeros((1, emb_orig_dim))
    emb_matrix[0, emb_kept] = 1  # Imposta a 1 (Nero) i mantenuti

    plt.figure(figsize=(15, 2))
    plt.imshow(emb_matrix, cmap=cmap_red_black, aspect='auto', interpolation='nearest')
    plt.title(f"Embedding (Dim: {emb_orig_dim}) | {len(emb_pruned)} Potate (Rosso)", fontsize=16, pad=15)
    plt.yticks([])
    plt.xlabel("Indice Dimensione Embedding", fontsize=12)
    plt.tight_layout()

    path_emb = os.path.join(save_dir, "barcode_Embedding.png")
    plt.savefig(path_emb, dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {path_emb}")
    plt.close()

    # ==========================================
    # 2. COMPONENTI DEI BLOCCHI (Heads, QK, VProj, MLP)
    # ==========================================
    block_components = ['Heads', 'QK', 'VProj', 'MLP']

    for comp in block_components:
        # Calcolo dimensione originale in base al primo blocco
        comp_orig_dim = len(blocks[0][comp]['kept']) + len(blocks[0][comp]['pruned'])

        matrix = np.zeros((n_blocks, comp_orig_dim))
        total_pruned_comp = 0

        for i, block in enumerate(blocks):
            kept_indices = block[comp]['kept']
            matrix[i, kept_indices] = 1  # Imposta a 1 i mantenuti
            total_pruned_comp += block[comp]['num_pruned']

        plt.figure(figsize=(15, 6))
        plt.imshow(matrix, cmap=cmap_red_black, aspect='auto', interpolation='nearest')
        plt.title(f"{comp} (Dim: {comp_orig_dim}) | Totale tagliate: {total_pruned_comp} (Rosso)",
                  fontsize=16, pad=15)
        plt.yticks(range(n_blocks), [f"B{i}" for i in range(n_blocks)])
        plt.ylabel("Transformer Block", fontsize=12)
        plt.xlabel(f"Indice {comp}", fontsize=12)

        # Linee di separazione tra i blocchi per leggibilità
        for i in range(n_blocks - 1):
            plt.axhline(y=i + 0.5, color='white', linewidth=1.5)

        plt.tight_layout()
        path_comp = os.path.join(save_dir, f"barcode_{comp}.png")
        plt.savefig(path_comp, dpi=300, bbox_inches='tight')
        print(f"✅ Salvato: {path_comp}")
        plt.close()


def plot_all_percentages(data, save_dir):
    blocks = data['blocks']
    n_blocks = len(blocks)

    print("\nGenerazione grafico a Percentuali...")

    components = {
        'Heads': '#1f77b4',  # Blu
        'QK': '#ff7f0e',  # Arancione
        'VProj': '#2ca02c',  # Verde
        'MLP': '#d62728'  # Rosso
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Percentuale di Riduzione per Dimensione e Blocco (%)", fontsize=18, fontweight='bold', y=0.98)
    axes = axes.flatten()

    for idx, (comp, color) in enumerate(components.items()):
        ax = axes[idx]
        pct_list = []

        for b in blocks:
            num_pruned = b[comp]['num_pruned']
            total = num_pruned + len(b[comp]['kept'])
            pct = (num_pruned / total * 100) if total > 0 else 0
            pct_list.append(pct)

        ax.plot(range(n_blocks), pct_list, marker='o', color=color, linewidth=2.5)
        ax.fill_between(range(n_blocks), pct_list, color=color, alpha=0.15)

        ax.set_title(f'Pruning {comp}', fontsize=14)
        ax.set_xlabel('Transformer Block', fontsize=12)
        ax.set_ylabel('% Dimensioni Eliminate', fontsize=12)
        ax.set_xticks(range(n_blocks))
        ax.set_xticklabels([f"B{i}" for i in range(n_blocks)])
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.4, linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "percentages_pruned_grid.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Seleziona il file JSON dalla finestra di dialogo...")
    json_data, directory = load_json_from_disk()

    plot_individual_barcodes(json_data, directory)
    plot_all_percentages(json_data, directory)

    print("\nFinito! I file PNG sono stati salvati nella stessa cartella del JSON selezionato.")