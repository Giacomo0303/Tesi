import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog, messagebox
import os

# Stile globale per articoli scientifici
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'figure.autolayout': True,
    'font.family': 'serif'
})


def parse_nas_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Estrazione Baseline
    baseline_match = re.search(r"Baseline Model:.*?Params: ([\d.]+)M.*?Accuracy: ([\d.]+)%", content, re.S)
    baseline_params = float(baseline_match.group(1)) if baseline_match else None
    baseline_acc = float(baseline_match.group(2)) if baseline_match else None

    iterations = content.split("ITERAZIONE")[1:]
    data = []

    for i, iter_text in enumerate(iterations):
        iter_num = i + 1

        # 1. Numero di nodi esplorati
        total_iters_match = re.search(r"Iterazioni totali: (\d+)", iter_text)
        if total_iters_match:
            nodes_explored = int(total_iters_match.group(1))
            is_search = True
        else:
            steps = re.findall(r"Step (\d+)/\d+", iter_text)
            nodes_explored = int(steps[-1]) if steps else 0
            is_search = False

        # 2. Accuracy Pre/Post
        post_taglio_acc_match = re.search(r"REPORT POST-TAGLIO:.*?Accuracy: ([\d.]+)%", iter_text, re.S)
        acc_pre_ft = float(post_taglio_acc_match.group(1)) if post_taglio_acc_match else None

        recuperata_acc_match = re.search(r"Accuracy Recuperata: ([\d.]+)%", iter_text)
        acc_post_ft = float(recuperata_acc_match.group(1)) if recuperata_acc_match else None

        # 3. Parametri
        params_match = re.search(r"REPORT POST-TAGLIO:.*?Params: ([\d.]+)M", iter_text, re.S)
        params = float(params_match.group(1)) if params_match else None

        data.append({
            'Iteration': iter_num,
            'Nodes': nodes_explored,
            'Acc_Pre_FT': acc_pre_ft,
            'Acc_Post_FT': acc_post_ft,
            'Params_M': params,
            'Is_Search': is_search
        })

    df = pd.DataFrame(data)

    # FORZA l'interpolazione per evitare linee spezzate
    cols_to_interpolate = ['Acc_Pre_FT', 'Acc_Post_FT', 'Params_M']
    df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear')

    return df, baseline_acc, baseline_params


def create_dashboard(df, b_acc, b_params, filename):
    has_search = df['Is_Search'].any()

    # Palette colori
    color_pre = '#d95f02'
    color_post = '#1b9e77'
    color_base = '#7570b3'
    color_bars = '#4c72b0'
    color_avg = '#e41a1c'

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    sns.set_style("white")

    # --- 1. Grafico Accuracy (Pre vs Post Fine-Tuning) ---
    ax1 = fig.add_subplot(gs[0, :])

    ax1.plot(df['Iteration'], df['Acc_Pre_FT'], marker='o', linestyle='-', linewidth=2.5,
             markersize=7, color=color_pre, label='Post-Pruning Accuracy')
    ax1.plot(df['Iteration'], df['Acc_Post_FT'], marker='s', linestyle='-', linewidth=2.5,
             markersize=7, color=color_post, label='Fine-Tuned Accuracy')

    ax1.fill_between(df['Iteration'], df['Acc_Pre_FT'], df['Acc_Post_FT'],
                     color=color_post, alpha=0.15, label='Accuracy Recovered')

    if b_acc:
        ax1.axhline(y=b_acc, color=color_base, linestyle=':', linewidth=2, label=f'Baseline Acc. ({b_acc}%)')

    ax1.set_title("Model Accuracy Evolution Over NAS Iterations", fontweight='bold', pad=10)
    ax1.set_ylabel("Accuracy (%)", fontweight='bold')
    ax1.set_xlabel("Iteration", fontweight='bold')
    ax1.set_xticks(df['Iteration'])
    ax1.legend(loc='lower right', framealpha=1)

    # --- 2. Grafico Nodi Esplorati ---
    ax2 = fig.add_subplot(gs[1, 0])
    if has_search:
        sns.barplot(x='Iteration', y='Nodes', data=df, ax=ax2, color=color_bars, edgecolor='black', alpha=0.85)

        avg_nodes = df['Nodes'].mean()
        ax2.axhline(y=avg_nodes, color=color_avg, linestyle='--', linewidth=2.5,
                    label=f'Average Nodes: {avg_nodes:.1f}')

        ax2.set_title("Explored Nodes per Search Phase", fontweight='bold')
        ax2.set_ylabel("Number of Nodes", fontweight='bold')
        ax2.set_xlabel("Iteration", fontweight='bold')

        for ind, label in enumerate(ax2.get_xticklabels()):
            if ind % 2 == 1:
                label.set_visible(False)

        ax2.legend(loc='upper right', framealpha=1)
    else:
        ax2.text(0.5, 0.5, "Static Sampling\n(No search iterations logged)",
                 ha='center', va='center', fontsize=12, color='gray', style='italic')
        ax2.set_title("Explored Nodes (N/A)", fontweight='bold')
        ax2.axis('off')

    # --- 3. Riduzione Parametri ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df['Iteration'], df['Params_M'], marker='D', linestyle='-', linewidth=2.5,
             color='#2b8cbe', markerfacecolor='white', markeredgewidth=1.5, label='Model Size')

    ax3.fill_between(df['Iteration'], df['Params_M'], color='#2b8cbe', alpha=0.1)

    if b_params:
        ax3.axhline(y=b_params, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'Baseline ({b_params}M)')

    ax3.set_title("Model Complexity Reduction", fontweight='bold')
    ax3.set_ylabel("Parameters (Millions)", fontweight='bold')
    ax3.set_xlabel("Iteration", fontweight='bold')
    ax3.set_xticks(df['Iteration'])
    ax3.legend(loc='upper right', framealpha=1)

    # Pulizia visiva
    for ax in [ax1, ax2, ax3]:
        if not ax.axis() == (0.0, 1.0, 0.0, 1.0):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # --- Richiesta di salvataggio all'utente ---
    root = Tk()
    root.withdraw()
    # Mantiene la finestra in primo piano
    root.attributes('-topmost', True)

    salva = messagebox.askyesno("Salvataggio", "Vuoi salvare il grafico in formato PDF?")
    root.destroy()

    if salva:
        output_pdf = "nas_analysis_plot.pdf"
        plt.savefig(output_pdf, bbox_inches='tight')
        print(f"Grafico salvato con successo: {output_pdf}")
    else:
        print("Salvataggio annullato.")

    plt.show()


def main():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Seleziona il file di log NAS",
                                           filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    root.destroy()

    if not file_path:
        print("Nessun file selezionato.")
        return

    df, b_acc, b_params = parse_nas_log(file_path)

    if df.empty:
        print("Impossibile trovare dati nel file selezionato.")
        return

    print(f"\n--- Analisi completata per {os.path.basename(file_path)} ---")
    create_dashboard(df, b_acc, b_params, file_path)


if __name__ == "__main__":
    main()