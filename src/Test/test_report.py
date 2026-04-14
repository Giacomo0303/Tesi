import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog, messagebox
import os

# Stile globale (ho rimosso 'figure.autolayout': True per gestire a mano i margini)
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'axes.grid': False,  # Lo gestiamo manualmente asse per asse giù
    'font.family': 'serif'
})


def parse_nas_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    clean_content = re.sub(r'\'', '', content)

    baseline_match = re.search(r"Baseline Model:.*?Params:\s*([\d.]+)M.*?Accuracy:\s*([\d.]+)", clean_content,
                               re.IGNORECASE | re.S)
    baseline_params = float(baseline_match.group(1)) if baseline_match else None
    baseline_acc = float(baseline_match.group(2)) if baseline_match else None

    data = []
    pattern = r"ITERAZIONE\s+(\d+)/\d+(.*?)(?=\s+ITERAZIONE\s+\d+/\d+|\s+VALUTAZIONE FINALE|$)"
    iteration_blocks = re.finditer(pattern, clean_content, re.IGNORECASE | re.S)

    for match in iteration_blocks:
        iter_num = int(match.group(1))
        iter_text = match.group(2)

        total_iters_match = re.search(r"Iterazioni totali:\s*(\d+)", iter_text, re.IGNORECASE)
        if total_iters_match:
            nodes_explored = int(total_iters_match.group(1))
            is_search = True
        else:
            steps = re.findall(r"Step (\d+)/\d+", iter_text, re.IGNORECASE)
            nodes_explored = int(steps[-1]) if steps else 0
            is_search = False

        post_taglio_acc_match = re.search(r"REPORT POST-TAGLIO.*?Accuracy:\s*([\d.]+)", iter_text, re.IGNORECASE | re.S)
        acc_pre_ft = float(post_taglio_acc_match.group(1)) if post_taglio_acc_match else None

        recuperata_acc_match = re.search(r"Accuracy Recuperata:\s*([\d.]+)", iter_text, re.IGNORECASE | re.S)
        acc_post_ft = float(recuperata_acc_match.group(1)) if recuperata_acc_match else None

        params_match = re.search(r"REPORT POST-TAGLIO.*?Params:\s*([\d.]+)", iter_text, re.IGNORECASE | re.S)
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

    if not df.empty:
        cols_to_numeric = ['Nodes', 'Acc_Pre_FT', 'Acc_Post_FT', 'Params_M']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        cols_to_interpolate = ['Acc_Pre_FT', 'Acc_Post_FT', 'Params_M']
        df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear', limit_direction='both')

    return df, baseline_acc, baseline_params


def create_dashboard(df, b_acc, b_params, root):
    has_search = df['Is_Search'].any()

    color_pre = '#d95f02'
    color_post = '#1b9e77'
    color_base = '#7570b3'
    color_bars = '#4c72b0'
    color_avg = '#e41a1c'

    # 1. Modificato figsize per evitare l'effetto "stirato" (più largo, meno alto)
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.45)
    sns.set_style("white")

    # --- 1. Grafico Accuracy ---
    ax1 = fig.add_subplot(gs[0, 0])

    if not df['Acc_Pre_FT'].isna().all():
        ax1.plot(df['Iteration'], df['Acc_Pre_FT'], marker='o', linestyle='-', linewidth=2.5,
                 markersize=7, color=color_pre, label='Post-Pruning Accuracy')

    if not df['Acc_Post_FT'].isna().all():
        ax1.plot(df['Iteration'], df['Acc_Post_FT'], marker='s', linestyle='-', linewidth=2.5,
                 markersize=7, color=color_post, label='Fine-Tuned Accuracy')

        ax1.fill_between(df['Iteration'], df['Acc_Pre_FT'], df['Acc_Post_FT'],
                         color=color_post, alpha=0.15, label='Accuracy Recovered')

    if b_acc:
        ax1.axhline(y=b_acc, color=color_base, linestyle=':', linewidth=2, label=f'Baseline Acc. ({b_acc}%)')

    ax1.set_ylabel("Accuracy (%)", fontweight='bold')
    ax1.set_xlabel("Iteration", fontweight='bold')
    ax1.set_xticks(df['Iteration'])

    # Modificata la posizione della legenda (1.02 invece di 0.96)
    ax1.legend(loc='center left', framealpha=1, bbox_to_anchor=(1.02, 0.5))

    # 2. Aggiunte solo le linee orizzontali
    ax1.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax1.xaxis.grid(False)

    # --- 2. Grafico Nodi Esplorati ---
    ax2 = fig.add_subplot(gs[1, 0])
    if has_search and not df['Nodes'].isna().all():
        ax2.bar(df['Iteration'], df['Nodes'], color=color_bars, edgecolor='black', alpha=0.85)
        avg_nodes = df['Nodes'].mean()
        ax2.axhline(y=avg_nodes, color=color_avg, linestyle='--', linewidth=2.5,
                    label=f'Average Nodes: {avg_nodes:.1f}')
        ax2.set_ylabel("Number of Nodes", fontweight='bold')
        ax2.set_xlabel("Iteration", fontweight='bold')
        ax2.set_xticks(df['Iteration'])
        ax2.legend(loc='center left', framealpha=1, bbox_to_anchor=(1.02, 0.5))

        # Linee orizzontali anche qui
        ax2.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax2.xaxis.grid(False)
    else:
        ax2.text(0.5, 0.5, "Static Sampling\n(No search iterations logged)",
                 ha='center', va='center', fontsize=12, color='gray', style='italic')
        ax2.axis('off')

    # --- 3. Riduzione Parametri ---
    ax3 = fig.add_subplot(gs[2, 0])
    if not df['Params_M'].isna().all():
        ax3.plot(df['Iteration'], df['Params_M'], marker='o', linestyle='-', linewidth=2.5,
                 color='#2b8cbe', markerfacecolor='white', markeredgewidth=1.5, label='Model Size')
        ax3.fill_between(df['Iteration'], df['Params_M'], color='#2b8cbe', alpha=0.1)

    if b_params:
        ax3.axhline(y=b_params, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'Baseline ({b_params}M)')

    ax3.set_ylabel("Parameters (Millions)", fontweight='bold')
    ax3.set_xlabel("Iteration", fontweight='bold')
    ax3.set_xticks(df['Iteration'])
    ax3.legend(loc='center left', framealpha=1, bbox_to_anchor=(1.02, 0.5))

    # Linee orizzontali per il terzo grafico
    ax3.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax3.xaxis.grid(False)

    for ax in [ax1, ax2, ax3]:
        if not ax.axis() == (0.0, 1.0, 0.0, 1.0):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # 3. FIX SPAZIATURA: Riduce lo spazio a sinistra (0.06) e lascia il 20% dello schermo a destra (0.80) per la legenda
    fig.subplots_adjust(left=0.06, right=0.80, top=0.92, bottom=0.08)

    root.attributes('-topmost', True)
    salva = messagebox.askyesno("Salvataggio", "Vuoi salvare il grafico in formato PDF?")

    if salva:
        output_pdf = "nas_analysis_plot.pdf"
        # Manteniamo bbox_inches='tight' solo per il salvataggio così il PDF esce perfetto
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

    if not file_path:
        print("Nessun file selezionato.")
        root.destroy()
        return

    df, b_acc, b_params = parse_nas_log(file_path)

    if df.empty:
        print("Impossibile trovare dati validi nel file selezionato.")
        root.destroy()
        return

    print(f"\n--- Analisi completata per {os.path.basename(file_path)} ---")
    create_dashboard(df, b_acc, b_params, root)
    root.destroy()


if __name__ == "__main__":
    main()