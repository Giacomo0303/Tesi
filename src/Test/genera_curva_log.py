import re
import matplotlib.pyplot as plt


def plot_pareto_from_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Parsing della Baseline (Validation Set)
    baseline_match = re.search(r'Baseline Model:\s+- Params:\s+([0-9.]+)M\s+- Accuracy:\s+([0-9.]+)%', content)
    if not baseline_match:
        print("Errore: Baseline non trovata nel log.")
        return

    baseline_params = float(baseline_match.group(1))
    baseline_acc = float(baseline_match.group(2))

    # 2. Parsing delle Iterazioni Intermedie (Validation Set)
    param_matches = re.findall(r'REPORT POST-TAGLIO:\s+- Params:\s+([0-9.]+)M', content)
    acc_matches = re.findall(r'FINE ITERAZIONE \d+\s+- Accuracy Recuperata:\s+([0-9.]+)%', content)

    iter_params = [float(p) for p in param_matches]
    iter_accs = [float(a) for a in acc_matches]

    # 3. Parsing del Risultato Finale (Test Set)
    test_acc_match = re.search(r'Accuracy Test Set:\s+([0-9.]+)%', content)
    final_test_acc = float(test_acc_match.group(1)) if test_acc_match else None

    final_params_match = re.search(r'Parametri Finali:\s+([0-9.]+)M', content)
    final_test_params = float(final_params_match.group(1)) if final_params_match else iter_params[-1]

    # --- Preparazione Dati per il Plot ---
    val_params = [baseline_params] + iter_params
    val_accs = [baseline_acc] + iter_accs
    iterations = [0] + list(range(1, len(iter_params) + 1))

    # --- Setup Grafico ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(11, 7))

    # Disegna la curva di Pareto sul Validation Set
    ax.plot(val_params, val_accs, 'o-', color='tab:blue', linewidth=2.5, markersize=7,
            label="Frontiera (Validation Set)")

    # Aggiungi etichette ad alcune iterazioni per non sovraffollare il grafico
    for i, txt in enumerate(iterations):
        if txt == 0 or txt == 5 or txt == 10 or txt == 15:
            ax.annotate(f"Iter {txt}", (val_params[i], val_accs[i]),
                        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='darkblue')

    # Disegna il punto finale sul Test Set
    if final_test_acc:
        ax.plot([final_test_params], [final_test_acc], 'X', color='tab:red', markersize=14,
                label="Valutazione Finale (Test Set)")
        ax.annotate(f"Test: {final_test_acc}%\nParams: {final_test_params}M",
                    (final_test_params, final_test_acc),
                    textcoords="offset points", xytext=(0, -35), ha='center', fontsize=11, color='darkred',
                    fontweight='bold')

        # Linea tratteggiata per far capire la transizione Val -> Test
        ax.plot([val_params[-1], final_test_params], [val_accs[-1], final_test_acc], '--', color='gray', alpha=0.5)

    # Inverti l'asse X per leggere la riduzione dei parametri da sinistra verso destra
    ax.invert_xaxis()

    # Formattazione e Label
    ax.set_xlabel('Parametri (Milioni)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Frontiera di Pareto: Pruning ViT-Small con Knowledge Distillation', fontsize=15, fontweight='bold')

    ax.legend(loc='center left', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("Pareto_Frontier_Distillation.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Assicurati che il nome del file log coincida con quello che hai caricato
    log_path = "C:\\Users\\cvip\\Desktop\\Tesi_Lombardo\\src\\NAS\\Results_imagenet\\log_KD_81_11.txt"
    plot_pareto_from_log(log_path)