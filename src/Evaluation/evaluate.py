import torch
from src.Datasets.Cifar100 import Cifar100
from src.utils.FineTuneUtils import eval_loop
from time import time
from torch.utils.flop_counter import FlopCounterMode
from src.utils.NAS_Utils import load_model


def get_flops(model, inp, with_backward=False):
    model.eval()

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops = flop_counter.get_total_flops()
    return total_flops

# Configurazione
device = "cuda" if torch.cuda.is_available() else "cpu"
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Caricamento Dati
# Nota: Aggiungi seed se richiesto dal costruttore
dataset = Cifar100(root_path="/CIFAR100", img_size=224, batch_size=128, mean_std="imagenet",
                   model_name="vit_small_patch16_224", seed=42)
test_loader = dataset.get_test_loader()

# 1. MODELLO ORIGINALE
original_model = load_model(model_name="vit_small_patch16_224", num_classes=dataset.num_classes,
                            path="/FirstFineTuning/best_model.pth")
original_model.to(device)
original_model.eval()

# Calcolo FLOPs Originale
original_flops = get_flops(model=original_model, inp=dummy_input, with_backward=False)

# Calcolo Accuracy e Tempo Originale
time_start = time()
_, original_acc, _, _ = eval_loop(original_model, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), device=device,
                                  classes=dataset.classes)
time_end = time()
original_time = time_end - time_start


# 2. MODELLO PRUNED
pruned_model = torch.load("/NASv2/best_model.pth", weights_only=False) # Assicurati sia l'intero modello
pruned_model.to(device)
pruned_model.eval()

# Calcolo FLOPs Pruned (CORRETTO: passiamo pruned_model)
pruned_flops = get_flops(model=pruned_model, inp=dummy_input, with_backward=False)

# Calcolo Accuracy e Tempo Pruned
time_start = time()
_, pruned_acc, _, _ = eval_loop(pruned_model, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), device=device,
                                classes=dataset.classes)
time_end = time()
pruned_time = time_end - time_start

# --- CALCOLO DELLE METRICHE DERIVATE ---

# Conversione in GFLOPs (10^9)
gflops_orig = original_flops / 1e9
gflops_pruned = pruned_flops / 1e9

# Throughput (Immagini / secondo)
# Nota: Questo è il throughput sul test set intero (includendo dataloading overhead),
# per un throughput puro si dovrebbe misurare solo l'inference loop.
n_samples = len(test_loader.dataset)
throughput_orig = n_samples / original_time
throughput_pruned = n_samples / pruned_time

# Calcolo Delta (%)
delta_flops = (1 - (pruned_flops / original_flops)) * 100
delta_time = (1 - (pruned_time / original_time)) * 100
delta_acc = (pruned_acc - original_acc) * 100 # Differenza assoluta per l'accuracy
delta_throughput = ((throughput_pruned - throughput_orig) / throughput_orig) * 100

# --- STAMPA TABELLARE ---

print("\n" + "="*85)
print(f"📊 CONFRONTO PRESTAZIONI: ViT Original vs ViT Pruned")
print("="*85)
print(f"{'Metrica':<25} | {'Originale':<15} | {'Pruned':<15} | {'Delta':<15}")
print("-" * 85)

# FLOPs
print(f"{'GFLOPs (Inference)':<25} | {gflops_orig:<15.4f} | {gflops_pruned:<15.4f} | -{delta_flops:<10.2f}% (Riduzione)")

# Accuracy
print(f"{'Accuracy (Top-1)':<25} | {original_acc*100:<14.2f}% | {pruned_acc*100:<14.2f}% | {delta_acc:<+10.2f} p.p.")

# Tempo Totale
print(f"{'Tempo Test Set (s)':<25} | {original_time:<15.2f} | {pruned_time:<15.2f} | -{delta_time:<10.2f}% (Velocità)")

# Throughput
print(f"{'Throughput (img/s)':<25} | {throughput_orig:<15.1f} | {throughput_pruned:<15.1f} | +{delta_throughput:<10.1f}% (Speedup)")

print("="*85 + "\n")
