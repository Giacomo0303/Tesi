import torch
import time
from src.Datasets.Cifar100 import Cifar100
from src.Datasets.Imagenet import ImageNet
from src.utils.FineTuneUtils import eval_loop
from torch.utils.flop_counter import FlopCounterMode
from torch.nn.attention import sdpa_kernel, SDPBackend  # Nuovo import corretto
from src.utils.NAS_Utils import load_model


def get_flops(model, inp, with_backward=False):
    model.eval()
    flop_counter = FlopCounterMode(display=False, depth=None)

    # Utilizzo del nuovo context manager (PyTorch 2.1+)
    # Forza l'uso del backend matematico per permettere a FlopCounter di vedere le operazioni
    with sdpa_kernel(SDPBackend.MATH):
        with flop_counter:
            if with_backward:
                model(inp).sum().backward()
            else:
                model(inp)

    total_flops = flop_counter.get_total_flops()
    return total_flops


def measure_throughput(model, device, batch_size=128, img_size=224, warmup_runs=50, test_runs=200):
    """
    Misura il throughput puro del modello isolandolo dal Dataloader e dai colli di bottiglia della CPU.
    """
    model.to(device)
    model.eval()

    # 1. Creiamo i dati direttamente sulla memoria della GPU (Elimina il Dataloader)
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    with torch.no_grad():
        # 2. WARM-UP: Svegliamo la GPU
        for _ in range(warmup_runs):
            _ = model(dummy_input)

        # 3. SINCRONIZZAZIONE: Assicuriamoci che la GPU abbia finito il warm-up
        if device == "cuda":
            torch.cuda.synchronize()

        # --- INIZIO MISURAZIONE ---
        start_time = time.time()

        # 4. ESECUZIONE REALE
        for _ in range(test_runs):
            _ = model(dummy_input)

        # 5. SINCRONIZZAZIONE: Obbliga Python ad aspettare che l'ultima immagine sia finita
        if device == "cuda":
            torch.cuda.synchronize()

        # --- FINE MISURAZIONE ---
        end_time = time.time()

    # Calcolo delle metriche
    total_time = end_time - start_time
    total_images = batch_size * test_runs

    throughput_imgs_per_sec = total_images / total_time
    latency_per_batch_ms = (total_time / test_runs) * 1000

    return throughput_imgs_per_sec, latency_per_batch_ms


if __name__ == "__main__":
    # Configurazione
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print("Caricamento dataset in corso...")
    dataset = ImageNet(root_path="D:\\Lombardo\\ImageNet", batch_size=128,
             model_name="vit_small_patch16_224", seed=42, train_size=0.97)
    test_loader = dataset.get_test_loader()

    print("Caricamento modelli in corso...")
    original_model = load_model(model_name="vit_small_patch16_224", num_classes=dataset.num_classes,
                                path="C:\\Users\\cvip\\Desktop\\Tesi_Lombardo\\src\\FineTuning\\vit_small_imagenet.pth")
    pruned_model = torch.load("C:\\Users\\cvip\\Desktop\\Tesi_Lombardo\\src\\NAS\\Results_imagenet\\vit_small_patch16_224_iter14.pth", weights_only=False)

    # ==========================================
    # 1. MODELLO ORIGINALE
    # ==========================================
    print("\n--- Valutazione Modello Originale ---")
    original_model.to(device)

    original_flops = get_flops(model=original_model, inp=dummy_input, with_backward=False)

    print("   Benchmarking hardware...")
    pure_thr_orig, pure_lat_orig = measure_throughput(original_model, device)

    print("   Valutazione Test Set...")
    time_start = time.time()
    _, original_acc, _, _ = eval_loop(original_model, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), device=device,
                                      classes=dataset.classes)
    original_time = time.time() - time_start

    # ==========================================
    # 2. MODELLO PRUNED
    # ==========================================
    print("\n--- Valutazione Modello Pruned ---")
    pruned_model.to(device)

    pruned_flops = get_flops(model=pruned_model, inp=dummy_input, with_backward=False)

    print("   Benchmarking hardware...")
    pure_thr_pruned, pure_lat_pruned = measure_throughput(pruned_model, device)

    print("   Valutazione Test Set...")
    time_start = time.time()
    _, pruned_acc, _, _ = eval_loop(pruned_model, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), device=device,
                                    classes=dataset.classes)
    pruned_time = time.time() - time_start

    # ==========================================
    # CALCOLO DELLE METRICHE DERIVATE
    # ==========================================
    gflops_orig = original_flops / 2e9
    gflops_pruned = pruned_flops / 2e9

    # Calcolo Delta (%)
    delta_flops = (1 - (pruned_flops / original_flops)) * 100
    delta_acc = (pruned_acc - original_acc) * 100

    # Delta Tempo/Throughput su Test Set completo (sporco, include dataloader)
    n_samples = len(test_loader.dataset)
    test_thr_orig = n_samples / original_time
    test_thr_pruned = n_samples / pruned_time
    delta_test_thr = ((test_thr_pruned - test_thr_orig) / test_thr_orig) * 100
    delta_test_time = (1 - (pruned_time / original_time)) * 100

    # Delta Throughput Hardware Puro (pulito)
    delta_pure_thr = ((pure_thr_pruned - pure_thr_orig) / pure_thr_orig) * 100
    delta_pure_lat = (1 - (pure_lat_pruned / pure_lat_orig)) * 100  # Riduzione Latenza

    # ==========================================
    # STAMPA TABELLARE
    # ==========================================
    print("\n" + "=" * 85)
    print(f"📊 CONFRONTO PRESTAZIONI: ViT Original vs ViT Pruned")
    print("=" * 85)
    print(f"{'Metrica':<30} | {'Originale':<15} | {'Pruned':<15} | {'Delta':<15}")
    print("-" * 85)

    print("\n[ 1. COMPLESSITÀ TEORICA (Math) ]")
    print(
        f"{'GFLOPs (Inference)':<30} | {gflops_orig:<15.4f} | {gflops_pruned:<15.4f} | -{delta_flops:<10.2f}% (Riduzione)")

    print("\n[ 2. HARDWARE BENCHMARK (Puro GPU, No CPU/Disco) ]")
    print(
        f"{'Throughput (img/s)':<30} | {pure_thr_orig:<15.1f} | {pure_thr_pruned:<15.1f} | +{delta_pure_thr:<10.2f}% (Speedup)")
    print(
        f"{'Batch Latency (ms)':<30} | {pure_lat_orig:<15.1f} | {pure_lat_pruned:<15.1f} | -{delta_pure_lat:<10.2f}% (Velocità)")

    print("\n[ 3. PIPELINE END-TO-END (Con Dataloader) ]")
    print(
        f"{'Accuracy (Top-1)':<30} | {original_acc * 100:<14.2f}% | {pruned_acc * 100:<14.2f}% | {delta_acc:<+10.2f} p.p.")
    print(
        f"{'Throughput Raw (img/s)':<30} | {test_thr_orig:<15.1f} | {test_thr_pruned:<15.1f} | +{delta_test_thr:<10.2f}% (Speedup)")
    print(
        f"{'Tempo Test Set Totale (s)':<30} | {original_time:<15.2f} | {pruned_time:<15.2f} | -{delta_test_time:<10.2f}%")

    print("=" * 85 + "\n")