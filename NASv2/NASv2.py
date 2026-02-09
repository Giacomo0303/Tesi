import os
import timm, torch
from Dataset.Cifar100 import Cifar100
from torch.utils.data import DataLoader
from FirstFineTuning.FineTuneUtils import eval_loop
from NAS.NAS_Utils import count_params_no_mask
from NASv2utils import updatePruningReport, createPruningReport, savePruningReport
from NASv2utils import load_model, pruningNAS, recoveryFineTune, save_model
import time, copy

batch_size = 128
N_iterations = 1
lr = 0.5e-5
weight_decay = 0.05
images_per_class = 20
depth_limit = 6
max_epochs = 15
patience = 2
min_delta = 0.0001
early_stop_path = "D:\\Tesi\\NASv2"
seed = 42
num_classes = 100
search_threshold = 0.005
distillation = True
T = 2.0

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_name="vit_small_patch16_224", num_classes=num_classes,
                       path="D:\\Tesi\\FirstFineTuning\\best_model.pth")
    teacher_model = None

    if distillation:
        teacher_model = copy.deepcopy(model)
        teacher_model.to(device)
        teacher_model.eval()

    model = model.to(device)

    dataset = Cifar100(root_path="D:\\Tesi\\CIFAR100", img_size=224, batch_size=batch_size, mean_std="imagenet", model_name="vit_small_patch16_224", seed=seed)
    train_loader = dataset.get_train_loader(num_workers=2)
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()

    loss_fn = torch.nn.CrossEntropyLoss()

    initial_params_count = count_params_no_mask(model)
    curr_params = initial_params_count

    _, initial_acc, _, _ = eval_loop(model, val_loader, loss_fn, device, dataset.classes)
    pruningReport = createPruningReport(model)

    print(f"\n{'=' * 60}")
    print(f"AVVIO PIPELINE ITERATIVE NAS (CIFAR-100)")
    print(f"{'=' * 60}")
    print(f"Baseline Model:")
    print(f"   - Params: {initial_params_count:.2f}M")
    print(f"   - Accuracy: {initial_acc * 100:.2f}%")
    print(f"{'=' * 60}\n")

    total_start_time = time.time()

    for n in range(N_iterations):
        iter_start_time = time.time()
        print(f"\n ITERAZIONE {n + 1}/{N_iterations}")
        print(f"   [1/3] Campionamento Search Set & NAS Search...")

        # 1. Search Set & Loader
        search_loader = dataset.get_search_loader(n_per_classes=images_per_class)

        # 2. HybridNAS Execution
        # Nota: original_head_dim=64 è hardcoded per ViT-Small
        comp_model, nas_duration, state = pruningNAS(model=model, loss_fn=loss_fn, search_loader=search_loader, device=device,
                                              initial_params_count=initial_params_count, depth_limit=depth_limit,
                                              original_head_dim=64, threshold=search_threshold)

        pruningReport = updatePruningReport(pruningReport, state)
        savePruningReport(pruningReport, path="D:\\Tesi\\NASv2\\pruning_report.json")

        # --- METRICHE POST-PRUNING (A Freddo) ---
        _, acc_pruned, _, _ = eval_loop(comp_model, val_loader, loss_fn, device, dataset.classes)
        curr_params = count_params_no_mask(comp_model)

        # Statistiche
        params_dropped = (
                initial_params_count - count_params_no_mask(model))  # Nota: 'model' qui è quello vecchio mascherato
        iter_reduction = 100 * (1 - (curr_params / count_params_no_mask(model)))  # Riduzione locale
        total_reduction = 100 * (1 - (curr_params / initial_params_count))  # Riduzione globale
        acc_drop = (initial_acc - acc_pruned) * 100

        print(f"   REPORT POST-TAGLIO:")
        print(f"      - Params: {curr_params:.2f}M (Riduzione Globale: {total_reduction:.2f}%)")
        print(f"      - Accuracy: {acc_pruned * 100:.2f}% (Drop: {acc_drop:.2f} %)")
        print(f"      - Tempo Ricerca: {nas_duration:.1f}s")

        # 4. Recovery Fine-Tuning
        model = comp_model
        ft_duration = recoveryFineTune(model=model, lr=lr, weight_decay=weight_decay, max_epochs=max_epochs,
                                       early_stop_path=early_stop_path, patience=patience, min_delta=min_delta,
                                       device=device, train_loader=train_loader, val_loader=val_loader, loss_fn=loss_fn, teacher_model=teacher_model)

        best_model_path = os.path.join(early_stop_path, "best_model.pth")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # --- METRICHE FINALI ITERAZIONE ---
        _, acc_final, _, _ = eval_loop(model, val_loader, loss_fn, device, dataset.classes)
        recovered_points = (acc_final - acc_pruned) * 100

        print(f"   FINE ITERAZIONE {n + 1}")
        print(f"      - Accuracy Recuperata: {acc_final * 100:.2f}% (+{recovered_points:.2f}%)")
        print(f"      - Tempo Fine-Tuning: {ft_duration:.1f}s")
        print(f"      - Tempo Totale Iter: {(time.time() - iter_start_time):.1f}s")
        print(f"{'-' * 60}")


    print(f"\nVALUTAZIONE FINALE (Test Set)")
    _, final_test_acc, _, _ = eval_loop(model, test_loader, loss_fn, device, dataset.classes, report=True)

    total_duration = time.time() - total_start_time
    final_reduction = 100 * (1 - (curr_params / initial_params_count))

    print(f"\n{'=' * 60}")
    print(f"🏆 RISULTATI FINALI")
    print(f"{'=' * 60}")
    print(f"   - Tempo Totale: {total_duration / 60:.1f} min")
    print(f"   - Parametri Iniziali: {initial_params_count:.2f}M")
    print(f"   - Parametri Finali:   {curr_params:.2f}M")
    print(f"   - Riduzione Size:     {final_reduction:.2f}%")
    print(f"   - Accuracy Test Set:  {final_test_acc * 100:.2f}%")
    print(f"{'=' * 60}")

    save_model(model=model, path="D:\\Tesi\\NASv2\\best_model.pth")

