import os
from torchvision.datasets import CIFAR100
import timm, torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
from FirstFineTuning.FineTuneUtils import eval_loop, train_model, save_model, EarlyStopping
from NAS.CompressedViT import CompressedViT
from NAS.NAS_Utils import count_params_no_mask
from NAS.HybridNAS import HybridNAS
from NASv2utils import get_search_set
import time

batch_size = 128
N_iterations = 2
lr = 0.5e-5
weight_decay = 0.05
images_per_class = 20
depth_limit = 8
max_epochs = 5
patience = 2
min_delta = 0.0001
early_stop_path = "D:\\Tesi\\NASv2"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 100
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)
    checkpoint = torch.load("D:\\Tesi\\FirstFineTuning\\best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    original_params = count_params_no_mask(model)

    data_config = timm.data.resolve_model_data_config(model)
    imagenet_mean, imagenet_std = data_config["mean"], data_config["std"]

    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),
         transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])

    val_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),
         transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    total_size = 50000
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    train_set = CIFAR100(root="D:\\Tesi\\CIFAR100", train=True, download=True, transform=train_transform)
    val_set = CIFAR100(root="D:\\Tesi\\CIFAR100", train=True, download=True, transform=val_transform)
    test_set = CIFAR100(root="D:\\Tesi\\CIFAR100", train=False, download=True, transform=test_transform)

    generator = torch.Generator().manual_seed(42)
    train_split, val_split = random_split(train_set, [train_size, val_size], generator=generator)

    train_set = Subset(train_set, train_split.indices)
    val_set = Subset(val_set, val_split.indices)

    print(f"Train size: {len(train_set)}")
    print(f"Val size: {len(val_set)}")
    print(f"Test size: {len(test_set)}")

    classes = train_set.dataset.classes

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()

    initial_params_count = count_params_no_mask(model)
    curr_params = initial_params_count

    _, initial_acc, _, _ = eval_loop(model, val_loader, loss_fn, device, classes)

    print(f"\n{'=' * 60}")
    print(f"🚀 AVVIO PIPELINE ITERATIVE NAS (CIFAR-100)")
    print(f"{'=' * 60}")
    print(f"📌 Baseline Model:")
    print(f"   - Params: {initial_params_count:.2f}M")
    print(f"   - Accuracy: {initial_acc * 100:.2f}%")
    print(f"{'=' * 60}\n")

    total_start_time = time.time()

    for n in range(N_iterations):
        iter_start_time = time.time()
        print(f"\n🔹 ITERAZIONE {n + 1}/{N_iterations}")
        print(f"   [1/3] Campionamento Search Set & NAS Search...")

        # 1. Search Set & Loader
        search_set = get_search_set(train_set, images_per_class, num_classes)
        search_loader = DataLoader(search_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

        # 2. HybridNAS Execution
        nas_start = time.time()
        nas = HybridNAS(model, loss_fn=loss_fn, search_loader=search_loader, device=device,
                        original_params=original_params)
        state, best_val = nas.search(depth_limit=depth_limit)
        nas_duration = time.time() - nas_start

        # 3. Compressione Fisica
        print(f"   [2/3] Applicazione Pruning e Compressione Fisica...")
        model = nas.apply_pruning(state, model)
        # Nota: original_head_dim=64 è hardcoded per ViT-Small come discusso
        comp_model = CompressedViT(state, model, original_head_dim=64).to(device)

        # --- METRICHE POST-PRUNING (A Freddo) ---
        _, acc_pruned, _, _ = eval_loop(comp_model, val_loader, loss_fn, device, classes)
        curr_params = count_params_no_mask(comp_model)

        # Calcoli Statistici
        params_dropped = (
                original_params - count_params_no_mask(model))  # Nota: 'model' qui è quello vecchio mascherato
        iter_reduction = 100 * (1 - (curr_params / count_params_no_mask(model)))  # Riduzione locale
        total_reduction = 100 * (1 - (curr_params / initial_params_count))  # Riduzione globale
        acc_drop = (initial_acc - acc_pruned) * 100

        print(f"   📊 REPORT POST-TAGLIO:")
        print(f"      - Params: {curr_params:.2f}M (Riduzione Globale: {total_reduction:.2f}%)")
        print(f"      - Accuracy: {acc_pruned * 100:.2f}% (Drop: {acc_drop:.2f} %)")
        print(f"      - Tempo Ricerca: {nas_duration:.1f}s")

        # 4. Recovery Fine-Tuning
        print(f"   [3/3] Recovery Fine-Tuning...")
        ft_start = time.time()

        model = comp_model
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

        earlystop = EarlyStopping(path=early_stop_path, patience=patience, min_delta=min_delta)

        # Passiamo val_dataloader per vedere i log nel training, ma ignoriamo il return qui
        _, _, _ = train_model(
            model, max_epochs, optimizer=optim, device=device,
            train_dataloader=train_loader, loss_fn=loss_fn,
            scheduler=scheduler, val_dataloader=val_loader,
            early_stopping=earlystop
        )
        ft_duration = time.time() - ft_start

        best_model_path = os.path.join(early_stop_path, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        # --- METRICHE FINALI ITERAZIONE ---
        _, acc_final, _, _ = eval_loop(model, val_loader, loss_fn, device, classes)
        recovered_points = (acc_final - acc_pruned) * 100

        print(f"   ✅ FINE ITERAZIONE {n + 1}")
        print(f"      - Accuracy Recuperata: {acc_final * 100:.2f}% (+{recovered_points:.2f}%)")
        print(f"      - Tempo Fine-Tuning: {ft_duration:.1f}s")
        print(f"      - Tempo Totale Iter: {(time.time() - iter_start_time):.1f}s")
        print(f"{'-' * 60}")

    print(f"\n🎯 VALUTAZIONE FINALE (Test Set)")
    _, final_test_acc, _, _ = eval_loop(model, test_loader, loss_fn, device, classes, report=True)

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

    # SALVATAGGIO DEL MODELLO
    model.eval()

    random_input = torch.randn(1, 3, 224, 224).to(device)

    try:
        traced_model = torch.jit.trace(model, random_input)
        traced_model.save("D:\\Tesi\\NASv2\\best_model.pt")
        print("MODELLO SALVATO CORRETTAMENTE")

        # Test di verifica immediato
        print("   Verifica caricamento JIT...")
        loaded_jit = torch.jit.load("D:\\Tesi\\NASv2\\best_model.pt")
        loaded_jit.eval()
        with torch.no_grad():
            out_orig = model(random_input)
            out_jit = loaded_jit(random_input)
            # Verifica che i risultati siano identici
            diff = (out_orig - out_jit).abs().max()
            print(f"   Differenza max output Originale vs JIT: {diff:.6f}")  # Deve essere vicina a 0

    except Exception as e:
        print(f"❌ Errore durante l'export JIT: {e}")
