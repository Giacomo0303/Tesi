from torchvision.datasets import CIFAR100
import timm, torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
from FirstFineTuning.FineTuneUtils import eval_loop, train_model, save_model
from NAS.CompressedViT import CompressedViT
from NAS.NAS_Utils import count_params_no_mask
from NAS.HybridNAS import HybridNAS
from NASv2utils import get_search_set

batch_size = 128
N_iterations = 2
lr = 0.5e-5
weight_decay = 0.05
images_per_class = 20
depth_limit = 1
n_epochs = 1

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
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])

    val_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])

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

    _, acc, _, _ = eval_loop(model, val_loader, loss_fn, device, classes)
    print(f"VALORE INIZIALE DI ACCURACY SUL VAL SET: {acc}")

    for n in range(N_iterations):
        search_set = get_search_set(train_set, images_per_class, num_classes)
        search_loader = DataLoader(search_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        nas = HybridNAS(model, loss_fn=loss_fn, search_loader=search_loader, device=device, original_params=original_params)
        state, best_val = nas.search(depth_limit=depth_limit)
        model = nas.apply_pruning(state, model)
        comp_model = CompressedViT(state, model).to(device)
        _, acc, _, _ = eval_loop(comp_model, val_loader, loss_fn, device, classes)
        print(f"ACCURACY DOPO LA RICERCA ALL'ITERAZIONE {n}: {acc}")
        print(f"PARAMETRI DEL MODELLO DOPO LA RICERCA: {count_params_no_mask(comp_model)}")

        model = comp_model
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)
        _, _, _ = train_model(model, n_epochs, optimizer=optim, device=device,
                                                     train_dataloader=train_loader, loss_fn=loss_fn, scheduler=scheduler, val_dataloader=val_loader)

    save_model(model, 0, f"D:\\Tesi\\NASv2\\pruned_model.pth")
    _, acc, _, _ = eval_loop(model, test_loader, loss_fn, device, classes, report=True)
    print(f"ACCURACY FINALE SUL TEST SET: {acc}")


