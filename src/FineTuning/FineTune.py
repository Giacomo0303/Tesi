import timm
import torch
from src.Datasets.Imagenet import ImageNet
from src.utils.FineTuneUtils import EarlyStopping, train_model, eval_loop, check_top5_accuracy, plot_training_results
from src.Datasets.Cifar100 import Cifar100
from torch.optim import AdamW, lr_scheduler
from torch.nn import CrossEntropyLoss
from src.utils.NAS_Utils import load_model
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import torch.optim as optim

model_name = "deit_small_patch16_224"
teacher_name = None
teacher_path = "D:\\Tesi\\src\\FineTuning\\vit_base_cifar100.pth"
save_path = "C:\\Users\\cvip\\Desktop\\Tesi_Lombardo\\src\\FineTuning"
dataset_name = "imagenet"
img_size = 224
batch_size = 128
N_epochs = 50
validation = True
backbone_tuning = False
discriminative_lr = False
backbone_lr = 1e-5  # used only with discriminative_lr
head_lr = 1e-4  # used only with discriminative_lr
base_lr = 1e-6
weight_decay = 1e-8
patience = 10
min_delta = 0.0001

if __name__ == "__main__":
    # creazione del dataset
    if dataset_name == "imagenet":
        dataset = ImageNet(root_path="D:\\Lombardo\\ImageNet", batch_size=batch_size, model_name=model_name,
                           train_size=0.97)
    elif dataset_name == "cifar100":
        dataset = Cifar100(root_path="C:\\Users\\cvip\\Desktop\\Tesi_Lombardo\\Data\\CIFAR100", img_size=img_size,
                           batch_size=batch_size, model_name=model_name, mean_std="imagenet")
    else:
        raise Exception("Invalid dataset name")

    train_loader = dataset.get_train_loader(num_workers=6)
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()
    print(f"train set: {len(train_loader.dataset)}")
    print(f"val set: {len(val_loader.dataset)}")
    print(f"test set: {len(test_loader.dataset)}")

    # creazione del modello
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name=model_name, pretrained=True, num_classes=dataset.num_classes)
    model.reset_classifier(dataset.num_classes)
    model.to(device)
    print("modello creato.")

    teacher_model = None
    if teacher_name is not None:
        teacher_model = load_model(model_name=teacher_name, path=teacher_path, num_classes=dataset.num_classes)
        teacher_model.eval()
        teacher_model.to(device)

    # discriminative learning rates
    # prelevo i pesi della backbone controllando che non siano quelli della head
    if hasattr(model, "head_dist"):
        head_params = list(model.head.parameters()) + list(model.head_dist.parameters())
    else:
        head_params = list(model.get_classifier().parameters())

    head_params_pointers = set(p.data_ptr() for p in head_params)
    backbone_params = [p for p in model.parameters() if p.data_ptr() not in head_params_pointers]
    if backbone_tuning:
        param_groups = [
            {
                "params": backbone_params,
                "lr": backbone_lr,
            },
            {
                "params": head_params,
                "lr": head_lr,
            }
        ]
    else:
        param_groups = [
            {
                "params": head_params,
                "lr": head_lr,
            }
        ]
        for param in backbone_params:
            param.requires_grad = False

    train_loss_fn = CrossEntropyLoss()
    val_loss_fn = CrossEntropyLoss()

    if discriminative_lr:
        optimizer = AdamW(param_groups, weight_decay=weight_decay)
    else:
        active_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(active_params, lr=base_lr, weight_decay=weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_epochs, eta_min=1e-6)
    if validation:
        print("validazione...")
        _, acc, _, _ = eval_loop(model, test_loader, val_loss_fn, device, dataset.classes)
        print(f"Top1 accuracy iniziale: {acc * 100:.2f}%")
        if teacher_model is not None:
            _, acc_teacher, _, _ = eval_loop(teacher_model, test_loader, val_loss_fn, device, dataset.classes)
            print(f"Top1 accuracy teacher: {acc_teacher * 100:.2f}%")

    early_stopping = EarlyStopping(path=save_path, patience=patience, min_delta=min_delta)

    train_loss, val_loss, accuracy = train_model(model, N_epochs, optimizer=optimizer, device=device,
                                                 train_dataloader=train_loader, loss_fn=train_loss_fn,
                                                 val_loss_fn=val_loss_fn,
                                                 early_stopping=early_stopping, val_dataloader=val_loader,
                                                 scheduler=scheduler, teacher_model=teacher_model)

    plot_training_results(train_loss, val_loss, accuracy)

    checkpoint = torch.load("C:\\Users\\cvip\\Desktop\\Tesi_Lombardo\\src\\FineTuning\\best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    _, _, y_true, y_pred = eval_loop(model, test_loader, val_loss_fn, device, dataset.classes, report=False)
    print(f"Top5 accuracy: {check_top5_accuracy(model, test_loader, device):.2f}%")
