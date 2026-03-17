import timm
import torch
from Datasets.Imagenet import ImageNet
from src.utils.FineTuneUtils import EarlyStopping, train_model, eval_loop, check_top5_accuracy, plot_training_results
from src.Datasets.Cifar100 import Cifar100
from torch.optim import AdamW, lr_scheduler
from torch.nn import CrossEntropyLoss

model_name = "vit_small_patch16_224"
save_path = "C:\\Users\\cvip\\Desktop\\Tesi_Lombardo\\src\\FineTuning"
img_size = 224
batch_size = 128
N_epochs = 30
validation = True
backbone_tuning = False
backbone_lr = 0.5e-5
head_lr = 0.5e-4
weight_decay = 0.05
patience = 5
min_delta = 0.001

if __name__ == "__main__":
    # creazione del dataset
    dataset = ImageNet(root_path="D:\\Lombardo\\ImageNet", batch_size=batch_size, model_name=model_name, train_size=0.97)
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
    # discriminative learning rates

    # prelevo i pesi della backbone controllando che non siano quelli della head
    head_params_pointers = set(p.data_ptr() for p in model.head.parameters())
    backbone_params = [p for p in model.parameters() if p.data_ptr() not in head_params_pointers]
    if backbone_tuning:
        param_groups = [
            {
                "params": backbone_params,
                "lr": backbone_lr,
            },
            {
                "params": model.head.parameters(),
                "lr": head_lr,
            }
        ]
    else:
        param_groups = [
            {
                "params": model.head.parameters(),
                "lr": head_lr,
            }
        ]
        for param in backbone_params:
            param.requires_grad = False

    optim = AdamW(param_groups, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=N_epochs, eta_min=1e-7)
    loss_fn = CrossEntropyLoss()
    if validation:
        print("validazione...")
        _, acc, y_true, y_pred = eval_loop(model, test_loader, loss_fn, device, dataset.classes)
        #print(f"Top5 accuracy: {check_top5_accuracy(model, test_loader, device):.2f}%")
        print(f"Top1 accuracy: {acc * 100:.2f}%")

    early_stopping = EarlyStopping(path=save_path, patience=patience, min_delta=min_delta)

    train_loss, val_loss, accuracy = train_model(model, N_epochs, optimizer=optim, device=device,
                                                 train_dataloader=train_loader, loss_fn=loss_fn,
                                                 early_stopping=early_stopping, val_dataloader=val_loader,
                                                 scheduler=scheduler)

    plot_training_results(train_loss, val_loss, accuracy)

    checkpoint = torch.load("C:\\Users\\cvip\\Desktop\\Tesi_Lombardo\\src\\FineTuning\\best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    _, _, y_true, y_pred = eval_loop(model, test_loader, loss_fn, device, dataset.classes, report=True)
    print(f"Top5 accuracy: {check_top5_accuracy(model, test_loader, device):.2f}%")
