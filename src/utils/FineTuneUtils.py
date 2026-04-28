import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, classification_report
from tqdm import trange


def save_model(net, current_epoch, path):
    torch.save({
        'epoch': current_epoch,
        'model_state_dict': net.state_dict(),
    }, path)


class EarlyStopping:
    def __init__(self, path, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.min_validation_loss = torch.inf
        self.path = path

    def __call__(self, validation_loss, current_epoch, model):
        if validation_loss + self.min_delta >= self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self.min_validation_loss = validation_loss
            print("---Best model saved---")
            save_model(model, current_epoch, f"{self.path}\\best_model.pth")


def train_loop(model, dataloader, loss_fn, optimizer, device, pbar, scaler, teacher_model=None, T=4.0, alpha=0.9,
               mixup_fn=None):
    num_batches = len(dataloader)
    epoch_loss = 0.0

    model.train()
    for _, (X, y) in zip(pbar, dataloader):
        X, y = X.to(device), y.to(device)
        if hasattr(model, "dist_token"):
            batch_loss = deit_train_loop(model, X, y, loss_fn, teacher_model, mixup_fn)
        else:
            batch_loss = vit_train_loop(model, X, y, loss_fn, teacher_model=teacher_model, T=T, alpha=alpha , mixup_fn=mixup_fn)

        epoch_loss += batch_loss.item()

        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return epoch_loss / num_batches


def vit_train_loop(model, X, y, loss_fn, teacher_model=None, T=4.0, alpha=0.9, mixup_fn=None):
    if mixup_fn is not None:
        X, y = mixup_fn(X, y)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        student_logits = model(X)
        batch_loss = loss_fn(student_logits, y)

        # distillation with temperature
        if teacher_model is not None:
            with torch.inference_mode():
                teacher_logits = teacher_model(X)

            distillation_loss = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                                         F.softmax(teacher_logits / T, dim=1), reduction='batchmean') * (T ** 2)

            batch_loss = (1 - alpha) * batch_loss + alpha * distillation_loss

    return batch_loss


def deit_train_loop(model, X, y, loss_fn, teacher_model, mixup_fn):
    if mixup_fn is not None:
        X, y = mixup_fn(X, y)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output_tokens = model.forward_features(X)
        logits_cls = model.head(output_tokens[:, 0])
        logits_distil = model.head_dist(output_tokens[:, 1])

        cls_loss = loss_fn(logits_cls, y)
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X)
                # hard distillation
                teacher_labels = teacher_logits.argmax(dim=1)
                dist_loss = loss_fn(logits_distil, teacher_labels)
        else:
            dist_loss = loss_fn(logits_distil, y)

        batch_loss = (cls_loss + dist_loss) / 2
        return batch_loss


def eval_loop(model, dataloader, loss_fn, device, classes=None, report=False):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    epoch_loss = 0.0
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(X)
                batch_loss = loss_fn(logits, y)

            epoch_loss += batch_loss.item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

    epoch_loss = epoch_loss / num_batches
    accuracy = balanced_accuracy_score(y_true, y_pred)

    if report:
        print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    print(f"Balanced Accuracy: {accuracy * 100:.2f}%")

    return epoch_loss, accuracy, y_true, y_pred


def check_top5_accuracy(model, dataloader, device):
    correct_top5 = 0
    samples_count = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(X)

            top5 = torch.topk(logits, k=5, dim=1).indices
            y = y.reshape(y.shape[0], 1)
            top5_check = (top5 == y)

            correct_top5 += top5_check.any(dim=1).sum().item()
            samples_count += y.size(0)

    return (correct_top5 / samples_count) * 100


def train_model(model, epoch, optimizer, device, train_dataloader, loss_fn, val_loss_fn, early_stopping=None,
                val_dataloader=None, scheduler=None, teacher_model=None, T=4.0, alpha=0.9, mixup_fn=None):
    train_loss, val_loss = [], []
    accuracy = []
    scaler = torch.amp.GradScaler()

    num_batches = len(train_dataloader)

    for epoch in range(1, epoch + 1):
        pbar = trange(num_batches)
        pbar.set_description(desc=f"Epoch {epoch}")
        epoch_loss = train_loop(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer,
                                device=device, pbar=pbar, scaler=scaler, teacher_model=teacher_model, T=T, alpha=alpha,
                                mixup_fn=mixup_fn)
        train_loss.append(epoch_loss)

        if val_dataloader is not None:
            epoch_val_loss, epoch_accuracy, _, _ = eval_loop(model=model, dataloader=val_dataloader, loss_fn=val_loss_fn,
                                                             device=device)
            val_loss.append(epoch_val_loss)
            accuracy.append(epoch_accuracy)
            val_str = f"Val loss: {epoch_val_loss:6.4f}\tAccuracy: {epoch_accuracy:6.4f}"
            print(f"Train_loss: {epoch_loss:6.4f}\n{val_str}\n")

            if early_stopping is not None:
                early_stopping(epoch_val_loss, epoch, model)
                if early_stopping.early_stop:
                    break

        if scheduler is not None:
            scheduler.step()

    return train_loss, val_loss, accuracy


def plot_training_results(train_loss, val_loss, accuracy):
    plt.style.use('seaborn-v0_8-whitegrid')
    epochs = range(1, len(train_loss) + 1)

    # Creazione della figura con due subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Grafico 1: Loss (Train vs Val) ---
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', markersize=4, linewidth=1.5)
    ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss', markersize=4, linewidth=1.5)
    ax1.set_title('Andamento della Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoca', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- Grafico 2: Accuracy ---
    ax2.plot(epochs, [a * 100 for a in accuracy], 'g-^', label='Balanced Accuracy', markersize=4, linewidth=1.5)
    ax2.set_title('Andamento dell\'Accuratezza', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoca', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylim(0, 100)  # Spesso utile per vedere il margine di miglioramento reale
    ax2.legend(frameon=True, shadow=True, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Ottimizzazione dello spazio tra i grafici
    plt.tight_layout()

    # Salvataggio opzionale per la tesi
    # plt.savefig("training_results.pdf", bbox_inches='tight')
    plt.show()
