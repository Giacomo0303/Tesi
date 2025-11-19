import torch
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
            save_model(model, current_epoch, f"{self.path}\\best_model.pth")

def train_loop(model, dataloader, loss_fn, optimizer, device, pbar, scaler):
    num_batches = len(dataloader)
    epoch_loss = 0.0

    model.train()
    for _, (X, y) in zip(pbar, dataloader):
        X, y = X.to(device), y.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(X)
            batch_loss = loss_fn(logits, y)

        epoch_loss += batch_loss.item()

        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return epoch_loss / num_batches


def eval_loop(model, dataloader, loss_fn, device, classes=None, report=False):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    epoch_loss = 0.0
    accuracy = 0.0
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
            accuracy += (logits.argmax(dim=1) == y).type(torch.float).sum().item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

    epoch_loss = epoch_loss / num_batches
    accuracy = accuracy / size

    if report:
        print(classification_report(y_true, y_pred, target_names=classes, digits=3))
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred) * 100:.2f}%")

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



def train_model(model, epoch, optimizer, device, train_dataloader, loss_fn, early_stopping=None,
                val_dataloader=None, scheduler=None):
    train_loss, val_loss = [], []
    accuracy = []
    scaler = torch.amp.GradScaler()

    num_batches = len(train_dataloader)

    for epoch in range(1, epoch + 1):
        pbar = trange(num_batches)
        pbar.set_description(desc=f"Epoch {epoch}")
        epoch_loss = train_loop(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer,
                                device=device, pbar=pbar, scaler=scaler)
        train_loss.append(epoch_loss)

        if val_dataloader is not None:
            epoch_val_loss, epoch_accuracy, _, _ = eval_loop(model=model, dataloader=val_dataloader, loss_fn=loss_fn,
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