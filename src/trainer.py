import torch


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp
                    ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Training accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(dataloader)
    train_acc = correct / total

    return train_loss, train_acc


def evaluate(model, dataloader, criterion, device, use_amp):
    model.eval()
    total_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = total_loss / len(dataloader)
    val_acc = correct / total

    return val_loss, val_acc


def should_stop_early(val_acc, best_acc, counter):
    if val_acc > best_acc:
        return True, 0
    else:
        counter += 1
        return False, counter