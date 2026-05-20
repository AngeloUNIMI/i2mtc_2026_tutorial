import torch
import torch.nn as nn
import torch.optim as optim
import os


def warmup(
    model,
    dataloader_train,
    log,
    cuda,
    dirResults,
    iteration=0,
    criterion=None,
    num_epochs_warmup=5,
    lr_warmup=1e-4,
    weight_decay=5e-4,
):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # File where the warmed-up model is saved
    fileNameWarmup = dirResults / f"modelsave_{iteration + 1}_warmup.pt"

    # If warmup already exists, load and return
    if os.path.isfile(fileNameWarmup):
        if log:
            print(f"\tWarmup model found. Loading: {fileNameWarmup}")

        state = torch.load(fileNameWarmup, map_location=device)
        model.load_state_dict(state)
        return model

    # Prefer same loss as main training.
    # Pass BCEWithLogitsLoss(pos_weight=...) from outside if you use class weights.
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr_warmup / 10,
        weight_decay=weight_decay,
    )

    learning_rate_min = lr_warmup / 10
    learning_rate_max = lr_warmup

    model.train()

    for epoch in range(num_epochs_warmup):
        # Linear warmup: reaches lr_warmup on final epoch
        if num_epochs_warmup > 1:
            lr = learning_rate_min + (learning_rate_max - learning_rate_min) * (
                epoch / (num_epochs_warmup - 1)
            )
        else:
            lr = learning_rate_max

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if log:
            print(f"\tWarmup Epoch {epoch + 1:03d}/{num_epochs_warmup:03d}: lr: {lr:.6f}", end=" ")

        running_loss = 0.0
        total_samples = 0

        for inputs, dummyTargets, filename, label in dataloader_train:
            inputs = inputs.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs.float(), label)

            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        if log:
            epoch_loss = running_loss / total_samples
            print(f"\tWarmup Loss: {epoch_loss:.4f}", end=" ")

        print()

    # Save warmed-up model
    torch.save(model.state_dict(), fileNameWarmup)

    if log:
        print(f"\tWarmup model saved: {fileNameWarmup}")

    return model