import torch
import time
import copy
import math
import os
import functions.utils as utils


def multilabel_jaccard_sum(preds, labels, eps=1e-8):
    preds = preds.bool()
    labels = labels.bool()

    intersection = (preds & labels).sum(dim=1).float()
    union = (preds | labels).sum(dim=1).float()

    score = torch.where(union > 0, intersection / (union + eps), torch.ones_like(union))
    return score.sum()


def compute_orth_loss(model, modelName):
    device = next(model.parameters()).device
    diff = torch.zeros((), device=device)

    if modelName == "resnet18":
        diff = utils.orth_dist(model.layer2[0].downsample[0].weight) + \
               utils.orth_dist(model.layer3[0].downsample[0].weight) + \
               utils.orth_dist(model.layer4[0].downsample[0].weight)

        diff = diff + utils.deconv_orth_dist(model.layer1[0].conv1.weight, stride=1) + \
                    utils.deconv_orth_dist(model.layer1[1].conv1.weight, stride=1)

        diff = diff + utils.deconv_orth_dist(model.layer2[0].conv1.weight, stride=2) + \
                    utils.deconv_orth_dist(model.layer2[1].conv1.weight, stride=1)

        diff = diff + utils.deconv_orth_dist(model.layer3[0].conv1.weight, stride=2) + \
                    utils.deconv_orth_dist(model.layer3[1].conv1.weight, stride=1)

        diff = diff + utils.deconv_orth_dist(model.layer4[0].conv1.weight, stride=2) + \
                    utils.deconv_orth_dist(model.layer4[1].conv1.weight, stride=1)

    elif modelName == "resnet34":
        diff = utils.orth_dist(model.layer2[0].downsample[0].weight) + \
               utils.orth_dist(model.layer3[0].downsample[0].weight) + \
               utils.orth_dist(model.layer4[0].downsample[0].weight)

        for i in range(3):
            diff = diff + utils.deconv_orth_dist(model.layer1[i].conv1.weight, stride=1)

        diff = diff + utils.deconv_orth_dist(model.layer2[0].conv1.weight, stride=2)
        for i in range(1, 4):
            diff = diff + utils.deconv_orth_dist(model.layer2[i].conv1.weight, stride=1)

        diff = diff + utils.deconv_orth_dist(model.layer3[0].conv1.weight, stride=2)
        for i in range(1, 6):
            diff = diff + utils.deconv_orth_dist(model.layer3[i].conv1.weight, stride=1)

        diff = diff + utils.deconv_orth_dist(model.layer4[0].conv1.weight, stride=2)
        for i in range(1, 3):
            diff = diff + utils.deconv_orth_dist(model.layer4[i].conv1.weight, stride=1)

    return diff


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_acc,
    min_val_loss,
    best_model_wts,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_acc": best_acc,
        "min_val_loss": min_val_loss,
        "best_model_wts": best_model_wts,
    }
    torch.save(checkpoint, checkpoint_path)


def train_model_val(
    model, classVec,
    optimizer, scheduler, criterion,
    num_epochs, dataset_sizes, dataloader_train, dataloader_val,
    batch_sizeP, num_classes, modelName,
    dirResults, iteration, r_orth, log, cuda,
    checkpoint_every=10
):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    fileNameSaveFinal = dirResults / f"modelsave_{iteration + 1}_final.pt"
    checkpoint_path = dirResults / f"modelsave_{iteration + 1}_checkpoint.pt"

    if os.path.isfile(fileNameSaveFinal):
        if log:
            print("\tFinal model found. Loading final model.")
        state = torch.load(fileNameSaveFinal, map_location=device)
        model.load_state_dict(state)
        return model

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_val_loss = float("inf")
    start_epoch = 0

    # Resume from checkpoint if present
    if os.path.isfile(checkpoint_path):
        if log:
            print(f"\tCheckpoint found. Resuming from: {checkpoint_path}")
        #
        checkpoint = torch.load(checkpoint_path, map_location=device)
        #
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        #
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        min_val_loss = checkpoint.get("min_val_loss", float("inf"))
        best_model_wts = checkpoint.get("best_model_wts", copy.deepcopy(model.state_dict()))
        #
        if log:
            print(f"\tRestarting from epoch {start_epoch + 1}/{num_epochs}")

    dataloaders = {
        "train": dataloader_train,
        "val": dataloader_val,
    }

    numBatches = {
        "train": math.ceil(dataset_sizes["train"] / batch_sizeP),
        "val": math.ceil(dataset_sizes["val"] / batch_sizeP),
    }

    for epoch in range(start_epoch, num_epochs):
        if log:
            print(f"\tEpoch {epoch + 1}/{num_epochs}", end=" ")

        for phase in ["train", "val"]:
            is_train = phase == "train"

            if is_train:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_jaccard = 0.0

            for batch_num, (inputs, dummyTargets, filename, label) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True).float()

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)

                    base_loss = criterion(outputs.float(), label)

                    if is_train and r_orth > 0:
                        diff = compute_orth_loss(model, modelName)
                        loss = base_loss + r_orth * diff
                    else:
                        loss = base_loss

                    if is_train:
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    current_batch_size = inputs.size(0)

                    running_loss += base_loss.item() * current_batch_size

                    preds = (torch.sigmoid(outputs) > 0.5)
                    label_int = label.bool()

                    running_corrects += (preds == label_int).sum().item()
                    running_jaccard += multilabel_jaccard_sum(preds, label_int).item()

            if is_train and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / (dataset_sizes[phase] * num_classes)
            epoch_jaccard = running_jaccard / dataset_sizes[phase]

            if log:
                print(
                    f"\t{phase} Loss: {epoch_loss:.4f}; "
                    f"Acc (1-HL): {epoch_acc:.4f}; "
                    f"Jaccard: {epoch_jaccard:.4f}",
                    end=" "
                )

            if phase == "val":
                if (epoch_acc > best_acc) or (epoch_acc == best_acc and epoch_loss < min_val_loss):
                    best_acc = epoch_acc
                    min_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

        # Save checkpoint every N epochs, and also at the last epoch
        should_save_checkpoint = (
            ((epoch + 1) % checkpoint_every == 0) or
            ((epoch + 1) == num_epochs)
        )

        if should_save_checkpoint:
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=best_acc,
                min_val_loss=min_val_loss,
                best_model_wts=best_model_wts,
            )

            if log:
                print(f"\tCheckpoint saved at epoch {epoch + 1}: {checkpoint_path}")

    time_elapsed = time.time() - since
    print(f"\tTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"\tBest val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), fileNameSaveFinal)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return model
