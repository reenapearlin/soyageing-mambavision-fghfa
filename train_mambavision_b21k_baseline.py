import argparse
import csv
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from mambavision import create_model

NUM_CLASSES = 198
IMAGE_SIZE = 448
RESIZE_SIZE = 600

_original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load


def read_samples(annotation_file, image_dir):
    samples = []

    with open(annotation_file, "r") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if len(fields) != 2:
                raise ValueError(f"Invalid annotation at {annotation_file}:{line_number}")

            image_name, raw_label = fields
            label = int(raw_label) - 1

            if not 0 <= label < NUM_CLASSES:
                raise ValueError(f"Invalid class label {raw_label} in {annotation_file}")

            image_path = os.path.join(image_dir, image_name)
            if not os.path.isfile(image_path):
                image_path = os.path.join(image_dir, os.path.basename(image_name))

            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Missing image: {image_name}")

            samples.append((image_path, label))

    if len(samples) == 0:
        raise RuntimeError(f"No samples found in {annotation_file}")

    return samples


class SoyAgeingDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


class MambaVisionB21KBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        print("[INFO] Loading NVIDIA MambaVision-B-21K backbone...")
        self.backbone = create_model(
            "mamba_vision_B_21k",
            pretrained=True,
            model_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "pretrained",
                "mambavision_base_21k.pth.tar",
            ),
        )

        feature_dim = self.backbone.head.in_features
        self.backbone.head = nn.Linear(feature_dim, NUM_CLASSES)

    def forward(self, images):
        return self.backbone(images)


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=3e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.abspath(args.base_dir)
    train_file = os.path.join(base_dir, "anno", "train.txt")
    val_file = os.path.join(base_dir, "anno", "val.txt")
    image_dir = os.path.join(base_dir, "images")

    train_samples = read_samples(train_file, image_dir)
    val_samples = read_samples(val_file, image_dir)

    train_classes = {label for _, label in train_samples}
    val_classes = {label for _, label in val_samples}

    if len(train_classes) != NUM_CLASSES or len(val_classes) != NUM_CLASSES:
        raise RuntimeError("Each split must contain all 198 SoyAgeing classes.")

    print("\n" + "=" * 78)
    print("[MAMBAVISION-B-21K BASELINE]")
    print("Dataset:", base_dir)
    print("Train samples:", len(train_samples))
    print("Validation samples:", len(val_samples))
    print("Classes:", len(train_classes))
    print("=" * 78)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20),
        transforms.ToTensor(),
        normalize,
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = DataLoader(
        SoyAgeingDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    val_loader = DataLoader(
        SoyAgeingDataset(val_samples, validation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    model = MambaVisionB21KBaseline().to(device)

    criterion_train = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_val = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("backbone.head.")
                ],
                "lr": args.backbone_lr,
            },
            {
                "params": model.backbone.head.parameters(),
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.30,
        patience=5,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    subset_name = os.path.basename(os.path.normpath(base_dir)).lower()
    output_dir = f"{subset_name}_mambavision_b21k_baseline_results"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        output_dir,
        f"best_{subset_name}_mambavision_b21k_baseline.pth",
    )
    history_path = os.path.join(output_dir, "training_history.csv")

    best_accuracy = -1.0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion_train(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = loss_sum / total
        train_accuracy = 100.0 * correct / total
        val_loss, val_accuracy = evaluate(model, val_loader, criterion_val, device)
        scheduler.step(val_accuracy)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_accuracy,
            "val_loss": val_loss,
            "val_acc": val_accuracy,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train {train_accuracy:.2f}% | Val {val_accuracy:.2f}% | "
            f"ValLoss {val_loss:.4f}"
        )

        with open(history_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_accuracy,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"[SUCCESS] New baseline best: {best_accuracy:.2f}%")
        else:
            stale_epochs += 1

        if stale_epochs >= args.early_stop_patience:
            print(f"[INFO] Early stopping after {stale_epochs} stale epochs.")
            break

    print("\n" + "=" * 78)
    print(f"[DONE] Best baseline validation accuracy: {best_accuracy:.2f}%")
    print(f"[DONE] Checkpoint: {checkpoint_path}")
    print(f"[DONE] History: {history_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
