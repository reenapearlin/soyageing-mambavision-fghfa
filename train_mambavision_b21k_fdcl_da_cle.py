import argparse
import csv
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_mambavision_b21k_fdcl_da import (
    NUM_CLASSES,
    IMAGE_SIZE,
    RESIZE_SIZE,
    PairBalancedBatchSampler,
    RandomPatchShuffle,
    audit_r1_dataset,
    evaluate,
    foreground_background_distillation_loss,
    load_compatible_initialization,
    supervised_contrastive_feature_loss,
    MambaVisionPaperFDCLDA,
)


class CLEPatchShuffleMask:
    """Partial patch shuffle plus optional masking for the CLE-only view."""

    def __init__(self, grid_size=4, shuffle_ratio=0.25, mask_ratio=0.10):
        self.grid_size = grid_size
        self.shuffle_ratio = shuffle_ratio
        self.mask_ratio = mask_ratio

    def __call__(self, image):
        width, height = image.size
        grid = self.grid_size
        patch_w = width // grid
        patch_h = height // grid

        positions = [(row, col) for row in range(grid) for col in range(grid)]
        patches = []

        for row, col in positions:
            left, top = col * patch_w, row * patch_h
            right = width if col == grid - 1 else (col + 1) * patch_w
            bottom = height if row == grid - 1 else (row + 1) * patch_h
            patches.append(image.crop((left, top, right, bottom)))

        count = len(patches)
        shuffle_count = min(
            count, max(0, round(count * self.shuffle_ratio))
        )
        selected = random.sample(range(count), shuffle_count)
        shuffled = [patches[index] for index in selected]
        random.shuffle(shuffled)

        result = Image.new("RGB", (width, height))
        replacement = dict(zip(selected, shuffled))

        mask_count = min(
            count - shuffle_count,
            max(0, round(count * self.mask_ratio)),
        )
        remaining = [index for index in range(count) if index not in selected]
        masked = set(random.sample(remaining, mask_count))

        for index, (row, col) in enumerate(positions):
            left, top = col * patch_w, row * patch_h
            if index in masked:
                patch = Image.new(
                    "RGB",
                    patches[index].size,
                    color=(127, 127, 127),
                )
            else:
                patch = replacement.get(index, patches[index])
            result.paste(patch, (left, top))

        return result


class SoyCLETripleAugmentDataset(Dataset):
    """Existing standard/FDCL-DA views plus an independent CLE-only view."""

    def __init__(
        self,
        samples,
        standard_transform,
        auxiliary_transform,
        cle_transform,
    ):
        self.samples = samples
        self.standard_transform = standard_transform
        self.auxiliary_transform = auxiliary_transform
        self.cle_transform = cle_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return (
            self.standard_transform(image),
            self.auxiliary_transform(image),
            self.cle_transform(image),
            label,
        )


class MambaVisionCLEFDCLDA(MambaVisionPaperFDCLDA):
    """
    Original B-21K MambaVision + unchanged FDCL-DA.
    Only adds a projection head used exclusively by L_CLE.
    """

    def __init__(self, pretrained=True, projection_dim=128):
        super().__init__(pretrained=pretrained)
        self.cle_projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, projection_dim),
        )

    def cle_projection(self, features):
        return F.normalize(self.cle_projector(features), dim=1)


def cle_instance_contrastive_loss(z_normal, z_cle, temperature=0.20):
    """
    Symmetric NT-Xent / InfoNCE.

    Positive: same image normal-view <-> CLE-transformed-view.
    Negatives: every other image instance in the batch, including same-class
    images. This deliberately differs from the existing label-supervised CL.
    """
    if z_normal.shape != z_cle.shape:
        raise ValueError("CLE pair tensors must have identical shape.")

    batch = z_normal.size(0)
    if batch < 2:
        raise ValueError("CLE requires at least two image instances per batch.")

    features = torch.cat([z_normal, z_cle], dim=0)
    logits = features @ features.t() / temperature
    logits.fill_diagonal_(-torch.inf)

    targets = torch.cat([
        torch.arange(batch, 2 * batch, device=features.device),
        torch.arange(0, batch, device=features.device),
    ])

    return F.cross_entropy(logits, targets)


def model_parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
    )


def run_smoke_test(model, device, lambda_fd, lambda_cl, lambda_cle):
    print("\n[TEST] Running CLE forward/backward smoke test...")
    model.train()

    images = torch.randn(2, 3, 448, 448, device=device)
    auxiliary = torch.randn(2, 3, 448, 448, device=device)
    cle_images = torch.randn(2, 3, 448, 448, device=device)
    labels = torch.tensor([0, 0], device=device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        logits, features, stages = model(images)
        auxiliary_logits, _, _ = model(auxiliary)
        _, cle_features, _ = model(cle_images)

        ce_standard = criterion(logits, labels)
        ce_auxiliary = criterion(auxiliary_logits, labels)
        fd_loss = foreground_background_distillation_loss(
            stages, model.fd_classifiers
        )
        existing_cl = supervised_contrastive_feature_loss(features, labels)
        cle_loss = cle_instance_contrastive_loss(
            model.cle_projection(features),
            model.cle_projection(cle_features),
        )

        total = (
            ce_standard
            + ce_auxiliary
            + lambda_fd * fd_loss
            + lambda_cl * existing_cl
            + lambda_cle * cle_loss
        )

    total.backward()

    if not torch.isfinite(total):
        raise RuntimeError("CLE smoke test produced non-finite total loss.")

    projector_has_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.cle_projector.parameters()
    )
    backbone_has_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.backbone.parameters()
    )

    if not projector_has_grad or not backbone_has_grad:
        raise RuntimeError("Expected CLE projector/backbone gradients were absent.")

    expected = [
        (2, 256, 56, 56),
        (2, 512, 28, 28),
        (2, 1024, 14, 14),
        (2, 1024, 14, 14),
    ]
    for index, shape in enumerate(expected):
        if tuple(stages[index].shape) != shape:
            raise RuntimeError(
                f"Stage {index + 1} expected {shape}, got {tuple(stages[index].shape)}"
            )

    print(f"[PASS] CE standard: {ce_standard.item():.6f}")
    print(f"[PASS] CE auxiliary: {ce_auxiliary.item():.6f}")
    print(f"[PASS] FD: {fd_loss.item():.6f}")
    print(f"[PASS] Existing CL: {existing_cl.item():.6f}")
    print(f"[PASS] CLE raw: {cle_loss.item():.6f}")
    print(f"[PASS] CLE weighted: {(lambda_cle * cle_loss).item():.6f}")
    print(f"[PASS] Total: {total.item():.6f}")
    print("[SUCCESS] CLE smoke test passed.")



def evaluate_cle(model, data_loader, criterion, device):
    """Validation for the four-item CLE dataset batch format."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for standard_images, _, _, labels in data_loader:
            standard_images = standard_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _, _ = model(standard_images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="/home/subsea/reena/intern/VMambaFDCL/data/SoyAgeing-R1/R1",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=3e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lambda_fd", type=float, default=1.25)
    parser.add_argument("--lambda_cl", type=float, default=1.0)

    # Conservative pilot: InfoNCE is typically larger than CE/FD/CL.
    parser.add_argument("--lambda_cle", type=float, default=0.05)
    parser.add_argument("--cle_temperature", type=float, default=0.20)
    parser.add_argument("--cle_grid_size", type=int, default=4)
    parser.add_argument("--cle_shuffle_ratio", type=float, default=0.25)
    parser.add_argument("--cle_mask_ratio", type=float, default=0.10)
    parser.add_argument("--early_stop_patience", type=int, default=25)
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.cle_shuffle_ratio <= 1.0:
        raise ValueError("--cle_shuffle_ratio must be in [0, 1].")
    if not 0.0 <= args.cle_mask_ratio <= 1.0:
        raise ValueError("--cle_mask_ratio must be in [0, 1].")
    if args.cle_shuffle_ratio + args.cle_mask_ratio > 1.0:
        raise ValueError("CLE shuffle_ratio + mask_ratio must not exceed 1.")

    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = MambaVisionCLEFDCLDA(pretrained=True).to(device)
    print(f"[INFO] Total initialized parameters: {model_parameter_count(model):,}")
    print(f"[INFO] CLE lambda: {args.lambda_cle}")
    print(
        f"[INFO] CLE view: grid={args.cle_grid_size}, "
        f"shuffle={args.cle_shuffle_ratio}, mask={args.cle_mask_ratio}, "
        f"temperature={args.cle_temperature}"
    )

    if args.smoke_test:
        run_smoke_test(
            model, device, args.lambda_fd, args.lambda_cl, args.lambda_cle
        )
        return

    train_samples, val_samples = audit_r1_dataset(args.base_dir)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Original FDCL normal view: unchanged.
    standard_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.20),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), value=0),
    ])

    # Original FDCL auxiliary view: unchanged.
    auxiliary_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        RandomPatchShuffle(grid_size=4, probability=1.0),
        transforms.ToTensor(),
        normalize,
    ])

    # New CLE-only third view.
    cle_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        CLEPatchShuffleMask(
            grid_size=args.cle_grid_size,
            shuffle_ratio=args.cle_shuffle_ratio,
            mask_ratio=args.cle_mask_ratio,
        ),
        transforms.ToTensor(),
        normalize,
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = SoyCLETripleAugmentDataset(
        train_samples,
        standard_transform,
        auxiliary_transform,
        cle_transform,
    )
    val_dataset = SoyCLETripleAugmentDataset(
        val_samples,
        validation_transform,
        validation_transform,
        validation_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PairBalancedBatchSampler(train_samples, args.batch_size),
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    loaded_best_accuracy = load_compatible_initialization(
        model, args.init_checkpoint, device
    )

    train_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    validation_ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {
                "params": (
                    list(model.classifier.parameters())
                    + list(model.fd_classifiers.parameters())
                    + list(model.cle_projector.parameters())
                ),
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.30, patience=5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    subset_name = os.path.basename(os.path.normpath(args.base_dir)).lower()
    output_dir = f"{subset_name}_mambavision_b21k_fdcl_da_cle_results"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        output_dir,
        f"best_{subset_name}_mambavision_b21k_fdcl_da_cle.pth",
    )
    history_path = os.path.join(output_dir, "training_history.csv")
    config_path = os.path.join(output_dir, "experiment_config.json")

    with open(config_path, "w") as handle:
        json.dump(vars(args), handle, indent=2)

    best_accuracy = loaded_best_accuracy
    stale_epochs = 0
    history = []

    print("\n[INFO] Loss = CE_standard + CE_auxiliary + "
          f"{args.lambda_fd}*FD + {args.lambda_cl}*CL + "
          f"{args.lambda_cle}*CLE\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {
            key: 0.0 for key in (
                "total", "ce_standard", "ce_auxiliary", "fd", "cl", "cle"
            )
        }
        correct = total = 0

        for standard, auxiliary, cle_view, labels in train_loader:
            standard = standard.to(device, non_blocking=True)
            auxiliary = auxiliary.to(device, non_blocking=True)
            cle_view = cle_view.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=torch.cuda.is_available(),
            ):
                logits, features, stage_features = model(standard)
                auxiliary_logits, _, _ = model(auxiliary)
                _, cle_features, _ = model(cle_view)

                ce_standard = train_ce(logits, labels)
                ce_auxiliary = train_ce(auxiliary_logits, labels)
                fd_loss = foreground_background_distillation_loss(
                    stage_features, model.fd_classifiers
                )
                existing_cl = supervised_contrastive_feature_loss(features, labels)
                cle_loss = cle_instance_contrastive_loss(
                    model.cle_projection(features),
                    model.cle_projection(cle_features),
                    temperature=args.cle_temperature,
                )

                loss = (
                    ce_standard + ce_auxiliary
                    + args.lambda_fd * fd_loss
                    + args.lambda_cl * existing_cl
                    + args.lambda_cle * cle_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            batch = labels.size(0)
            sums["total"] += loss.item() * batch
            sums["ce_standard"] += ce_standard.item() * batch
            sums["ce_auxiliary"] += ce_auxiliary.item() * batch
            sums["fd"] += fd_loss.item() * batch
            sums["cl"] += existing_cl.item() * batch
            sums["cle"] += cle_loss.item() * batch
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch

        val_loss, val_accuracy = evaluate_cle(model, val_loader, validation_ce, device)
        scheduler.step(val_accuracy)

        row = {
            "epoch": epoch,
            "train_loss": sums["total"] / total,
            "train_acc": 100.0 * correct / total,
            "val_loss": val_loss,
            "val_acc": val_accuracy,
            "ce_standard": sums["ce_standard"] / total,
            "ce_auxiliary": sums["ce_auxiliary"] / total,
            "fd_loss": sums["fd"] / total,
            "cl_loss": sums["cl"] / total,
            "cle_loss_raw": sums["cle"] / total,
            "cle_loss_weighted": args.lambda_cle * sums["cle"] / total,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train {row['train_acc']:.2f}% | Val {val_accuracy:.2f}% | "
            f"CE {(row['ce_standard'] + row['ce_auxiliary']):.4f} | "
            f"FD {row['fd_loss']:.4f} | CL {row['cl_loss']:.4f} | "
            f"CLE {row['cle_loss_raw']:.4f} "
            f"(weighted {row['cle_loss_weighted']:.4f}) | "
            f"Total {row['train_loss']:.4f}"
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
                    "parameter_count": model_parameter_count(model),
                },
                checkpoint_path,
            )
            print(f"[SUCCESS] New best checkpoint: {best_accuracy:.2f}%")
        else:
            stale_epochs += 1

        if stale_epochs >= args.early_stop_patience:
            print(
                f"[INFO] Early stopping after {args.early_stop_patience} "
                "epochs without validation improvement."
            )
            break

    report = {
        "baseline_r1_accuracy": 85.86,
        "best_validation_accuracy": best_accuracy,
        "best_checkpoint": checkpoint_path,
        "parameter_count": model_parameter_count(model),
        "history_file": history_path,
    }
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "=" * 78)
    print(f"[DONE] Baseline R1: 85.86%")
    print(f"[DONE] CLE best validation accuracy: {best_accuracy:.2f}%")
    print(f"[DONE] Difference: {best_accuracy - 85.86:+.2f} percentage points")
    print(f"[DONE] Checkpoint: {checkpoint_path}")
    print(f"[DONE] History: {history_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
