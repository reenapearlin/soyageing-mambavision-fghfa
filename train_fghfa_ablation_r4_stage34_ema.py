import argparse
import csv
import json
import os
import random
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.fghfa_mambavision import MambaVisionFGHFAFDCLDA
from ema_utils import ModelEMA
from train_mambavision_b21k_fdcl_da import (
    IMAGE_SIZE,
    RESIZE_SIZE,
    PairBalancedBatchSampler,
    RandomPatchShuffle,
    SoyDualAugmentDataset,
    audit_r1_dataset as _base_dataset_audit,
    evaluate,
    foreground_background_distillation_loss,
    supervised_contrastive_feature_loss,
)

def checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_baseline_into_fghfa(model, checkpoint_path, device):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "FGHFA requires the trained B-21K FDCL-DA baseline checkpoint."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state = checkpoint_state(checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}

    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)

    allowed_missing = {
        key for key in model.state_dict()
        if key.startswith("fghfa.")
    }
    invalid_missing = missing - allowed_missing

    if invalid_missing or unexpected:
        raise RuntimeError(
            "Unsafe B-21K baseline-to-FGHFA checkpoint transfer.\n"
            f"Unexpected missing keys: {sorted(invalid_missing)}\n"
            f"Unexpected checkpoint keys: {sorted(unexpected)}"
        )

    baseline_accuracy = float(checkpoint.get("val_acc", -1.0))
    print("[SUCCESS] Loaded B-21K FDCL-DA baseline into FGHFA model.")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"[INFO] Baseline validation accuracy: {baseline_accuracy:.2f}%")
    print(f"[INFO] Preserved checkpoint tensors: {len(state)}")
    print(f"[INFO] Newly initialized FGHFA tensors: {len(missing)}")
    print(f"[INFO] Missing keys: {sorted(missing)}")
    return baseline_accuracy



def audit_r4_dataset(base_dir):
    root = Path(base_dir).resolve()
    if root.name != "SoyAgeing-R4":
        raise RuntimeError(f"Refusing non-R4 dataset root: {root}")

    # Reuse the project’s existing parser/path validator, but suppress its
    # old R1-only display label and print a truthful R4 audit below.
    with redirect_stdout(StringIO()):
        train_samples, val_samples = _base_dataset_audit(str(root))

    train_labels = {label for _, label in train_samples}
    val_labels = {label for _, label in val_samples}

    for image_path, _ in train_samples + val_samples:
        if root not in Path(image_path).resolve().parents:
            raise RuntimeError(f"Non-R4 image path detected: {image_path}")

    overlap = {
        Path(path).resolve()
        for path, _ in train_samples
    } & {
        Path(path).resolve()
        for path, _ in val_samples
    }

    if (len(train_samples), len(val_samples), len(train_labels), len(val_labels), len(overlap)) != (990, 990, 198, 198, 0):
        raise RuntimeError(
            "R4 audit failed: "
            f"train={len(train_samples)}, val={len(val_samples)}, "
            f"classes={len(train_labels)}/{len(val_labels)}, overlap={len(overlap)}"
        )

    print("\n" + "=" * 78)
    print("Subset: R4")
    print("Dataset: data/SoyAgeing-R4")
    print("Training samples: 990")
    print("Validation samples: 990")
    print("Training classes: 198/198")
    print("Validation classes: 198/198")
    print("Train/val overlap: 0")
    print("[SUCCESS] Correct SoyAgeing-R4/R4 split, labels, and image paths verified.")
    print("=" * 78)
    return train_samples, val_samples


def subset_name_from_path(base_dir):
    match = re.search(r"SoyAgeing-(R[1-6])", os.path.normpath(base_dir))
    if not match:
        raise ValueError(f"Could not determine subset from: {base_dir}")
    return match.group(1).lower()


def count_parameters(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
    )


def smoke_test(model, device, args):
    print("\n[TEST] FGHFA shape, baseline-equivalence, FDCL, and backward test...")
    model.eval()
    images = torch.randn(2, 3, 448, 448, device=device)

    # FGHFA disabled: exact original classifier path.
    model.set_fghfa_enabled(False)
    with torch.no_grad():
        baseline_logits, baseline_features, baseline_stages = model(images)

    expected = [
        (2, 256, 56, 56),
        (2, 512, 28, 28),
        (2, 1024, 14, 14),
        (2, 1024, 14, 14),
    ]
    for index, shape in enumerate(expected):
        actual = tuple(baseline_stages[index].shape)
        if actual != shape:
            raise RuntimeError(f"Stage {index + 1}: expected {shape}, got {actual}")
        print(f"[PASS] Stage {index + 1}: {actual}")

    if tuple(baseline_features.shape) != (2, 1024):
        raise RuntimeError("Final baseline feature is not B,1024.")

    # Gate zero must mathematically reproduce the original classifier output.
    original_gate = model.fghfa.residual_gate.detach().clone()
    model.set_fghfa_enabled(True)
    with torch.no_grad():
        model.fghfa.residual_gate.zero_()
        zero_gate_logits, _, _ = model(images)
    max_difference = (baseline_logits - zero_gate_logits).abs().max().item()
    if max_difference > 1e-5:
        raise RuntimeError(
            f"FGHFA zero-gate baseline mismatch: max difference {max_difference}"
        )
    with torch.no_grad():
        model.fghfa.residual_gate.copy_(original_gate)
    print(f"[PASS] Zero-gate baseline equivalence: max diff {max_difference:.8f}")

    model.train()
    labels = torch.tensor([0, 0], device=device)
    auxiliary = torch.randn(2, 3, 448, 448, device=device)
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

        ce_standard = criterion(logits, labels)
        ce_auxiliary = criterion(auxiliary_logits, labels)
        fd_loss = foreground_background_distillation_loss(
            stages, model.fd_classifiers
        )
        cl_loss = supervised_contrastive_feature_loss(features, labels)
        total = (
            ce_standard + ce_auxiliary
            + args.lambda_fd * fd_loss
            + args.lambda_cl * cl_loss
        )

    total.backward()

    fghfa_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.fghfa.parameters()
    )
    if not fghfa_grad:
        raise RuntimeError("FGHFA parameters did not receive finite gradients.")

    print(f"[PASS] CE: {(ce_standard + ce_auxiliary).item():.6f}")
    print(f"[PASS] FD: {fd_loss.item():.6f}")
    print(f"[PASS] Existing CL: {cl_loss.item():.6f}")
    print(f"[PASS] Total: {total.item():.6f}")
    print(f"[PASS] FGHFA information: {model.fghfa.last_info}")
    print("[SUCCESS] FGHFA smoke test passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=3e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lambda_fd", type=float, default=1.25)
    parser.add_argument("--lambda_cl", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=25)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--fghfa_enabled", action="store_true")
    parser.add_argument("--fghfa_stages", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--fghfa_fusion_dim", type=int, default=1024)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--ema_enabled", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = MambaVisionFGHFAFDCLDA(
        pretrained=True,
        fghfa_enabled=args.fghfa_enabled,
        fghfa_stages=tuple(args.fghfa_stages),
        fusion_dim=args.fghfa_fusion_dim,
    ).to(device)

    baseline_accuracy = load_baseline_into_fghfa(
        model, args.init_checkpoint, device
    )

    print(f"[INFO] Total parameters: {count_parameters(model):,}")
    print(f"[INFO] New FGHFA parameters: {count_parameters(model.fghfa):,}")
    print(f"[INFO] FGHFA enabled: {args.fghfa_enabled}")
    print(f"[INFO] FGHFA stages: {args.fghfa_stages}")
    print(f"[INFO] Gate initialization: {model.fghfa.residual_gate.item():.6f}")

    if args.smoke_test:
        smoke_test(model, device, args)
        return

    train_samples, val_samples = audit_r4_dataset(args.base_dir)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
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
    auxiliary_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        RandomPatchShuffle(grid_size=4, probability=1.0),
        transforms.ToTensor(),
        normalize,
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = SoyDualAugmentDataset(
        train_samples, standard_transform, auxiliary_transform
    )
    val_dataset = SoyDualAugmentDataset(
        val_samples, validation_transform, validation_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PairBalancedBatchSampler(train_samples, args.batch_size),
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
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
                    + list(model.fghfa.parameters())
                ),
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.30, patience=5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay) if args.ema_enabled else None
    if ema is not None:
        print(f"[INFO] EMA enabled: decay={ema.decay}, tracked tensors={len(ema.tracked_names)}")

    subset = subset_name_from_path(args.base_dir)
    output_dir = os.path.join("experiments", "soyageing_fghfa34_ema", subset.upper())
    os.makedirs(output_dir, exist_ok=True)
    ordinary_checkpoint_path = os.path.join(output_dir, "best_ordinary_model.pth")
    ema_checkpoint_path = os.path.join(output_dir, "best_ema_model.pth")
    last_checkpoint_path = os.path.join(output_dir, "last_checkpoint.pth")
    history_path = os.path.join(output_dir, "training_history.csv")

    with open(os.path.join(output_dir, "experiment_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    best_ordinary_accuracy = float("-inf")
    best_ema_accuracy = float("-inf")
    best_ordinary_epoch = 0
    best_ema_epoch = 0
    stale_epochs = 0
    history = []

    print("\n[INFO] FGHFA loss remains unchanged:")
    print(f"CE_standard + CE_auxiliary + {args.lambda_fd}*FD + "
          f"{args.lambda_cl}*existing_CL\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {key: 0.0 for key in ("total", "ce", "fd", "cl")}
        correct = total = 0

        for standard, auxiliary, labels in train_loader:
            standard = standard.to(device, non_blocking=True)
            auxiliary = auxiliary.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits, features, stages = model(standard)
                auxiliary_logits, _, _ = model(auxiliary)

                ce_standard = train_ce(logits, labels)
                ce_auxiliary = train_ce(auxiliary_logits, labels)
                fd_loss = foreground_background_distillation_loss(
                    stages, model.fd_classifiers
                )
                cl_loss = supervised_contrastive_feature_loss(features, labels)
                loss = (
                    ce_standard + ce_auxiliary
                    + args.lambda_fd * fd_loss
                    + args.lambda_cl * cl_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Update EMA only when AMP did not skip the optimizer step.
            if ema is not None and scaler.get_scale() >= scale_before:
                ema.update(model)

            batch = labels.size(0)
            sums["total"] += loss.item() * batch
            sums["ce"] += (ce_standard.item() + ce_auxiliary.item()) * batch
            sums["fd"] += fd_loss.item() * batch
            sums["cl"] += cl_loss.item() * batch
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch

        val_loss, val_accuracy = evaluate(model, val_loader, validation_ce, device)
        if ema is not None:
            with ema.average_parameters(model):
                ema_val_loss, ema_val_accuracy = evaluate(
                    model, val_loader, validation_ce, device
                )
        else:
            ema_val_loss, ema_val_accuracy = val_loss, val_accuracy

        # Preserve the original scheduler criterion.
        scheduler.step(val_accuracy)

        row = {
            "epoch": epoch,
            "train_loss": sums["total"] / total,
            "train_acc": 100.0 * correct / total,
            "val_loss": val_loss,
            "val_acc": val_accuracy,
            "ce_loss": sums["ce"] / total,
            "fd_loss": sums["fd"] / total,
            "cl_loss": sums["cl"] / total,
            "ema_val_loss": ema_val_loss,
            "ema_val_acc": ema_val_accuracy,
            "ema_updates": 0 if ema is None else ema.num_updates,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gate": float(torch.tanh(model.fghfa.residual_gate).detach().cpu()),
            "stage_weights": "|".join(
                f"{value:.6f}"
                for value in torch.softmax(
                    model.fghfa.stage_logits.detach().cpu(), dim=0
                ).tolist()
            ),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train {row['train_acc']:.2f}% | Val {val_accuracy:.2f}% | "
            f"CE {row['ce_loss']:.4f} | FD {row['fd_loss']:.4f} | "
            f"CL {row['cl_loss']:.4f} | EMA Val {ema_val_accuracy:.2f}% | "
            f"Gate {row['gate']:.5f} | StageW [{row['stage_weights']}]"
        )

        with open(history_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        common = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_acc": val_accuracy,
            "val_loss": val_loss,
            "ema_val_acc": ema_val_accuracy,
            "ema_val_loss": ema_val_loss,
            "args": vars(args),
            "fghfa_gate": row["gate"],
            "fghfa_stage_weights": row["stage_weights"],
            "ema_state_dict": None if ema is None else ema.model_state_dict(),
            "ema_metadata": None if ema is None else {
                "decay": ema.decay,
                "num_updates": ema.num_updates,
                "tracked_names": ema.tracked_names,
            },
        }
        torch.save(common, last_checkpoint_path)

        if val_accuracy > best_ordinary_accuracy:
            best_ordinary_accuracy = val_accuracy
            best_ordinary_epoch = epoch
            torch.save(
                {**common, "checkpoint_kind": "ordinary"},
                ordinary_checkpoint_path,
            )
            print(f"[SUCCESS] New ordinary best: {best_ordinary_accuracy:.2f}%")

        if ema_val_accuracy > best_ema_accuracy:
            best_ema_accuracy = ema_val_accuracy
            best_ema_epoch = epoch
            ema_payload = {
                **common,
                "checkpoint_kind": "ema",
                "ordinary_model_state_dict": model.state_dict(),
                "model_state_dict": (
                    model.state_dict() if ema is None else ema.model_state_dict()
                ),
                "val_acc": ema_val_accuracy,
                "val_loss": ema_val_loss,
            }
            torch.save(ema_payload, ema_checkpoint_path)
            print(f"[SUCCESS] New EMA best: {best_ema_accuracy:.2f}%")

        selection_accuracy = ema_val_accuracy if ema is not None else val_accuracy
        selection_best = best_ema_accuracy if ema is not None else best_ordinary_accuracy
        if selection_accuracy >= selection_best:
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= args.early_stop_patience:
            print(f"[INFO] Early stopping after {stale_epochs} stale epochs.")
            break

    print("\n" + "=" * 78)
    print(f"[DONE] Reference FGHFA {subset.upper()}: {baseline_accuracy:.2f}%")
    print(f"[DONE] Ordinary best: {best_ordinary_accuracy:.2f}% at epoch {best_ordinary_epoch}")
    print(f"[DONE] EMA best: {best_ema_accuracy:.2f}% at epoch {best_ema_epoch}")
    print(f"[DONE] Ordinary checkpoint: {ordinary_checkpoint_path}")
    print(f"[DONE] EMA checkpoint: {ema_checkpoint_path}")
    print(f"[DONE] Last checkpoint: {last_checkpoint_path}")
    print(f"[DONE] History: {history_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
