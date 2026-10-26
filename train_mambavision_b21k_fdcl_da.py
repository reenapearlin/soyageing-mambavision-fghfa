import argparse
import csv
import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
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


class RandomPatchShuffle:
    """Paper-style auxiliary augmentation: shuffle equally sized image patches."""

    def __init__(self, grid_size=4, probability=1.0):
        self.grid_size = grid_size
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        width, height = image.size
        grid = self.grid_size
        patch_w = width // grid
        patch_h = height // grid

        patches = []
        for row in range(grid):
            for col in range(grid):
                left = col * patch_w
                top = row * patch_h
                right = width if col == grid - 1 else (col + 1) * patch_w
                bottom = height if row == grid - 1 else (row + 1) * patch_h
                patches.append(image.crop((left, top, right, bottom)))

        random.shuffle(patches)
        shuffled = Image.new("RGB", (width, height))

        index = 0
        for row in range(grid):
            for col in range(grid):
                left = col * patch_w
                top = row * patch_h
                shuffled.paste(patches[index], (left, top))
                index += 1

        return shuffled


def resolve_image_path(img_dir, image_name):
    candidate = os.path.join(img_dir, image_name)
    if os.path.exists(candidate):
        return candidate

    candidate = os.path.join(img_dir, os.path.basename(image_name))
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Image listed in annotation was not found: {image_name}\n"
        f"Checked: {os.path.join(img_dir, image_name)}"
    )


def read_annotation(anno_file, img_dir):
    if not os.path.isfile(anno_file):
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")

    samples = []
    raw_labels = []
    missing_images = []

    with open(anno_file, "r") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(
                    f"{anno_file}:{line_number} must contain exactly "
                    f"'image_name label', but got: {line.strip()}"
                )

            image_name, raw_label_text = parts
            raw_label = int(raw_label_text)
            raw_labels.append(raw_label)

            try:
                image_path = resolve_image_path(img_dir, image_name)
            except FileNotFoundError:
                missing_images.append(image_name)
                continue

            samples.append((image_path, raw_label - 1))

    if not samples:
        raise RuntimeError(f"No valid samples were found in {anno_file}")

    if min(raw_labels) != 1 or max(raw_labels) != NUM_CLASSES:
        raise ValueError(
            f"{anno_file} has labels [{min(raw_labels)}, {max(raw_labels)}], "
            f"but SoyAgeing R1 must use raw labels 1..{NUM_CLASSES}."
        )

    invalid = [label for label in raw_labels if not 1 <= label <= NUM_CLASSES]
    if invalid:
        raise ValueError(f"Invalid labels in {anno_file}: {invalid[:10]}")

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images listed in {anno_file} are missing. "
            f"Examples: {missing_images[:10]}"
        )

    return samples, raw_labels


def audit_r1_dataset(base_dir):
    normalized_path = os.path.normpath(base_dir)
    if "SoyAgeing-R" not in normalized_path or not any(
            normalized_path.endswith(f"SoyAgeing-R{split_id}")
            or normalized_path.endswith(os.path.join(f"SoyAgeing-R{split_id}", f"R{split_id}"))
            for split_id in range(1, 7)
        ):
        raise ValueError(
            "This script requires a SoyAgeing subset directory ending in SoyAgeing-R#/R#. "
            f"Expected a path ending in SoyAgeing-R1/R1, got: {base_dir}"
        )

    image_dir = os.path.join(base_dir, "images")
    train_file = os.path.join(base_dir, "anno", "train.txt")
    val_file = os.path.join(base_dir, "anno", "val.txt")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"R1 images folder not found: {image_dir}")

    train_samples, train_raw = read_annotation(train_file, image_dir)
    val_samples, val_raw = read_annotation(val_file, image_dir)

    train_names = {path for path, _ in train_samples}
    val_names = {path for path, _ in val_samples}
    overlap = train_names.intersection(val_names)

    train_classes = set(train_raw)
    val_classes = set(val_raw)
    expected_classes = set(range(1, NUM_CLASSES + 1))

    print("\n" + "=" * 78)
    print("[SOYAGEING SUBSET DATASET AUDIT]")
    print(f"Base directory: {base_dir}")
    print(f"Training samples:   {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Training classes:   {len(train_classes)}/{NUM_CLASSES}")
    print(f"Validation classes: {len(val_classes)}/{NUM_CLASSES}")
    print(f"Train/val overlap:  {len(overlap)}")

    if train_classes != expected_classes:
        raise ValueError("Training annotations do not cover raw labels 1..198.")
    if val_classes != expected_classes:
        raise ValueError("Validation annotations do not cover raw labels 1..198.")
    if overlap:
        raise ValueError(
            f"R1 train/validation leakage detected: {len(overlap)} duplicate images. "
            f"Examples: {list(overlap)[:5]}"
        )

    print("[SUCCESS] Correct SoyAgeing-R1/R1 split, labels, and image paths verified.")
    print("=" * 78 + "\n")
    return train_samples, val_samples


class SoyDualAugmentDataset(Dataset):
    def __init__(self, samples, standard_transform, auxiliary_transform):
        self.samples = samples
        self.standard_transform = standard_transform
        self.auxiliary_transform = auxiliary_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        standard_view = self.standard_transform(image)
        auxiliary_view = self.auxiliary_transform(image)
        return standard_view, auxiliary_view, label


class PairBalancedBatchSampler(Sampler):
    """
    Each batch contains two images from each selected class.
    This guarantees positive same-class pairs for the CL loss.
    """

    def __init__(self, samples, batch_size):
        if batch_size < 4 or batch_size % 2 != 0:
            raise ValueError("batch_size must be even and at least 4.")

        self.batch_size = batch_size
        self.classes_per_batch = batch_size // 2
        self.class_to_indices = {}

        for index, (_, label) in enumerate(samples):
            self.class_to_indices.setdefault(label, []).append(index)

        self.classes = sorted(self.class_to_indices)
        if len(self.classes) < self.classes_per_batch:
            raise ValueError("Not enough classes for balanced contrastive batches.")

        self.num_batches = len(samples) // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            selected_classes = random.sample(self.classes, self.classes_per_batch)
            batch = []

            for label in selected_classes:
                indices = self.class_to_indices[label]
                if len(indices) >= 2:
                    batch.extend(random.sample(indices, 2))
                else:
                    batch.extend([indices[0], indices[0]])

            random.shuffle(batch)
            yield batch


class MambaVisionPaperFDCLDA(nn.Module):
    """
    MambaVision-B adaptation of FDCL-DA:
    - standard and patch-shuffled auxiliary views
    - multi-stage spatial classifiers for FD
    - final-stage feature representation for CL
    """

    def __init__(self, pretrained=True):
        super().__init__()
        print("Loading NVIDIA MambaVision-B-21K...")
        self.backbone = create_model(
            "mamba_vision_B_21k",
            pretrained=pretrained,
            model_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "pretrained",
                "mambavision_base_21k.pth.tar",
            ),
        )

        if not hasattr(self.backbone, "head") or not hasattr(self.backbone, "levels"):
            raise RuntimeError("Installed MambaVision package does not expose head and levels.")

        feature_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(feature_dim, NUM_CLASSES),
        )

        # One spatial classifier per backbone stage. Lazy layers infer stage channels
        # from MambaVision during the first CUDA forward pass.
        self.fd_classifiers = nn.ModuleList(
            [nn.LazyConv2d(NUM_CLASSES, kernel_size=1) for _ in range(4)]
        )
        self._stage_features = []
        self._hook_handles = []

        for level in self.backbone.levels:
            self._hook_handles.append(level.register_forward_hook(self._capture_stage))

    def _capture_stage(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            self._stage_features.append(output)

    @staticmethod
    def _as_nchw(feature):
        if feature.ndim != 4:
            raise RuntimeError(f"Expected 4D stage feature map, got {tuple(feature.shape)}")

        # MambaVision normally emits B,C,H,W. This also supports B,H,W,C.
        if feature.shape[1] <= 64 and feature.shape[-1] > feature.shape[1]:
            feature = feature.permute(0, 3, 1, 2).contiguous()

        return feature

    @staticmethod
    def _pool(features):
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.ndim == 4:
            return features.mean(dim=(2, 3))
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise RuntimeError(f"Unexpected backbone output shape: {tuple(features.shape)}")

    def forward(self, images):
        self._stage_features = []
        final_features = self.backbone(images)
        stage_features = [self._as_nchw(item) for item in self._stage_features]

        if len(stage_features) != 4:
            raise RuntimeError(
                f"Expected four MambaVision stages, received {len(stage_features)}."
            )

        pooled_features = self._pool(final_features)
        logits = self.classifier(pooled_features)
        return logits, pooled_features, stage_features


def foreground_background_distillation_loss(stage_features, stage_classifiers):
    """
    Stable foreground-background loss.

    Foreground positions are top-K locations for the predicted class.
    All other locations are background and are pushed below logit -1.
    Softplus keeps a useful gradient where HardTanh previously saturated.
    """
    selected_counts = (256, 128, 64, 32)
    total_loss = stage_features[0].new_zeros(())

    for feature_map, classifier, select_count in zip(
        stage_features, stage_classifiers, selected_counts
    ):
        class_maps = classifier(feature_map)
        batch_size = class_maps.shape[0]

        token_logits = class_maps.flatten(2).transpose(1, 2).contiguous()
        probabilities = torch.softmax(token_logits, dim=-1)
        image_probabilities = torch.softmax(token_logits.mean(dim=1), dim=-1)
        predicted_class = image_probabilities.argmax(dim=1)

        batch_indices = torch.arange(batch_size, device=class_maps.device)
        predicted_probabilities = probabilities[batch_indices, :, predicted_class]

        token_count = predicted_probabilities.shape[1]
        k = min(select_count, token_count - 1)
        sorted_indices = predicted_probabilities.argsort(dim=1, descending=True)
        background_indices = sorted_indices[:, k:]

        predicted_logits = token_logits[batch_indices, :, predicted_class]
        background_logits = torch.gather(
            predicted_logits,
            dim=1,
            index=background_indices,
        )

        # Non-saturating equivalent of forcing background activations to -1.
        total_loss = total_loss + F.softplus(background_logits + 1.0).mean()

    return total_loss / len(stage_features)

def supervised_contrastive_feature_loss(features, labels, margin=0.20):
    """
    Paper Eq. 7:
    same-label features move together; different-label features with
    similarity above alpha/margin are penalized.
    """
    features = F.normalize(features, dim=1)
    similarity = features @ features.t()

    same_label = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(labels.numel(), device=labels.device, dtype=torch.bool)
    same_label = same_label & ~diagonal
    different_label = ~same_label & ~diagonal

    if same_label.any():
        positive_loss = (1.0 - similarity[same_label]).mean()
    else:
        positive_loss = similarity.new_zeros(())

    if different_label.any():
        negative_loss = F.relu(similarity[different_label] - margin).mean()
    else:
        negative_loss = similarity.new_zeros(())

    return positive_loss + negative_loss


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Checkpoint does not contain a model state dictionary.")


def load_compatible_initialization(model, checkpoint_path, device):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print("[INFO] No fine-tuning checkpoint found. Using ImageNet-1K initialization.")
        return -1.0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = get_state_dict(checkpoint)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    is_paper_aligned_checkpoint = (
        "mambavision_b21k_paper_fdcl_da_results"
        in os.path.normpath(checkpoint_path)
    )

    if is_paper_aligned_checkpoint:
        # Same architecture: restore backbone, classifier, and trained FD heads.
        load_state = state_dict
        print("[INFO] Full resume from the paper-aligned FDCL-DA checkpoint.")
    else:
        # Older checkpoint: transfer only compatible backbone/final classifier.
        load_state = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("fd_classifiers.")
            and not key.startswith("fdcl_head.")
        }
        print("[INFO] Old checkpoint detected: FD heads intentionally skipped.")

    incompatible = model.load_state_dict(load_state, strict=False)
    loaded_accuracy = float(checkpoint.get("val_acc", -1.0))

    print(f"[INFO] Initialized from: {checkpoint_path}")
    print(f"[INFO] Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"[INFO] Checkpoint validation accuracy: {loaded_accuracy:.2f}%")
    print(f"[INFO] Missing keys: {len(incompatible.missing_keys)}")
    print(f"[INFO] Ignored keys: {len(incompatible.unexpected_keys)}")

    return loaded_accuracy

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for standard_images, _, labels in data_loader:
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
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=3e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lambda_fd", type=float, default=1.25)
    parser.add_argument("--lambda_cl", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument(
        "--init_checkpoint",
        default="r1_mambavision_b21k_fdcl_da_results/best_r1_mambavision_b21k_fdcl_da.pth",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_samples, val_samples = audit_r1_dataset(args.base_dir)

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

    balanced_batch_sampler = PairBalancedBatchSampler(
        train_samples,
        batch_size=args.batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=balanced_batch_sampler,
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

    model = MambaVisionPaperFDCLDA(pretrained=True).to(device)
    loaded_best_accuracy = load_compatible_initialization(
        model, args.init_checkpoint, device
    )

    train_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    validation_ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {
                "params": list(model.classifier.parameters()) +
                          list(model.fd_classifiers.parameters()),
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

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    subset_name = os.path.basename(os.path.normpath(args.base_dir)).lower()
    output_dir = f"{subset_name}_mambavision_b21k_paper_fdcl_da_results"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        output_dir,
        f"best_{subset_name}_mambavision_b21k_paper_fdcl_da.pth",
    )
    history_path = os.path.join(output_dir, "training_history.csv")

    best_accuracy = loaded_best_accuracy
    epochs_without_improvement = 0
    history = []

    print("\n[INFO] Starting paper-aligned FDCL-DA fine-tuning...")
    print(f"[INFO] Backbone LR: {args.backbone_lr}")
    print(f"[INFO] Head LR: {args.head_lr}")
    print(f"[INFO] Loss = CE_standard + CE_auxiliary + "
          f"{args.lambda_fd}*FD + {args.lambda_cl}*CL\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_ce_standard = 0.0
        running_ce_auxiliary = 0.0
        running_fd = 0.0
        running_cl = 0.0
        correct = 0
        total = 0

        for standard_images, auxiliary_images, labels in train_loader:
            standard_images = standard_images.to(device, non_blocking=True)
            auxiliary_images = auxiliary_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=torch.cuda.is_available(),
            ):
                standard_logits, standard_features, stage_features = model(standard_images)
                auxiliary_logits, _, _ = model(auxiliary_images)

                ce_standard = train_ce(standard_logits, labels)
                ce_auxiliary = train_ce(auxiliary_logits, labels)
                fd_loss = foreground_background_distillation_loss(
                    stage_features, model.fd_classifiers
                )
                cl_loss = supervised_contrastive_feature_loss(
                    standard_features, labels
                )

                loss = (
                    ce_standard
                    + ce_auxiliary
                    + args.lambda_fd * fd_loss
                    + args.lambda_cl * cl_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_ce_standard += ce_standard.item() * batch_size
            running_ce_auxiliary += ce_auxiliary.item() * batch_size
            running_fd += fd_loss.item() * batch_size
            running_cl += cl_loss.item() * batch_size
            correct += (standard_logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total
        val_loss, val_accuracy = evaluate(model, val_loader, validation_ce, device)
        scheduler.step(val_accuracy)

        current_backbone_lr = optimizer.param_groups[0]["lr"]
        current_head_lr = optimizer.param_groups[1]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_accuracy,
            "val_loss": val_loss,
            "val_acc": val_accuracy,
            "ce_standard": running_ce_standard / total,
            "ce_auxiliary": running_ce_auxiliary / total,
            "fd_loss": running_fd / total,
            "cl_loss": running_cl / total,
            "backbone_lr": current_backbone_lr,
            "head_lr": current_head_lr,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train {train_accuracy:.2f}% | Val {val_accuracy:.2f}% | "
            f"ValLoss {val_loss:.4f} | FD {row['fd_loss']:.4f} | "
            f"CL {row['cl_loss']:.4f}"
        )

        with open(history_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
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
            print(f"[SUCCESS] New best checkpoint: {best_accuracy:.2f}%")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"[INFO] Early stopping after {args.early_stop_patience} "
                f"epochs without validation improvement."
            )
            break

    print("\n" + "=" * 78)
    print(f"[DONE] Best validation accuracy: {best_accuracy:.2f}%")
    print(f"[DONE] Best checkpoint: {checkpoint_path}")
    print(f"[DONE] History: {history_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
