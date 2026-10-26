import argparse
import csv
import json
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_mambavision_b21k_fdcl_da_cle_fghfa_stage34 import (
    MambaVisionCLEFGHFAStage34,
)
from train_mambavision_b21k_fdcl_da import (
    IMAGE_SIZE as TRAIN_IMAGE_SIZE,
    RESIZE_SIZE as TRAIN_RESIZE_SIZE,
    audit_r1_dataset,
)

NUM_CLASSES = 198
IMAGE_SIZE = 448
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

CONFIG = {
    "R1": {
        "data": "data/SoyAgeing-R1/R1",
        "checkpoint": "cle_fghfa_r1_stage34_results/best_cle_fghfa_r1_stage34.pth",
        "history": "cle_fghfa_r1_stage34_results/training_history.csv",
    },
    "R3": {
        "data": "data/SoyAgeing-R3/R3",
        "checkpoint": "cle_fghfa_r3_stage34_results/best_cle_fghfa_r3_stage34.pth",
        "history": "cle_fghfa_r3_stage34_results/training_history.csv",
    },
    "R4": {
        "data": "data/SoyAgeing-R4",
        "checkpoint": "cle_fghfa_r4_stage34_results/best_cle_fghfa_r4_stage34.pth",
        "history": "cle_fghfa_r4_stage34_results/training_history.csv",
    },
    "R5": {
        "data": "data/SoyAgeing-R5",
        "checkpoint": "cle_fghfa_r5_stage34_results/best_cle_fghfa_r5_stage34.pth",
        "history": "cle_fghfa_r5_stage34_results/training_history.csv",
    },
}


class SoyAgeingEvaluationDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir).resolve()
        annotation = self.base_dir / "anno" / "val.txt"
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing validation split: {annotation}")

        self.samples = []
        for raw in annotation.read_text().splitlines():
            fields = raw.strip().split()
            if len(fields) < 2:
                continue

            image_name, raw_label = fields[0], int(fields[-1])
            if not 1 <= raw_label <= NUM_CLASSES:
                raise RuntimeError(
                    f"Invalid SoyAgeing label {raw_label}; expected labels in [1, {NUM_CLASSES}]"
                )
            # SoyAgeing annotation labels are one-based; PyTorch CE expects 0..197.
            label = raw_label - 1
            candidates = [
                self.base_dir / image_name,
                self.base_dir / "images" / image_name,
            ]
            image_path = next((p.resolve() for p in candidates if p.is_file()), None)
            if image_path is None:
                raise FileNotFoundError(f"Missing image: {image_name}")
            if self.base_dir not in image_path.parents:
                raise RuntimeError(f"Image outside subset root: {image_path}")
            self.samples.append((image_path, label))

        labels = {label for _, label in self.samples}
        if len(self.samples) != 990 or len(labels) != NUM_CLASSES:
            raise RuntimeError(
                f"Invalid evaluation split: samples={len(self.samples)}, classes={len(labels)}"
            )

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label, str(image_path)



class ExactSoyAgeingEvaluationDataset(Dataset):
    """Uses the trainer's audited samples and exact validation transform."""

    def __init__(self, samples, validation_transform):
        self.samples = samples
        self.transform = validation_transform

        labels = [label for _, label in samples]
        if len(samples) != 990:
            raise RuntimeError(f"Expected 990 validation samples, got {len(samples)}")
        if min(labels) != 0 or max(labels) != 197 or len(set(labels)) != 198:
            raise RuntimeError(
                f"Unexpected trainer label mapping: min={min(labels)}, "
                f"max={max(labels)}, classes={len(set(labels))}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label, str(image_path)

def resolve_logits(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2 and item.shape[1] == NUM_CLASSES:
                return item
    if isinstance(output, dict):
        for key in ("logits", "output", "main_logits"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise RuntimeError("Could not identify classification logits from model output.")


def build_model():
    signature = inspect.signature(MambaVisionCLEFGHFAStage34)
    possible = {
        "num_classes": NUM_CLASSES,
        "fghfa_stages": (3, 4),
        "fghfa_fusion_dim": 1024,
    }
    kwargs = {key: value for key, value in possible.items() if key in signature.parameters}
    return MambaVisionCLEFGHFAStage34(**kwargs)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint.get("model")
        or checkpoint
    )
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    required = [
        "fghfa.projections.3",
        "fghfa.projections.4",
        "cle_projector.0",
        "cle_projector.2",
    ]
    for token in required:
        if not any(token in key for key in state_dict):
            raise RuntimeError(
                f"Checkpoint is not a CLE + FGHFA [3,4] checkpoint; missing {token}"
            )

    model = build_model()
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing: {incompatible.missing_keys}\n"
            f"Unexpected: {incompatible.unexpected_keys}"
        )

    model.to(device).eval()
    return model, checkpoint


def denormalize(tensor):
    mean = torch.tensor(MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def pick_column(frame, names):
    lowered = {column.lower(): column for column in frame.columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def plot_curves(history_path, output_path, subset):
    frame = pd.read_csv(history_path)

    epoch = pick_column(frame, ["epoch"])
    train_loss = pick_column(frame, ["train_loss", "total_loss", "loss"])
    val_loss = pick_column(frame, ["val_loss", "validation_loss"])
    train_acc = pick_column(frame, ["train_accuracy", "train_acc"])
    val_acc = pick_column(frame, ["val_accuracy", "validation_accuracy", "val_acc"])

    if epoch is None:
        frame["epoch"] = np.arange(1, len(frame) + 1)
        epoch = "epoch"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    if train_loss:
        axes[0].plot(frame[epoch], frame[train_loss], label="Training loss", linewidth=2)
    if val_loss:
        axes[0].plot(frame[epoch], frame[val_loss], label="Validation loss", linewidth=2)
    axes[0].set_title(f"{subset}: Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if train_acc:
        axes[1].plot(frame[epoch], frame[train_acc], label="Training accuracy", linewidth=2)
    if val_acc:
        axes[1].plot(frame[epoch], frame[val_acc], label="Validation accuracy", linewidth=2)
    axes[1].set_title(f"{subset}: Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(matrix, output_path, subset):
    normalized = matrix.astype(np.float64)
    normalized /= np.maximum(normalized.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"{subset}: Normalized Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_gradcam(model, dataset, device, output_path):
    target_layer = model.backbone.levels[3]
    activations, gradients = [], []

    def forward_hook(_, __, output):
        activations.append(output[0] if isinstance(output, (tuple, list)) else output)

    def backward_hook(_, __, output):
        gradients.append(output[0] if isinstance(output, (tuple, list)) else output)

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    indices = np.linspace(0, len(dataset) - 1, 5, dtype=int)
    figure, axes = plt.subplots(len(indices), 3, figsize=(10, 3.4 * len(indices)))

    for row, index in enumerate(indices):
        image, true_label, _ = dataset[index]
        input_tensor = image.unsqueeze(0).to(device)

        activations.clear()
        gradients.clear()
        model.zero_grad(set_to_none=True)

        logits, _, _ = model(input_tensor)
        prediction = int(logits.argmax(dim=1).item())
        logits[0, prediction].backward()

        activation = activations[-1]
        gradient = gradients[-1]

        if activation.ndim == 4 and activation.shape[1] != gradient.shape[1]:
            activation = activation.permute(0, 3, 1, 2)
            gradient = gradient.permute(0, 3, 1, 2)

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        original = denormalize(image.to(device)).permute(1, 2, 0).detach().cpu().numpy()
        heatmap = plt.get_cmap("jet")(cam.detach().cpu().numpy())[:, :, :3]
        overlay = (0.55 * original + 0.45 * heatmap).clip(0, 1)

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"Original\nTrue: {true_label}")
        axes[row, 1].imshow(cam.detach().cpu().numpy(), cmap="jet")
        axes[row, 1].set_title(f"Grad-CAM\nPredicted: {prediction}")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")

        for column in range(3):
            axes[row, column].axis("off")

    forward_handle.remove()
    backward_handle.remove()
    figure.tight_layout()
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def evaluate(model, loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    true_labels, predictions, paths = [], [], []
    total_loss, total = 0.0, 0

    for images, labels, batch_paths in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _, _ = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        true_labels.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        paths.extend(batch_paths)

    return (
        np.asarray(true_labels),
        np.asarray(predictions),
        paths,
        total_loss / total,
    )


def evaluate_subset(subset, output_root, batch_size, device):
    config = CONFIG[subset]
    checkpoint = Path(config["checkpoint"])
    history = Path(config["history"])
    base_dir = Path(config["data"])

    if not checkpoint.is_file() or not history.is_file():
        raise FileNotFoundError(f"Missing checkpoint or history for {subset}")

    output_dir = output_root / subset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the exact same audited label mapping used by the trainer.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        _, val_samples = audit_r1_dataset(str(base_dir))

    validation_transform = transforms.Compose([
        transforms.Resize((TRAIN_RESIZE_SIZE, TRAIN_RESIZE_SIZE)),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    dataset = ExactSoyAgeingEvaluationDataset(val_samples, validation_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model, checkpoint_data = load_model(checkpoint, device)
    labels, predictions, paths, loss = evaluate(model, loader, device)

    accuracy = float((labels == predictions).mean() * 100)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(NUM_CLASSES), zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=np.arange(NUM_CLASSES))

    pd.DataFrame({
        "image_path": paths,
        "true_label": labels,
        "predicted_label": predictions,
        "correct": labels == predictions,
    }).to_csv(output_dir / "predictions.csv", index=False)

    pd.DataFrame(matrix).to_csv(output_dir / "confusion_matrix.csv", index=False)

    pd.DataFrame({
        "class_id": np.arange(NUM_CLASSES),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": matrix.sum(axis=1),
    }).to_csv(output_dir / "per_class_metrics.csv", index=False)

    report = classification_report(
        labels, predictions, labels=np.arange(NUM_CLASSES), zero_division=0, digits=4
    )
    (output_dir / "classification_report.txt").write_text(report)

    summary = {
        "subset": subset,
        "evaluation_split": "val.txt (used as the provided unseen evaluation/test split)",
        "samples": int(len(dataset)),
        "accuracy_percent": accuracy,
        "cross_entropy_loss": float(loss),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_data.get("epoch"),
        "checkpoint_val_accuracy": checkpoint_data.get("best_accuracy", checkpoint_data.get("val_accuracy")),
        "architecture": "MambaVision-B-21K + FDCL-DA + FGHFA [3,4] + CLE",
        "fghfa_stages": [3, 4],
        "fghfa_fusion_dim": 1024,
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))

    plot_curves(history, output_dir / "training_validation_curves.png", subset)
    plot_confusion(matrix, output_dir / "normalized_confusion_matrix.png", subset)
    make_gradcam(model, dataset, device, output_dir / "GradCAM_heatmaps.png")

    print("=" * 78)
    print(f"[SUCCESS] {subset} evaluation saved: {output_dir}")
    print(f"Accuracy: {accuracy:.2f}% | Loss: {loss:.6f} | Macro F1: {f1.mean():.4f}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", default=["R1", "R3", "R4", "R5"])
    parser.add_argument("--output_root", default="final_cle_fghfa_evaluations")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    for subset in args.subsets:
        if subset not in CONFIG:
            raise ValueError(f"Unsupported subset: {subset}")
        evaluate_subset(subset, Path(args.output_root), args.batch_size, device)


if __name__ == "__main__":
    main()
