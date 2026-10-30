import torch
import torch.nn as nn
import torch.nn.functional as F
import mamba_ssm.ops.selective_scan_interface as ssi

# Route selective scan to pure PyTorch reference for RTX 5090 stability
ssi.selective_scan_fn = ssi.selective_scan_ref

from transformers import AutoModel

class MambaVisionDAFD(nn.Module):
    def __init__(self, num_classes=198):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained NVIDIA MambaVision-B backbone
        self.backbone = AutoModel.from_pretrained(
            'nvidia/MambaVision-B-1K',
            trust_remote_code=True
        )

        # MambaVision-B stage channels: Stage 1 (128), Stage 2 (256), Stage 3 (512), Stage 4 (1024)
        self.stage_dims = [128, 256, 512, 1024]

        # 1x1 stage-wise classifiers to compute class activation maps Y_i (Eq. 1)
        self.fd_classifiers = nn.ModuleList([
            nn.Conv2d(dim, num_classes, kernel_size=1) for dim in self.stage_dims
        ])

        # Primary classification head
        self.classifier = nn.Linear(self.stage_dims[-1], num_classes)

    def compute_fd_loss(self, stage_feats):
        """
        Foreground-background feature distillation (FD) module
        Implements Algorithm 1 and Eq. (1)-(6) from Chen et al. (Pattern Recognition 2024).
        """
        total_fd_loss = 0.0
        # K_i selection: K_i > K_j when i < j (earlier stages keep higher top-K proportion)
        k_ratio = [0.4, 0.3, 0.2, 0.1]

        for i, (feat, clf) in enumerate(zip(stage_feats, self.fd_classifiers)):
            # Eq. 1: Classification map Y_i
            logits_map = clf(feat) # [B, num_classes, H_i, W_i]
            prob_map = F.softmax(logits_map, dim=1)

            # Eq. 2: Maximum score map
            y_max, _ = torch.max(prob_map, dim=1) # [B, H_i, W_i]
            B, H, W = y_max.shape
            y_max_flat = y_max.view(B, H * W)

            # Determine Ki count
            K_i = max(1, int((H * W) * k_ratio[i]))

            # Sort descending to separate foreground and background
            sorted_scores, _ = torch.sort(y_max_flat, descending=True, dim=1)

            # Background features: scores past top-K
            bg_scores = sorted_scores[:, K_i:]

            # Eq. 4 & 5: Hardtanh activation function mapping background features to (-1, 1)
            tanh_val = torch.tanh(bg_scores)
            p_bg = (2.0 * tanh_val - 1.0) / (1.0 + tanh_val ** 2 + 1e-7)

            # Eq. 6: Distillation MSE loss against pseudo-target -1
            stage_fd_loss = torch.mean((p_bg + 1.0) ** 2)
            total_fd_loss += stage_fd_loss

        return total_fd_loss

    def forward(self, x):
        pooler, features = self.backbone(x)
        logits = self.classifier(pooler)
        return logits, features
