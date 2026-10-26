import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mambavision_b21k_fdcl_da import MambaVisionPaperFDCLDA

STAGE_FOREGROUND_COUNTS = {2: 128, 3: 64, 4: 32}


class StageProjection(nn.Module):
    def __init__(self, input_dim, output_dim=1024):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(self.norm(x))


class ForegroundGuidedHierarchicalAggregation(nn.Module):
    """
    Reuses the existing FD top-K predicted-class foreground mechanism.

    Stage feature -> existing FD classifier -> top-K foreground weighting
                  -> weighted pooling -> projection -> stage fusion.
    """

    def __init__(self, stage_channels, stages=(2, 3, 4), fusion_dim=1024):
        super().__init__()

        self.stages = tuple(stages)
        self.fusion_dim = fusion_dim

        for stage in self.stages:
            if stage not in STAGE_FOREGROUND_COUNTS:
                raise ValueError(f"FGHFA supports stages 2/3/4, got {stage}")
            if stage not in stage_channels:
                raise ValueError(f"Missing channel definition for Stage {stage}")

        self.projections = nn.ModuleDict({
            str(stage): StageProjection(stage_channels[stage], fusion_dim)
            for stage in self.stages
        })

        # Uniform initial aggregation; learned through softmax.
        self.stage_logits = nn.Parameter(torch.zeros(len(self.stages)))

        # Near-zero, not exactly zero, so the branch receives gradients immediately.
        self.residual_gate = nn.Parameter(torch.tensor(0.001))
        self.last_info = {}

    @staticmethod
    def foreground_weighted_pool(feature_map, fd_classifier, select_count):
        """
        Uses the same predicted-class and top-K foreground definition as the
        existing FD loss. The soft predicted-class probability is retained as
        a differentiable foreground weight inside the selected top-K region.
        """
        class_maps = fd_classifier(feature_map)  # B,198,H,W
        batch, _, height, width = class_maps.shape

        token_logits = class_maps.flatten(2).transpose(1, 2).contiguous()
        probabilities = torch.softmax(token_logits, dim=-1)
        image_probabilities = torch.softmax(token_logits.mean(dim=1), dim=-1)
        predicted_class = image_probabilities.argmax(dim=1)

        batch_indices = torch.arange(batch, device=feature_map.device)
        foreground_scores = probabilities[
            batch_indices, :, predicted_class
        ]  # B,H*W

        token_count = foreground_scores.shape[1]
        k = min(select_count, token_count - 1)
        topk_indices = foreground_scores.topk(k, dim=1).indices

        topk_mask = torch.zeros_like(foreground_scores)
        topk_mask.scatter_(1, topk_indices, 1.0)

        weights = foreground_scores * topk_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        flattened_features = feature_map.flatten(2)  # B,C,H*W
        pooled = (flattened_features * weights.unsqueeze(1)).sum(dim=2)

        return pooled, weights.view(batch, 1, height, width), predicted_class

    def forward(self, stage_features, fd_classifiers):
        projected_features = []
        masks = {}
        predicted_classes = {}

        for stage in self.stages:
            feature_map = stage_features[stage - 1]
            pooled, mask, predicted = self.foreground_weighted_pool(
                feature_map,
                fd_classifiers[stage - 1],
                STAGE_FOREGROUND_COUNTS[stage],
            )
            projected_features.append(self.projections[str(stage)](pooled))
            masks[stage] = mask
            predicted_classes[stage] = predicted

        stack = torch.stack(projected_features, dim=1)  # B,S,1024
        stage_weights = torch.softmax(self.stage_logits, dim=0)
        fused = (stack * stage_weights.view(1, -1, 1)).sum(dim=1)

        self.last_info = {
            "stage_weights": stage_weights.detach().cpu().tolist(),
            "gate": float(torch.tanh(self.residual_gate).detach().cpu()),
            "mask_shapes": {
                str(stage): tuple(mask.shape) for stage, mask in masks.items()
            },
            "predicted_classes": {
                str(stage): predicted.detach().cpu().tolist()
                for stage, predicted in predicted_classes.items()
            },
        }
        return fused


class MambaVisionFGHFAFDCLDA(MambaVisionPaperFDCLDA):
    """
    Original MambaVision-B-21K + FDCL-DA remains intact.

    FGHFA changes only the CE classifier representation:
      z_new = z_base + tanh(gate) * z_fghfa

    The returned feature for existing supervised CL remains z_base, preserving
    that CL objective and its original interface exactly.
    """

    def __init__(
        self,
        pretrained=True,
        fghfa_enabled=True,
        fghfa_stages=(2, 3, 4),
        fusion_dim=1024,
    ):
        super().__init__(pretrained=pretrained)

        if fusion_dim != 1024:
            raise ValueError(
                "fusion_dim must remain 1024 to preserve the existing classifier."
            )

        self.fghfa_enabled = fghfa_enabled
        self.fghfa = ForegroundGuidedHierarchicalAggregation(
            stage_channels={2: 512, 3: 1024, 4: 1024},
            stages=fghfa_stages,
            fusion_dim=fusion_dim,
        )

    def set_fghfa_enabled(self, enabled):
        self.fghfa_enabled = bool(enabled)

    def forward(self, images):
        self._stage_features = []
        final_features = self.backbone(images)
        stage_features = [self._as_nchw(item) for item in self._stage_features]

        if len(stage_features) != 4:
            raise RuntimeError(
                f"Expected four MambaVision stages, got {len(stage_features)}."
            )

        base_features = self._pool(final_features)  # B,1024

        if self.fghfa_enabled:
            fghfa_features = self.fghfa(stage_features, self.fd_classifiers)
            classifier_features = (
                base_features
                + torch.tanh(self.fghfa.residual_gate) * fghfa_features
            )
        else:
            # Identical original baseline classifier path.
            classifier_features = base_features

        logits = self.classifier(classifier_features)

        # Preserve existing FDCL supervised-CL input: original final feature.
        return logits, base_features, stage_features
