from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .model import Model


TASK_ALIASES = {
    "staging": "sleep_staging",
    "sleep_stage_classification": "sleep_staging",
}


def normalize_task_name(task_name: str | None) -> str:
    if task_name is None:
        return "sleep_staging"
    normalized = str(task_name).strip().lower()
    if normalized == "":
        return "sleep_staging"
    return TASK_ALIASES.get(normalized, normalized)


class LinearPredictionHead(nn.Sequential):
    def __init__(self, d_model: int, out_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )


class SleepStagingDownstreamModel(nn.Module):
    def __init__(
        self,
        backbone: Model,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.model_cfg = backbone.model_cfg
        self.d_model = backbone.d_model
        self.num_classes = int(num_classes)
        self.classifier = LinearPredictionHead(
            d_model=self.d_model,
            out_dim=self.num_classes,
            dropout=float(dropout),
        )

    def get_supported_tasks(self) -> list[str]:
        return ["sleep_staging"]

    def _check_task(self, task_name: str | None) -> str:
        normalized = normalize_task_name(task_name)
        if normalized != "sleep_staging":
            raise KeyError(
                f"Unknown downstream task '{task_name}'. "
                f"Supported tasks: {self.get_supported_tasks()}"
            )
        return normalized

    def freeze_backbone(self, freeze: bool = True) -> None:
        self.backbone.freeze_backbone(freeze=freeze)
        for param in self.classifier.parameters():
            param.requires_grad = True

    def load_pretrained_backbone_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        backbone_prefix = "backbone."
        module_prefix = "module."
        ignored_prefixes = (
            "classifier.",
            "task_heads.",
            "temporal_encoder.",
            "temporal_norm.",
        )
        target_state = self.backbone.state_dict()
        backbone_state = {}
        unexpected = []
        ignored = []

        for raw_key, value in state_dict.items():
            key = raw_key
            if key.startswith(module_prefix):
                key = key[len(module_prefix):]
            if key.startswith(backbone_prefix):
                key = key[len(backbone_prefix):]
            if key.startswith(ignored_prefixes):
                ignored.append(raw_key)
                continue
            if key not in target_state:
                unexpected.append(raw_key)
                continue
            if tuple(target_state[key].shape) != tuple(value.shape):
                unexpected.append(raw_key)
                continue
            backbone_state[key] = value

        missing, unexpected_from_load = self.backbone.load_state_dict(backbone_state, strict=False)
        unexpected.extend(unexpected_from_load)
        return {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "ignored_keys": ignored,
        }


class SleepStagingLinearModel(SleepStagingDownstreamModel):
    def forward(
        self,
        modalities: dict[str, torch.Tensor],
        modality_mask: dict[str, torch.Tensor],
        channel_mask: dict[str, torch.Tensor] | None = None,
        seq_padding_mask: torch.Tensor | None = None,
        task_name: str = "sleep_staging",
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._check_task(task_name)
        first_modality = next(iter(self.backbone.modality_names))
        if modalities[first_modality].ndim == 4:
            raise ValueError("Linear downstream mode expects single-epoch inputs [B, C, T].")
        _, features = self.backbone(
            modalities=modalities,
            modality_mask=modality_mask,
            channel_mask=channel_mask,
            seq_padding_mask=seq_padding_mask,
            return_features=True,
        )
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class SleepStagingSeq2SeqModel(SleepStagingDownstreamModel):
    def __init__(
        self,
        backbone: Model,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
        )
        tf_cfg = self.model_cfg.get("transformer", {})
        temporal_cfg = self.model_cfg.get("temporal", {})
        self.use_temporal_context = bool(temporal_cfg.get("enabled", True))
        temporal_layers = int(temporal_cfg.get("num_layers", 2))
        if self.use_temporal_context and temporal_layers > 0:
            temporal_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=int(temporal_cfg.get("nhead", tf_cfg.get("nhead", 8))),
                dim_feedforward=int(temporal_cfg.get("dim_feedforward", 4 * self.d_model)),
                dropout=float(temporal_cfg.get("dropout", self.model_cfg.get("dropout", 0.1))),
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer=temporal_layer,
                num_layers=temporal_layers,
                enable_nested_tensor=False,
            )
        else:
            self.temporal_encoder = None
        self.temporal_norm = nn.LayerNorm(self.d_model)

    def freeze_backbone(self, freeze: bool = True) -> None:
        super().freeze_backbone(freeze=freeze)
        for module in (self.temporal_encoder, self.temporal_norm):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True

    def forward(
        self,
        modalities: dict[str, torch.Tensor],
        modality_mask: dict[str, torch.Tensor],
        channel_mask: dict[str, torch.Tensor] | None = None,
        seq_padding_mask: torch.Tensor | None = None,
        task_name: str = "sleep_staging",
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._check_task(task_name)
        _, features = self.backbone(
            modalities=modalities,
            modality_mask=modality_mask,
            channel_mask=channel_mask,
            seq_padding_mask=seq_padding_mask,
            return_features=True,
        )
        if features.ndim == 3:
            if seq_padding_mask is None:
                valid_t = None
                for mod_name in self.backbone.modality_names:
                    mask = modality_mask[mod_name].to(dtype=torch.bool, device=features.device)
                    valid_t = mask if valid_t is None else (valid_t | mask)
                seq_padding_mask = ~valid_t
            else:
                seq_padding_mask = seq_padding_mask.to(dtype=torch.bool, device=features.device)
            if self.temporal_encoder is not None:
                features = self.temporal_encoder(features, src_key_padding_mask=seq_padding_mask)
            features = self.temporal_norm(features)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def _resolve_sleep_staging_head_cfg(model_cfg: dict[str, Any]) -> tuple[int, float]:
    task_cfg = model_cfg.get("downstream_tasks", {})
    sleep_cfg = task_cfg.get("sleep_staging", {}) if isinstance(task_cfg, dict) else {}
    if not isinstance(sleep_cfg, dict):
        sleep_cfg = {}
    raw_num_classes = sleep_cfg.get("num_classes", model_cfg.get("num_classes", None))
    if raw_num_classes is None:
        raise ValueError("model.downstream_tasks.sleep_staging.num_classes is required.")
    dropout = float(sleep_cfg.get("dropout", model_cfg.get("dropout", 0.1)))
    return int(raw_num_classes), dropout


def build_downstream_model(
    backbone: Model,
    model_cfg: dict[str, Any],
    method: str,
) -> SleepStagingDownstreamModel:
    num_classes, dropout = _resolve_sleep_staging_head_cfg(model_cfg)
    normalized_method = str(method).strip().lower()
    if normalized_method in {"linear", "epoch_linear", "single_epoch"}:
        return SleepStagingLinearModel(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
        )
    if normalized_method in {"seq2seq", "sequence_to_sequence", "sequence"}:
        return SleepStagingSeq2SeqModel(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(
        f"Unsupported training.downstream_method: {method}. "
        "Use 'seq2seq' or 'linear'."
    )
