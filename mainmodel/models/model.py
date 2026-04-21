from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .modules import ModalityEncoder


class Model(nn.Module):
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.num_classes = int(model_cfg["num_classes"])
        self.d_model = int(model_cfg["d_model"])
        self.dropout = float(model_cfg.get("dropout", 0.1))

        modalities_cfg = model_cfg.get("modalities", {})
        self.modality_names = list(modalities_cfg.keys())
        if len(self.modality_names) == 0:
            raise ValueError("model.modalities must not be empty.")

        self.encoders = nn.ModuleDict()
        for mod_name, mod_cfg in modalities_cfg.items():
            enc_cfg = mod_cfg.get("encoder", {})
            seq_len = int(mod_cfg.get("seq_len", int(mod_cfg.get("sample_rate", 100)) * int(model_cfg.get("epoch_seconds", 30))))
            self.encoders[mod_name] = ModalityEncoder(
                in_channels=int(mod_cfg["in_channels"]),
                d_model=self.d_model,
                hidden_channels=int(enc_cfg.get("hidden_channels", self.d_model // 2)),
                kernel_size=int(enc_cfg.get("kernel_size", 7)),
                dropout=self.dropout,
                seq_len=seq_len,
                n_blocks=int(enc_cfg.get("n_blocks", 3)),
                pool_sizes=enc_cfg.get("pool_sizes", None),
                d_state=int(enc_cfg.get("d_state", 64)),
            )

        self.modality_embedding = nn.Embedding(len(self.modality_names), self.d_model)

        reference_cfg = model_cfg.get("reference_embedding", {})
        self.reference_enabled = bool(reference_cfg.get("enabled", True))
        self.num_references = int(reference_cfg.get("num_references", 16))
        self.reference_embedding = (
            nn.Embedding(self.num_references, self.d_model)
            if self.reference_enabled
            else None
        )

        tf_cfg = model_cfg.get("transformer", {})
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(tf_cfg.get("nhead", 8)),
            dim_feedforward=int(tf_cfg.get("dim_feedforward", 4 * self.d_model)),
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(tf_cfg.get("num_layers", 2)),
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.num_classes),
        )

    def encode_modalities(self, modalities: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded: dict[str, torch.Tensor] = {}
        for mod_name in self.modality_names:
            encoded[mod_name] = self.encoders[mod_name](modalities[mod_name])
        return encoded

    def _build_tokens_and_padding_mask(
        self,
        encoded: dict[str, torch.Tensor],
        modality_mask: dict[str, torch.Tensor],
        reference_ids: torch.Tensor | None,
        disable_reference_embedding: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_list = []
        mask_list = []
        for mod_idx, mod_name in enumerate(self.modality_names):
            feat = encoded[mod_name]
            feat = feat + self.modality_embedding.weight[mod_idx].unsqueeze(0)
            token_list.append(feat)
            mask_list.append(modality_mask[mod_name].to(dtype=torch.bool, device=feat.device))

        tokens = torch.stack(token_list, dim=1)
        present_mask = torch.stack(mask_list, dim=1)
        padding_mask = ~present_mask

        if self.reference_embedding is not None and not disable_reference_embedding and reference_ids is not None:
            ref = reference_ids.clamp(min=0, max=self.num_references - 1)
            ref_embed = self.reference_embedding(ref)
            tokens = tokens + ref_embed.unsqueeze(1)

        all_missing = padding_mask.all(dim=1)
        if all_missing.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_missing, 0] = False
        return tokens, padding_mask

    def fuse_tokens(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(tokens, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).float()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = self.norm(pooled)
        return pooled

    def forward(
        self,
        modalities: dict[str, torch.Tensor],
        modality_mask: dict[str, torch.Tensor],
        reference_ids: torch.Tensor | None = None,
        disable_reference_embedding: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_modalities(modalities)
        tokens, padding_mask = self._build_tokens_and_padding_mask(
            encoded=encoded,
            modality_mask=modality_mask,
            reference_ids=reference_ids,
            disable_reference_embedding=disable_reference_embedding,
        )
        pooled = self.fuse_tokens(tokens=tokens, padding_mask=padding_mask)
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits

    def freeze_backbone(self, freeze: bool = True) -> None:
        modules = [self.encoders, self.modality_embedding, self.fusion, self.norm, self.reference_embedding]
        for module in modules:
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = not freeze
        for p in self.classifier.parameters():
            p.requires_grad = True
