from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .modules import ModalityEncoder


class Model(nn.Module):
    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.num_classes = self._resolve_default_num_classes(model_cfg)
        self.d_model = int(model_cfg["d_model"])
        self.dropout = float(model_cfg.get("dropout", 0.1))
        self.task_aliases = {
            "staging": "sleep_staging",
            "sleep_stage_classification": "sleep_staging",
        }

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
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(self.d_model)

        fusion_cfg = model_cfg.get("fusion", {})
        rope_dim = int(fusion_cfg.get("rope_dim", self.d_model))
        rope_dim = min(self.d_model, max(0, rope_dim))
        if rope_dim % 2 != 0:
            rope_dim -= 1
        self.use_rope = bool(fusion_cfg.get("use_rope", True)) and rope_dim >= 2
        self.rope_dim = rope_dim
        if self.use_rope:
            inv_freq = 1.0 / (
                10000
                ** (
                    torch.arange(0, self.rope_dim, 2, dtype=torch.float32)
                    / float(self.rope_dim)
                )
            )
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        else:
            self.register_buffer("rope_inv_freq", torch.empty(0), persistent=False)

        self.use_pairwise_interaction = bool(fusion_cfg.get("use_pairwise_interaction", True))
        self.interaction_scale = self.d_model ** -0.5
        self.interaction_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.interaction_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.interaction_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.interaction_out = nn.Linear(self.d_model, self.d_model, bias=False)
        temporal_cfg = model_cfg.get("temporal", {})
        self.use_temporal_context = bool(temporal_cfg.get("enabled", True))
        temporal_layers = int(temporal_cfg.get("num_layers", 2))
        if self.use_temporal_context and temporal_layers > 0:
            temporal_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=int(temporal_cfg.get("nhead", tf_cfg.get("nhead", 8))),
                dim_feedforward=int(temporal_cfg.get("dim_feedforward", 4 * self.d_model)),
                dropout=self.dropout,
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

        self.task_heads = nn.ModuleDict()
        task_cfg = model_cfg.get("downstream_tasks", {})
        if isinstance(task_cfg, dict):
            for task_name, head_cfg_raw in task_cfg.items():
                if not isinstance(head_cfg_raw, dict):
                    head_cfg_raw = {}
                normalized = self._normalize_task_name(str(task_name))
                if normalized in self.task_heads:
                    continue
                raw_out_dim = head_cfg_raw.get("num_classes", self.num_classes)
                if raw_out_dim is None:
                    raise ValueError(f"model.downstream_tasks.{normalized}.num_classes is required.")
                out_dim = int(raw_out_dim)
                head_dropout = float(head_cfg_raw.get("dropout", self.dropout))
                self.task_heads[normalized] = nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(self.d_model, out_dim),
                )

        if "sleep_staging" not in self.task_heads and self.num_classes is not None:
            self.task_heads["sleep_staging"] = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.num_classes),
            )
        if "sleep_staging" in self.task_heads:
            # Backward-compat alias used by some older code paths.
            self.classifier = self.task_heads["sleep_staging"]

    def _resolve_default_num_classes(self, model_cfg: dict[str, Any]) -> int | None:
        if "num_classes" in model_cfg and model_cfg.get("num_classes") is not None:
            return int(model_cfg["num_classes"])
        task_cfg = model_cfg.get("downstream_tasks", {})
        if isinstance(task_cfg, dict):
            sleep_cfg = task_cfg.get("sleep_staging", {})
            if isinstance(sleep_cfg, dict) and "num_classes" in sleep_cfg:
                return int(sleep_cfg["num_classes"])
        return None

    def _normalize_task_name(self, task_name: str | None) -> str:
        if task_name is None:
            return "sleep_staging"
        normalized = str(task_name).strip().lower()
        if normalized == "":
            return "sleep_staging"
        return self.task_aliases.get(normalized, normalized)

    def get_supported_tasks(self) -> list[str]:
        return sorted(list(self.task_heads.keys()))

    def get_task_head(self, task_name: str | None) -> tuple[nn.Module, str]:
        normalized = self._normalize_task_name(task_name)
        if normalized not in self.task_heads:
            raise KeyError(
                f"Unknown downstream task '{task_name}'. "
                f"Supported tasks: {self.get_supported_tasks()}"
            )
        return self.task_heads[normalized], normalized

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _build_rope_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, device=device, dtype=self.rope_inv_freq.dtype)
        freqs = torch.einsum("l,d->ld", pos, self.rope_inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1).unsqueeze(0)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1).unsqueeze(0)
        return cos, sin

    def _apply_rope(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_rope:
            return tokens
        if tokens.shape[-1] < self.rope_dim or self.rope_dim < 2:
            return tokens
        rope_part = tokens[..., : self.rope_dim]
        rest = tokens[..., self.rope_dim :]
        cos, sin = self._build_rope_cos_sin(
            seq_len=tokens.shape[1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        rope_part = rope_part * cos + self._rotate_half(rope_part) * sin
        if rest.numel() == 0:
            return rope_part
        return torch.cat([rope_part, rest], dim=-1)

    def encode_modalities(self, modalities: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded: dict[str, torch.Tensor] = {}
        for mod_name in self.modality_names:
            encoded[mod_name] = self.encoders[mod_name](modalities[mod_name])
        return encoded

    def _build_tokens_and_padding_mask(
        self,
        encoded: dict[str, torch.Tensor],
        modality_mask: dict[str, torch.Tensor],
        channel_mask: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_list = []
        mask_list = []
        for mod_name in self.modality_names:
            feat = encoded[mod_name]
            token_list.append(feat)
            present = modality_mask[mod_name].to(dtype=torch.bool, device=feat.device)
            if channel_mask is not None and mod_name in channel_mask:
                ch_present = channel_mask[mod_name].to(device=feat.device)
                if ch_present.ndim >= 2:
                    ch_present = ch_present.any(dim=1)
                ch_present = ch_present.to(dtype=torch.bool, device=feat.device)
                present = present & ch_present
            mask_list.append(present)

        tokens = torch.stack(token_list, dim=1)
        present_mask = torch.stack(mask_list, dim=1)
        padding_mask = ~present_mask

        tokens = self._apply_rope(tokens)

        all_missing = padding_mask.all(dim=1)
        if all_missing.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_missing, 0] = False
        return tokens, padding_mask, present_mask

    def _apply_pairwise_interaction(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_pairwise_interaction:
            return tokens
        valid = ~padding_mask
        q = self.interaction_q(tokens)
        k = self.interaction_k(tokens)
        v = self.interaction_v(tokens)
        logits = torch.matmul(q, k.transpose(1, 2)) * self.interaction_scale
        logits = logits.masked_fill(~valid.unsqueeze(1), -1e4)
        weights = torch.softmax(logits, dim=-1)
        interacted = torch.matmul(weights, v)
        interacted = self.interaction_out(interacted)
        return tokens + interacted * valid.unsqueeze(-1).to(dtype=tokens.dtype)

    def fuse_tokens(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self._apply_pairwise_interaction(tokens=tokens, padding_mask=padding_mask)
        fused = self.fusion(tokens, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).float()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = self.norm(pooled)
        return pooled

    def forward(
        self,
        modalities: dict[str, torch.Tensor],
        modality_mask: dict[str, torch.Tensor],
        channel_mask: dict[str, torch.Tensor] | None = None,
        seq_padding_mask: torch.Tensor | None = None,
        task_name: str = "sleep_staging",
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        first_modality = next(iter(self.modality_names))
        x0 = modalities[first_modality]
        is_sequence = x0.ndim == 4

        if not is_sequence:
            encoded = self.encode_modalities(modalities)
            tokens, padding_mask, _ = self._build_tokens_and_padding_mask(
                encoded=encoded,
                modality_mask=modality_mask,
                channel_mask=channel_mask,
            )
            pooled = self.fuse_tokens(tokens=tokens, padding_mask=padding_mask)
            head, _ = self.get_task_head(task_name)
            logits = head(pooled)
            if return_features:
                return logits, pooled
            return logits

        batch_size, seq_len = x0.shape[0], x0.shape[1]
        flat_modalities: dict[str, torch.Tensor] = {}
        flat_modality_mask: dict[str, torch.Tensor] = {}
        flat_channel_mask: dict[str, torch.Tensor] | None = {} if channel_mask is not None else None
        for mod_name in self.modality_names:
            x_mod = modalities[mod_name]
            if x_mod.ndim != 4:
                raise ValueError(
                    f"Sequence mode expects modality tensor [B, L, C, T], got {tuple(x_mod.shape)} for {mod_name}"
                )
            flat_modalities[mod_name] = x_mod.reshape(batch_size * seq_len, x_mod.shape[2], x_mod.shape[3])
            m_mod = modality_mask[mod_name]
            if m_mod.ndim != 2:
                raise ValueError(
                    f"Sequence mode expects modality mask [B, L], got {tuple(m_mod.shape)} for {mod_name}"
                )
            flat_modality_mask[mod_name] = m_mod.reshape(batch_size * seq_len)

            if flat_channel_mask is not None and channel_mask is not None and mod_name in channel_mask:
                c_mod = channel_mask[mod_name]
                if c_mod.ndim != 3:
                    raise ValueError(
                        f"Sequence mode expects channel mask [B, L, C], got {tuple(c_mod.shape)} for {mod_name}"
                    )
                flat_channel_mask[mod_name] = c_mod.reshape(batch_size * seq_len, c_mod.shape[2])

        encoded = self.encode_modalities(flat_modalities)
        tokens, padding_mask, _ = self._build_tokens_and_padding_mask(
            encoded=encoded,
            modality_mask=flat_modality_mask,
            channel_mask=flat_channel_mask,
        )
        pooled_flat = self.fuse_tokens(tokens=tokens, padding_mask=padding_mask)
        features = pooled_flat.reshape(batch_size, seq_len, self.d_model)

        if seq_padding_mask is None:
            valid_t = None
            for mod_name in self.modality_names:
                m = modality_mask[mod_name].to(dtype=torch.bool, device=features.device)
                valid_t = m if valid_t is None else (valid_t | m)
            seq_padding_mask = ~valid_t
        else:
            seq_padding_mask = seq_padding_mask.to(dtype=torch.bool, device=features.device)

        if self.temporal_encoder is not None:
            features = self.temporal_encoder(features, src_key_padding_mask=seq_padding_mask)
        features = self.temporal_norm(features)

        head, _ = self.get_task_head(task_name)
        logits = head(features)
        if return_features:
            return logits, features
        return logits

    def freeze_backbone(self, freeze: bool = True) -> None:
        modules = [
            self.encoders,
            self.fusion,
            self.norm,
            self.interaction_q,
            self.interaction_k,
            self.interaction_v,
            self.interaction_out,
            self.temporal_encoder,
            self.temporal_norm,
        ]
        for module in modules:
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = not freeze
        for head in self.task_heads.values():
            for p in head.parameters():
                p.requires_grad = True
