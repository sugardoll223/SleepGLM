from __future__ import annotations

import re
from typing import Any

import torch


def _normalize_channel_name(name: str) -> str:
    if not name:
        return ""
    # Uppercase and drop separators so "F3-M2", "F3_M2", "F3 M2" map together.
    return re.sub(r"[\s_\-:/]+", "", name).upper()


class ChannelAdapter:
    """
    Adapt raw channels from heterogeneous datasets/devices to a fixed channel layout.

    Priority:
    1) Name-based channel mapping (if channel names are available + canonical layout configured)
    2) Index-based fallback (resize/pad)
    """

    def __init__(self, data_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> None:
        self.epoch_seconds = int(data_cfg.get("epoch_seconds", 30))
        self.modalities_cfg = model_cfg.get("modalities", {})
        self.adapter_cfg = data_cfg.get("channel_adapter", {})

        self.enabled = bool(self.adapter_cfg.get("enabled", True))
        self.prefer_name_map = bool(self.adapter_cfg.get("prefer_name_map", True))
        self.missing_channel_policy = str(
            self.adapter_cfg.get(
                "missing_channel_policy",
                self.adapter_cfg.get("unmatched_strategy", "gaussian"),
            )
        ).lower()
        self.missing_gaussian_std = float(self.adapter_cfg.get("missing_gaussian_std", 0.05))

        # Config structure:
        # data.channel_adapter.modality_channel_schema.eeg.canonical_channels: [...]
        # data.channel_adapter.modality_channel_schema.eeg.alias_groups: {F3: [F3, F3-M2, F3-A2], ...}
        self.modality_schema = self.adapter_cfg.get("modality_channel_schema", {})

    @staticmethod
    def _resize_time(x: torch.Tensor, target_time: int) -> torch.Tensor:
        current_time = x.shape[1]
        if current_time == target_time:
            return x
        if current_time > target_time:
            start = (current_time - target_time) // 2
            return x[:, start : start + target_time]
        pad = torch.zeros((x.shape[0], target_time - current_time), dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    @staticmethod
    def _resize_channels(x: torch.Tensor, target_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
        current_channels = x.shape[0]
        if current_channels == target_channels:
            return x, torch.ones(target_channels, dtype=torch.bool)
        if current_channels > target_channels:
            return x[:target_channels, :], torch.ones(target_channels, dtype=torch.bool)
        pad_rows = target_channels - current_channels
        pad = torch.zeros((pad_rows, x.shape[1]), dtype=x.dtype)
        out = torch.cat([x, pad], dim=0)
        mask = torch.cat(
            [
                torch.ones(current_channels, dtype=torch.bool),
                torch.zeros(pad_rows, dtype=torch.bool),
            ],
            dim=0,
        )
        return out, mask

    @staticmethod
    def _zscore_per_channel(x: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if channel_mask is None:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            return (x - mean) / std
        out = x.clone()
        valid_idx = torch.where(channel_mask)[0]
        if valid_idx.numel() == 0:
            return out
        valid = out[valid_idx]
        mean = valid.mean(dim=1, keepdim=True)
        std = valid.std(dim=1, keepdim=True).clamp_min(1e-6)
        out[valid_idx] = (valid - mean) / std
        return out

    def _apply_missing_channel_mask(
        self,
        out: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        if channel_mask.all():
            return out
        if self.missing_channel_policy in {"zero", "zeros"}:
            return out
        # Default behavior: Gaussian noise mask for missing channels.
        missing_idx = torch.where(~channel_mask)[0]
        if missing_idx.numel() == 0:
            return out
        noise = torch.randn(
            (missing_idx.numel(), out.shape[1]),
            device=out.device,
            dtype=out.dtype,
        ) * self.missing_gaussian_std
        out = out.clone()
        out[missing_idx] = noise
        return out

    def _build_alias_lookup(self, modality_name: str) -> dict[str, int]:
        schema = self.modality_schema.get(modality_name, {})
        canonical_channels = list(schema.get("canonical_channels", []))
        alias_groups = schema.get("alias_groups", {})
        lookup: dict[str, int] = {}

        # Every canonical channel itself is also a valid alias.
        for idx, canonical in enumerate(canonical_channels):
            lookup[_normalize_channel_name(canonical)] = idx

        # Merge configured aliases.
        for canonical, aliases in alias_groups.items():
            if canonical not in canonical_channels:
                continue
            idx = canonical_channels.index(canonical)
            for alias in aliases:
                lookup[_normalize_channel_name(str(alias))] = idx
        return lookup

    def _name_map_channels(
        self,
        x: torch.Tensor,
        src_channel_names: list[str],
        target_channels: int,
        target_time: int,
        modality_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        schema = self.modality_schema.get(modality_name, {})
        canonical_channels = list(schema.get("canonical_channels", []))
        if not canonical_channels:
            return None

        alias_lookup = self._build_alias_lookup(modality_name)
        if not alias_lookup:
            return None

        # Keep stable shape based on model config even if canonical list is larger/smaller.
        effective_channels = min(target_channels, len(canonical_channels))
        out = torch.zeros((target_channels, target_time), dtype=x.dtype)
        channel_mask = torch.zeros(target_channels, dtype=torch.bool)

        x = self._resize_time(x, target_time=target_time)

        src_used = set()
        for src_idx, raw_name in enumerate(src_channel_names):
            norm_name = _normalize_channel_name(raw_name)
            if norm_name not in alias_lookup:
                continue
            dst_idx = alias_lookup[norm_name]
            if dst_idx >= effective_channels:
                continue
            if src_idx >= x.shape[0]:
                continue
            if dst_idx in src_used:
                # If multiple source channels map to one canonical channel, keep first match.
                continue
            out[dst_idx, :] = x[src_idx, :]
            channel_mask[dst_idx] = True
            src_used.add(dst_idx)

        if effective_channels < target_channels:
            # Unused tail channels remain 0 and mask False.
            pass
        return out, channel_mask

    def adapt(
        self,
        modality_name: str,
        x: torch.Tensor | None,
        src_channel_names: list[str],
        channel_last: bool,
        zscore: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_cfg = self.modalities_cfg[modality_name]
        target_channels = int(mod_cfg["in_channels"])
        target_time = int(mod_cfg["sample_rate"]) * self.epoch_seconds

        if x is None:
            out = torch.zeros((target_channels, target_time), dtype=torch.float32)
            channel_mask = torch.zeros(target_channels, dtype=torch.bool)
            out = self._apply_missing_channel_mask(out, channel_mask)
            return out, channel_mask

        if channel_last:
            x = x.transpose(0, 1)
        x = x.float()

        mapped = None
        if self.enabled and self.prefer_name_map and src_channel_names:
            mapped = self._name_map_channels(
                x=x,
                src_channel_names=src_channel_names,
                target_channels=target_channels,
                target_time=target_time,
                modality_name=modality_name,
            )

        # If mapping by channel name fails to match anything, fallback to index-based adaptation.
        if mapped is not None and bool(mapped[1].any().item()):
            out, channel_mask = mapped
        else:
            # Fallback for files without channel labels.
            x = self._resize_time(x, target_time=target_time)
            out, channel_mask = self._resize_channels(x, target_channels=target_channels)

        if zscore:
            out = self._zscore_per_channel(out, channel_mask=channel_mask)

        out = self._apply_missing_channel_mask(out, channel_mask)

        return out, channel_mask
