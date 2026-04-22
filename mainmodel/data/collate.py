from __future__ import annotations

from typing import Any

import torch

from .channel_adapter import ChannelAdapter


class SleepCollator:
    """
    Build model-ready mini-batches.

    Key behavior:
    1) Harmonize channel layout across datasets/devices via ChannelAdapter
    2) Produce modality-level and channel-level masks for missing-data handling
    """

    def __init__(self, data_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> None:
        self.zscore = bool(data_cfg.get("normalization", {}).get("per_sample_zscore", True))
        self.epoch_seconds = int(data_cfg.get("epoch_seconds", 30))
        self.return_sequence = bool(data_cfg.get("return_sequence", False))
        self.sequence_label_pad_value = int(data_cfg.get("sequence_label_pad_value", -100))
        self.modalities_cfg = model_cfg.get("modalities", {})
        self.channel_adapter = ChannelAdapter(data_cfg=data_cfg, model_cfg=model_cfg)

    def _empty_modality_epoch(self, mod_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        mod_cfg = self.modalities_cfg[mod_name]
        target_channels = int(mod_cfg["in_channels"])
        target_time = int(mod_cfg["sample_rate"]) * self.epoch_seconds
        x = torch.zeros((target_channels, target_time), dtype=torch.float32)
        c_mask = torch.zeros((target_channels,), dtype=torch.bool)
        return x, c_mask

    def _collate_epoch_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        labels = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
        dataset_ids = torch.tensor([int(x["dataset_id"]) for x in batch], dtype=torch.long)
        dataset_names = [x.get("dataset_name", "") for x in batch]
        subject_ids = [x.get("subject_id", "") for x in batch]
        sample_ids = [x["sample_id"] for x in batch]

        modalities_out: dict[str, torch.Tensor] = {}
        modality_mask: dict[str, torch.Tensor] = {}
        channel_mask: dict[str, torch.Tensor] = {}

        for mod_name, mod_cfg in self.modalities_cfg.items():
            channel_last = bool(mod_cfg.get("channel_last", False))

            xs = []
            ms = []
            cms = []
            for sample in batch:
                raw_x = sample["modalities"].get(mod_name)
                raw_channel_names = sample.get("channel_names", {}).get(mod_name, [])

                # Align raw channels to the model's canonical channel layout.
                x, c_mask = self.channel_adapter.adapt(
                    modality_name=mod_name,
                    x=raw_x,
                    src_channel_names=raw_channel_names,
                    channel_last=channel_last,
                    zscore=self.zscore,
                )
                # A modality is treated as present if at least one standardized channel is valid.
                present = bool(c_mask.any().item())

                xs.append(x)
                cms.append(c_mask)
                ms.append(present)

            modalities_out[mod_name] = torch.stack(xs, dim=0)
            modality_mask[mod_name] = torch.tensor(ms, dtype=torch.bool)
            channel_mask[mod_name] = torch.stack(cms, dim=0)

        return {
            "modalities": modalities_out,
            "modality_mask": modality_mask,
            "channel_mask": channel_mask,
            "labels": labels,
            "dataset_ids": dataset_ids,
            "dataset_names": dataset_names,
            "subject_ids": subject_ids,
            "sample_ids": sample_ids,
        }

    def _collate_sequence_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        seq_lengths = torch.tensor([int(x.get("seq_len", len(x.get("labels", [])))) for x in batch], dtype=torch.long)
        max_len = int(seq_lengths.max().item()) if batch_size > 0 else 0

        labels = torch.full((batch_size, max_len), fill_value=self.sequence_label_pad_value, dtype=torch.long)
        seq_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

        dataset_ids = torch.tensor([int(x["dataset_id"]) for x in batch], dtype=torch.long)
        dataset_names = [x.get("dataset_name", "") for x in batch]
        subject_ids = [x.get("subject_id", "") for x in batch]
        sample_ids = [x["sample_id"] for x in batch]

        for batch_idx, sample in enumerate(batch):
            y = [int(v) for v in sample.get("labels", [])]
            cur_len = min(len(y), max_len)
            if cur_len > 0:
                labels[batch_idx, :cur_len] = torch.tensor(y[:cur_len], dtype=torch.long)
                seq_padding_mask[batch_idx, :cur_len] = False

        modalities_out: dict[str, torch.Tensor] = {}
        modality_mask: dict[str, torch.Tensor] = {}
        channel_mask: dict[str, torch.Tensor] = {}

        for mod_name, mod_cfg in self.modalities_cfg.items():
            channel_last = bool(mod_cfg.get("channel_last", False))
            x_pad, c_pad = self._empty_modality_epoch(mod_name)
            channels = x_pad.shape[0]
            time_len = x_pad.shape[1]

            x_out = torch.zeros((batch_size, max_len, channels, time_len), dtype=torch.float32)
            m_out = torch.zeros((batch_size, max_len), dtype=torch.bool)
            c_out = torch.zeros((batch_size, max_len, channels), dtype=torch.bool)

            for batch_idx, sample in enumerate(batch):
                seq_x = sample.get("modalities", {}).get(mod_name, [])
                seq_names = sample.get("channel_names", {}).get(mod_name, [])
                cur_len = int(seq_lengths[batch_idx].item())
                for step_idx in range(cur_len):
                    raw_x = seq_x[step_idx] if step_idx < len(seq_x) else None
                    raw_channel_names = seq_names[step_idx] if step_idx < len(seq_names) else []
                    x, c_mask = self.channel_adapter.adapt(
                        modality_name=mod_name,
                        x=raw_x,
                        src_channel_names=raw_channel_names,
                        channel_last=channel_last,
                        zscore=self.zscore,
                    )
                    x_out[batch_idx, step_idx] = x
                    c_out[batch_idx, step_idx] = c_mask
                    m_out[batch_idx, step_idx] = bool(c_mask.any().item())

            modalities_out[mod_name] = x_out
            modality_mask[mod_name] = m_out
            channel_mask[mod_name] = c_out

        return {
            "modalities": modalities_out,
            "modality_mask": modality_mask,
            "channel_mask": channel_mask,
            "labels": labels,
            "seq_padding_mask": seq_padding_mask,
            "seq_lengths": seq_lengths,
            "dataset_ids": dataset_ids,
            "dataset_names": dataset_names,
            "subject_ids": subject_ids,
            "sample_ids": sample_ids,
        }

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if self.return_sequence:
            return self._collate_sequence_batch(batch)
        return self._collate_epoch_batch(batch)
