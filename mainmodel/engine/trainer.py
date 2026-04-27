from __future__ import annotations

from itertools import combinations
import math
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.nn import all_reduce as differentiable_all_reduce
from torch.distributed.nn import ReduceOp

from mainmodel.utils.distributed import is_main_process
from mainmodel.utils.metrics import metrics_from_confusion, update_confusion


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _normalize_stage(stage: str) -> str:
    return str(stage).strip().lower()


def _is_stage1_eeg_jepa(stage: str) -> bool:
    return _normalize_stage(stage) in {"stage1_eeg_jepa", "stage1", "eeg_jepa"}


def _is_stage2_multimodal_pretrain(stage: str) -> bool:
    return _normalize_stage(stage) in {"pretrain", "stage2_multimodal_pretrain", "stage2"}


def _is_finetune_stage(stage: str) -> bool:
    normalized = _normalize_stage(stage)
    return normalized in {"finetune", "stage3_finetune", "stage3_downstream_finetune", "stage3"}


def _normalize_task_name(task: str | None) -> str:
    if task is None:
        return "sleep_staging"
    return str(task).strip().lower()


def _ddp_all_reduce(x: torch.Tensor, op: str = "AVG") -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        reduce_op = ReduceOp.__dict__[op.upper()]
        return differentiable_all_reduce(x, reduce_op)
    return x


def build_optimizer(cfg: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    training_cfg = cfg["training"]
    optimizer_cfg = training_cfg.get("optimizer", {})
    lr = float(optimizer_cfg.get("lr", 3e-4))
    weight_decay = float(optimizer_cfg.get("weight_decay", 1e-2))
    betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))

    stage = cfg.get("experiment", {}).get("stage", "stage2_multimodal_pretrain")
    lr_scale_backbone = float(cfg.get("finetune", {}).get("lr_scale_backbone", 0.3))

    core_model = _unwrap_model(model)
    backbone_params = []
    head_params = []
    for name, p in core_model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier") or name.startswith("temporal_encoder") or name.startswith("temporal_norm"):
            head_params.append(p)
        else:
            backbone_params.append(p)

    if _is_finetune_stage(stage):
        param_groups = []
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": lr * lr_scale_backbone,
                    "weight_decay": weight_decay,
                }
            )
        if head_params:
            param_groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
    else:
        param_groups = [{"params": [p for p in core_model.parameters() if p.requires_grad], "lr": lr, "weight_decay": weight_decay}]

    return torch.optim.AdamW(param_groups, betas=betas)


def build_scheduler(
    cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    train_loader_len: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR | None, int]:
    if train_loader_len <= 0:
        return None, 0

    training_cfg = cfg["training"]
    scheduler_cfg = training_cfg.get("scheduler", {})
    base_lr = float(training_cfg.get("optimizer", {}).get("lr", 3e-4))
    min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
    epochs = int(training_cfg.get("epochs", 1))
    grad_accum = int(training_cfg.get("grad_accum_steps", 1))
    warmup_epochs = float(scheduler_cfg.get("warmup_epochs", 0))
    warmup_steps_cfg = int(scheduler_cfg.get("warmup_steps", 0))
    warmup_ratio = float(scheduler_cfg.get("warmup_ratio", 0.0))

    steps_per_epoch = math.ceil(train_loader_len / max(1, grad_accum))
    total_steps = max(1, steps_per_epoch * epochs)
    if warmup_steps_cfg > 0:
        warmup_steps = warmup_steps_cfg
    elif warmup_ratio > 0:
        warmup_steps = int(total_steps * warmup_ratio)
    else:
        warmup_steps = int(steps_per_epoch * warmup_epochs)
    warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))
    min_ratio = min(1.0, max(0.0, min_lr / max(base_lr, 1e-12)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, total_steps


class DDPTrainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR | None,
        scaler: torch.cuda.amp.GradScaler | None,
        logger: Any,
        device: torch.device,
        output_dir: str,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.logger = logger
        self.device = device
        self.output_dir = output_dir
        self.amp_dtype = amp_dtype

        self.stage = _normalize_stage(cfg.get("experiment", {}).get("stage", "stage2_multimodal_pretrain"))
        self.training_cfg = cfg["training"]
        self.downstream_task = _normalize_task_name(self.training_cfg.get("downstream_task", "sleep_staging"))
        self.downstream_task_alias = {
            "staging": "sleep_staging",
            "sleep_stage_classification": "sleep_staging",
        }
        self.stage1_cfg = self.training_cfg.get("stage1_eeg_jepa", {})
        self.stage2_cfg = self.training_cfg.get("stage2_multimodal", {})

        self.view_dropout_cfg = self.training_cfg.get("view_dropout", {})

        self.num_classes = self._resolve_num_classes(cfg["model"])
        self.global_step = 0
        self.sequence_label_pad_value = int(cfg.get("data", {}).get("sequence_label_pad_value", -100))

        class_weights = self.training_cfg.get("class_weights", [])
        if class_weights:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=self.sequence_label_pad_value)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.sequence_label_pad_value)

        self.stage1_eeg_groups = self._resolve_stage1_eeg_groups()
        self.stage1_single_channel_indices = self._resolve_stage1_single_channel_indices()

        self.stage1_num_global_views = max(1, int(self.stage1_cfg.get("num_global_views", len(self.stage1_cfg.get("global_views", [1, 2])))))
        self.stage1_lambda = float(self.stage1_cfg.get("lamb", self.stage1_cfg.get("lambda_sigreg", 0.05)))
        self.stage1_lambda = min(1.0, max(0.0, self.stage1_lambda))
        self.stage1_sigreg_cfg = self.stage1_cfg.get("sigreg", {})

        self.stage2_num_missing_views = int(self.stage2_cfg.get("planned_missing_views", 6))
        self.stage2_global_views = self.stage2_cfg.get(
            "global_views",
            ["all_modalities_present", "augmented_all_modalities_present"],
        )
        if not isinstance(self.stage2_global_views, list):
            self.stage2_global_views = [self.stage2_global_views]
        self.stage2_num_global_views = max(
            1,
            int(self.stage2_cfg.get("num_global_views", len(self.stage2_global_views))),
        )
        self.stage2_local_views = self.stage2_cfg.get(
            "local_views",
            ["eeg_only", "eeg_plus_eog", "missing_any_one_or_two_modalities"],
        )
        self.stage2_view_seed = int(self.stage2_cfg.get("view_seed", cfg.get("experiment", {}).get("seed", 42)))
        self.stage2_keep_set_protected_modalities = {
            str(x).strip().lower()
            for x in self.stage2_cfg.get("protected_modalities", [])
            if str(x).strip()
        }
        self.view_dropout_protected_modalities = {
            str(x).strip().lower()
            for x in self.view_dropout_cfg.get("protected_modalities", [])
            if str(x).strip()
        }
        self.stage2_global_aug_cfg = self.stage2_cfg.get("global_augmentation", {})
        self.modality_sample_rates = {
            str(mod_name).strip().lower(): int(mod_cfg.get("sample_rate", 1))
            for mod_name, mod_cfg in cfg.get("model", {}).get("modalities", {}).items()
        }
        self.stage2_lambda = float(self.stage2_cfg.get("lamb", self.stage2_cfg.get("lambda_sigreg", 0.05)))
        self.stage2_lambda = min(1.0, max(0.0, self.stage2_lambda))
        self.stage2_sigreg_cfg = self.stage2_cfg.get("sigreg", {})
        self._stage2_keep_sets_logged = False

    def _use_multi_view_pretrain_objective(self) -> bool:
        return _is_stage1_eeg_jepa(self.stage) or _is_stage2_multimodal_pretrain(self.stage)

    def _set_batch_norm_eval(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _normalized_downstream_task(self) -> str:
        normalized = _normalize_task_name(self.downstream_task)
        return self.downstream_task_alias.get(normalized, normalized)

    @staticmethod
    def _infer_head_output_dim(head: nn.Module) -> int | None:
        if hasattr(head, "out_features"):
            return int(head.out_features)
        for module in reversed(list(head.modules())):
            if isinstance(module, nn.Linear):
                return int(module.out_features)
        return None

    def _resolve_num_classes(self, model_cfg: dict[str, Any]) -> int | None:
        task_cfg = model_cfg.get("downstream_tasks", {})
        task_name = self._normalized_downstream_task()
        if isinstance(task_cfg, dict):
            task_head_cfg = task_cfg.get(task_name, {})
            if isinstance(task_head_cfg, dict) and "num_classes" in task_head_cfg:
                return int(task_head_cfg["num_classes"])
        if "num_classes" in model_cfg and model_cfg.get("num_classes") is not None:
            return int(model_cfg["num_classes"])

        core_model = _unwrap_model(self.model)
        if hasattr(core_model, "get_task_head"):
            try:
                head, _ = core_model.get_task_head(task_name)
                return self._infer_head_output_dim(head)
            except Exception:
                return None
        classifier = getattr(core_model, "classifier", None)
        if classifier is not None:
            return self._infer_head_output_dim(classifier)
        return None

    def _resolve_stage1_eeg_groups(self) -> dict[str, list[int]]:
        core_model = _unwrap_model(self.model)
        eeg_in_channels = int(core_model.model_cfg.get("modalities", {}).get("eeg", {}).get("in_channels", 0))
        schema = (
            self.cfg.get("data", {})
            .get("channel_adapter", {})
            .get("modality_channel_schema", {})
            .get("eeg", {})
        )
        canonical = [str(x) for x in schema.get("canonical_channels", [])]
        if len(canonical) < eeg_in_channels:
            canonical = canonical + [f"CH{i}" for i in range(len(canonical), eeg_in_channels)]
        canonical = canonical[:eeg_in_channels]

        groups = {"C": [], "P": [], "O": []}
        for idx, name in enumerate(canonical):
            norm = "".join(ch for ch in name.upper() if ch.isalnum())
            if "C" in norm:
                groups["C"].append(idx)
            if "P" in norm:
                groups["P"].append(idx)
            if "O" in norm:
                groups["O"].append(idx)

        if eeg_in_channels > 0 and any(len(v) == 0 for v in groups.values()):
            thirds = max(1, eeg_in_channels // 3)
            fallback = {
                "C": list(range(0, min(eeg_in_channels, thirds))),
                "P": list(range(min(eeg_in_channels, thirds), min(eeg_in_channels, 2 * thirds))),
                "O": list(range(min(eeg_in_channels, 2 * thirds), eeg_in_channels)),
            }
            for key in groups:
                if len(groups[key]) == 0:
                    groups[key] = fallback[key]

        for key in groups:
            groups[key] = sorted(set(groups[key]))
        return groups

    def _resolve_stage1_single_channel_indices(self) -> list[int]:
        core_model = _unwrap_model(self.model)
        eeg_in_channels = int(core_model.model_cfg.get("modalities", {}).get("eeg", {}).get("in_channels", 0))
        ordered = []
        for key in ("C", "P", "O"):
            ordered.extend(self.stage1_eeg_groups.get(key, []))
        seen = set()
        dedup = []
        for idx in ordered:
            if idx not in seen:
                dedup.append(idx)
                seen.add(idx)
        for idx in range(eeg_in_channels):
            if idx not in seen:
                dedup.append(idx)
        return dedup

    def _augment_eeg_weak(self, x: torch.Tensor, channel_mask: torch.Tensor | None) -> torch.Tensor:
        out = x.clone()
        jitter_std = float(self.stage1_cfg.get("weak_jitter_std", 0.02))
        scale_low = float(self.stage1_cfg.get("weak_scale_low", 0.95))
        scale_high = float(self.stage1_cfg.get("weak_scale_high", 1.05))
        time_mask_prob = float(self.stage1_cfg.get("weak_time_mask_prob", 0.02))

        if jitter_std > 0:
            out = out + torch.randn_like(out) * jitter_std
        if scale_high > scale_low:
            scale = torch.empty((out.shape[0], out.shape[1], 1), device=out.device, dtype=out.dtype).uniform_(scale_low, scale_high)
            out = out * scale
        if time_mask_prob > 0:
            t_mask = torch.rand((out.shape[0], 1, out.shape[2]), device=out.device) < time_mask_prob
            out = out.masked_fill(t_mask, 0.0)
        if channel_mask is not None:
            out = out * channel_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out

    def _mask_to_selected_eeg_channels(
        self,
        eeg: torch.Tensor,
        keep_indices: list[int],
        channel_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        out = torch.zeros_like(eeg)
        if len(keep_indices) == 0:
            return out
        idx = torch.tensor(keep_indices, device=eeg.device, dtype=torch.long)
        out[:, idx, :] = eeg[:, idx, :]
        if channel_mask is not None:
            valid = channel_mask[:, idx].unsqueeze(-1).to(dtype=eeg.dtype)
            out[:, idx, :] = out[:, idx, :] * valid
        return out

    def _build_stage1_local_views(self, eeg: torch.Tensor, channel_mask: torch.Tensor | None) -> list[torch.Tensor]:
        local_cfg = self.stage1_cfg.get("local_views", {})
        multi_cfg = local_cfg.get("multi_channel", {})
        single_cfg = local_cfg.get("single_channel", {})

        multi_groups = [str(x).upper() for x in multi_cfg.get("groups", ["C", "P", "O"])]
        num_multi = int(multi_cfg.get("num_views", 3))
        num_single = int(single_cfg.get("num_views", 6))

        views: list[torch.Tensor] = []
        if len(multi_groups) == 0:
            multi_groups = ["C", "P", "O"]

        for i in range(max(0, num_multi)):
            group_name = multi_groups[i % len(multi_groups)]
            idx = self.stage1_eeg_groups.get(group_name, [])
            local = self._mask_to_selected_eeg_channels(eeg=eeg, keep_indices=idx, channel_mask=channel_mask)
            local = self._augment_eeg_weak(local, channel_mask=channel_mask)
            views.append(local)

        if len(self.stage1_single_channel_indices) > 0 and num_single > 0:
            limit = min(num_single, len(self.stage1_single_channel_indices))
            for i in range(limit):
                idx = [self.stage1_single_channel_indices[i]]
                local = self._mask_to_selected_eeg_channels(eeg=eeg, keep_indices=idx, channel_mask=channel_mask)
                local = self._augment_eeg_weak(local, channel_mask=channel_mask)
                views.append(local)
        return views

    def _build_stage2_view(
        self,
        batch: dict[str, Any],
        keep_modalities: set[str],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        modalities_view: dict[str, torch.Tensor] = {}
        mask_view: dict[str, torch.Tensor] = {}
        channel_view: dict[str, torch.Tensor] = {}
        core_model = _unwrap_model(self.model)
        for mod_name in core_model.modality_names:
            x = batch["modalities"][mod_name]
            m = batch["modality_mask"][mod_name]
            c = batch.get("channel_mask", {}).get(mod_name)
            if mod_name in keep_modalities:
                modalities_view[mod_name] = x
                mask_view[mod_name] = m
                if c is not None:
                    channel_view[mod_name] = c
            else:
                modalities_view[mod_name] = torch.zeros_like(x)
                mask_view[mod_name] = torch.zeros_like(m)
                if c is not None:
                    channel_view[mod_name] = torch.zeros_like(c)
        return modalities_view, mask_view, channel_view

    def _clone_batch_for_view(self, batch: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in batch.items():
            if key in {"modalities", "modality_mask", "channel_mask"} and isinstance(value, dict):
                cloned[key] = {name: tensor.clone() for name, tensor in value.items()}
            else:
                cloned[key] = value
        return cloned

    def _resolve_stage2_global_view_spec(self, view_index: int) -> str:
        if len(self.stage2_global_views) == 0:
            return "all_modalities_present"
        spec = self.stage2_global_views[min(view_index, len(self.stage2_global_views) - 1)]
        return str(spec).strip().lower()

    def _resolve_stage2_global_aug_op_cfg(self, mod_name: str, op_name: str) -> dict[str, Any]:
        base_cfg = self.stage2_global_aug_cfg.get(op_name, {})
        if not isinstance(base_cfg, dict):
            base_cfg = {}
        overrides = self.stage2_global_aug_cfg.get("modality_overrides", {})
        mod_override = overrides.get(mod_name, {}) if isinstance(overrides, dict) else {}
        op_override = mod_override.get(op_name, {}) if isinstance(mod_override, dict) else {}
        if not isinstance(op_override, dict):
            op_override = {}
        merged = dict(base_cfg)
        merged.update(op_override)
        return merged

    @staticmethod
    def _roll_with_zero_fill(x: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return x
        out = torch.roll(x, shifts=shift, dims=-1)
        if shift > 0:
            out[..., :shift] = 0.0
        else:
            out[..., shift:] = 0.0
        return out

    @staticmethod
    def _apply_band_stop_to_signal(
        x: torch.Tensor,
        sample_rate: int,
        low_hz: float,
        high_hz: float,
        attenuation: float,
    ) -> torch.Tensor:
        if sample_rate <= 0 or x.shape[-1] <= 1 or high_hz <= low_hz:
            return x
        spec = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / float(sample_rate)).to(device=x.device)
        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not bool(band_mask.any().item()):
            return x
        spec[..., band_mask] = spec[..., band_mask] * attenuation
        filtered = torch.fft.irfft(spec, n=x.shape[-1], dim=-1)
        return filtered.to(dtype=x.dtype)

    def _augment_stage2_modality_signal(
        self,
        mod_name: str,
        x: torch.Tensor,
        modality_mask: torch.Tensor,
        channel_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not bool(self.stage2_global_aug_cfg.get("enabled", False)):
            return x

        sample_rate = max(1, int(self.modality_sample_rates.get(mod_name, 1)))
        default_prob = float(self.stage2_global_aug_cfg.get("per_transform_prob", 0.5))

        original_shape = x.shape
        if x.ndim == 3:
            flat_x = x.clone()
            flat_mask = modality_mask.reshape(-1).to(device=x.device, dtype=torch.bool)
            flat_channel_mask = channel_mask.reshape(-1, channel_mask.shape[-1]) if channel_mask is not None else None
        elif x.ndim == 4:
            flat_x = x.reshape(-1, x.shape[-2], x.shape[-1]).clone()
            flat_mask = modality_mask.reshape(-1).to(device=x.device, dtype=torch.bool)
            flat_channel_mask = channel_mask.reshape(-1, channel_mask.shape[-1]) if channel_mask is not None else None
        else:
            return x

        present_idx = torch.where(flat_mask)[0]
        if present_idx.numel() == 0:
            return x
        if flat_channel_mask is not None:
            flat_x = flat_x * flat_channel_mask.unsqueeze(-1).to(dtype=flat_x.dtype)

        num_steps = flat_x.shape[-1]

        def _sample_selected(prob: float) -> torch.Tensor:
            prob = min(1.0, max(0.0, prob))
            if prob <= 0.0 or present_idx.numel() == 0:
                return present_idx[:0]
            select_mask = torch.rand((present_idx.numel(),), device=flat_x.device) < prob
            return present_idx[select_mask]

        scale_cfg = self._resolve_stage2_global_aug_op_cfg(mod_name, "amplitude_scaling")
        if bool(scale_cfg.get("enabled", True)):
            selected = _sample_selected(float(scale_cfg.get("prob", default_prob)))
            if selected.numel() > 0:
                scale_min = float(scale_cfg.get("min", 0.8))
                scale_max = float(scale_cfg.get("max", 1.2))
                if scale_max < scale_min:
                    scale_min, scale_max = scale_max, scale_min
                scales = torch.empty((selected.numel(), 1, 1), device=flat_x.device, dtype=flat_x.dtype).uniform_(scale_min, scale_max)
                flat_x[selected] = flat_x[selected] * scales

        shift_cfg = self._resolve_stage2_global_aug_op_cfg(mod_name, "time_shift")
        if bool(shift_cfg.get("enabled", True)):
            selected = _sample_selected(float(shift_cfg.get("prob", default_prob)))
            max_seconds = max(0.0, float(shift_cfg.get("max_seconds", 3.0)))
            max_shift = min(max(0, num_steps - 1), int(round(sample_rate * max_seconds)))
            if selected.numel() > 0 and max_shift > 0:
                shifts = torch.randint(-max_shift, max_shift + 1, (selected.numel(),), device=flat_x.device)
                for row_idx, shift in zip(selected.tolist(), shifts.tolist()):
                    flat_x[row_idx] = self._roll_with_zero_fill(flat_x[row_idx], int(shift))

        shift_bias_cfg = self._resolve_stage2_global_aug_op_cfg(mod_name, "amplitude_shift")
        if bool(shift_bias_cfg.get("enabled", True)):
            selected = _sample_selected(float(shift_bias_cfg.get("prob", default_prob)))
            if selected.numel() > 0:
                bias_min = float(shift_bias_cfg.get("min", -0.2))
                bias_max = float(shift_bias_cfg.get("max", 0.2))
                if bias_max < bias_min:
                    bias_min, bias_max = bias_max, bias_min
                bias = torch.empty((selected.numel(), 1, 1), device=flat_x.device, dtype=flat_x.dtype).uniform_(bias_min, bias_max)
                flat_x[selected] = flat_x[selected] + bias

        zero_mask_cfg = self._resolve_stage2_global_aug_op_cfg(mod_name, "zero_mask")
        if bool(zero_mask_cfg.get("enabled", True)):
            selected = _sample_selected(float(zero_mask_cfg.get("prob", default_prob)))
            max_seconds = max(0.0, float(zero_mask_cfg.get("max_seconds", 3.0)))
            max_mask_len = min(num_steps, int(round(sample_rate * max_seconds)))
            if selected.numel() > 0 and max_mask_len > 0:
                mask_lens = torch.randint(0, max_mask_len + 1, (selected.numel(),), device=flat_x.device)
                for row_idx, mask_len in zip(selected.tolist(), mask_lens.tolist()):
                    if mask_len <= 0:
                        continue
                    start_max = max(1, num_steps - int(mask_len) + 1)
                    start = int(torch.randint(start_max, (1,), device=flat_x.device).item())
                    flat_x[row_idx, :, start : start + int(mask_len)] = 0.0

        noise_cfg = self._resolve_stage2_global_aug_op_cfg(mod_name, "additive_gaussian_noise")
        if bool(noise_cfg.get("enabled", True)):
            selected = _sample_selected(float(noise_cfg.get("prob", default_prob)))
            if selected.numel() > 0:
                sigma_min = max(0.0, float(noise_cfg.get("sigma_min", 0.0)))
                sigma_max = max(sigma_min, float(noise_cfg.get("sigma_max", 0.1)))
                sigma = torch.empty((selected.numel(), 1, 1), device=flat_x.device, dtype=flat_x.dtype).uniform_(sigma_min, sigma_max)
                noise = torch.randn_like(flat_x[selected]) * sigma
                flat_x[selected] = flat_x[selected] + noise

        band_stop_cfg = self._resolve_stage2_global_aug_op_cfg(mod_name, "band_stop")
        if bool(band_stop_cfg.get("enabled", True)):
            min_sample_rate = max(1, int(band_stop_cfg.get("min_sample_rate", 20)))
            if sample_rate >= min_sample_rate:
                selected = _sample_selected(float(band_stop_cfg.get("prob", default_prob)))
                band_width_hz = max(0.0, float(band_stop_cfg.get("width_hz", 2.0)))
                lower_min = max(0.0, float(band_stop_cfg.get("lower_hz_min", 0.5)))
                lower_max = max(lower_min, float(band_stop_cfg.get("lower_hz_max", 30.0)))
                attenuation = float(band_stop_cfg.get("attenuation", 0.0))
                nyquist = 0.5 * float(sample_rate)
                effective_lower_max = min(lower_max, max(lower_min, nyquist - band_width_hz))
                if selected.numel() > 0 and band_width_hz > 0 and effective_lower_max >= lower_min:
                    for row_idx in selected.tolist():
                        if effective_lower_max == lower_min:
                            low_hz = lower_min
                        else:
                            low_hz = float(
                                torch.empty((1,), device=flat_x.device, dtype=flat_x.dtype).uniform_(
                                    lower_min,
                                    effective_lower_max,
                                ).item()
                            )
                        high_hz = min(nyquist, low_hz + band_width_hz)
                        if high_hz <= low_hz:
                            continue
                        flat_x[row_idx] = self._apply_band_stop_to_signal(
                            flat_x[row_idx],
                            sample_rate=sample_rate,
                            low_hz=low_hz,
                            high_hz=high_hz,
                            attenuation=attenuation,
                        )

        if flat_channel_mask is not None:
            flat_x = flat_x * flat_channel_mask.unsqueeze(-1).to(dtype=flat_x.dtype)

        if len(original_shape) == 3:
            return flat_x
        return flat_x.reshape(original_shape)

    def _apply_stage2_global_augmentation(
        self,
        batch: dict[str, Any],
        training: bool,
    ) -> None:
        if not training:
            return
        if not bool(self.stage2_global_aug_cfg.get("enabled", False)):
            return
        core_model = _unwrap_model(self.model)
        for mod_name in core_model.modality_names:
            x = batch["modalities"].get(mod_name, None)
            m = batch["modality_mask"].get(mod_name, None)
            if x is None or m is None:
                continue
            c = batch.get("channel_mask", {}).get(mod_name, None)
            batch["modalities"][mod_name] = self._augment_stage2_modality_signal(
                mod_name=mod_name,
                x=x,
                modality_mask=m,
                channel_mask=c,
            )

    def _build_stage2_global_view(
        self,
        batch: dict[str, Any],
        training: bool,
        view_index: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        view_batch = self._clone_batch_for_view(batch)
        spec = self._resolve_stage2_global_view_spec(view_index)
        if spec in {"augmented_all_modalities_present", "augmented_all_modalities", "augmented"}:
            self._apply_stage2_global_augmentation(view_batch, training=training)
        return (
            view_batch["modalities"],
            view_batch["modality_mask"],
            view_batch.get("channel_mask"),
        )

    def _build_stage2_keep_sets(self) -> list[set[str]]:
        core_model = _unwrap_model(self.model)
        all_modalities = list(core_model.modality_names)
        if len(all_modalities) == 0:
            return []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.stage2_view_seed + int(self.global_step))
        protected_modalities = {m for m in self.stage2_keep_set_protected_modalities if m in all_modalities}
        drop_candidates = [m for m in all_modalities if m not in protected_modalities]

        def _missing_any_one_or_two(num_views: int) -> list[set[str]]:
            out: list[set[str]] = []
            requested = max(0, num_views)
            if len(drop_candidates) == 0:
                return [set(all_modalities)] if requested > 0 else []

            one_drop_choices = [{mod_name} for mod_name in drop_candidates]
            two_drop_choices = [set(pair) for pair in combinations(drop_candidates, 2)]
            seen_drop_keys: set[tuple[str, ...]] = set()

            for i in range(requested):
                desired_drop_n = 1 if i % 2 == 0 else 2
                if desired_drop_n == 1 or len(two_drop_choices) == 0:
                    candidate_pool = one_drop_choices
                else:
                    candidate_pool = two_drop_choices

                available = [drop for drop in candidate_pool if tuple(sorted(drop)) not in seen_drop_keys]
                if len(available) == 0:
                    fallback_pool = one_drop_choices if len(one_drop_choices) > 0 else [set(drop_candidates)]
                    available = [drop for drop in fallback_pool if tuple(sorted(drop)) not in seen_drop_keys]

                if len(available) > 0:
                    choice_idx = int(torch.randint(len(available), (1,), generator=generator).item())
                    drop_set = set(available[choice_idx])
                else:
                    max_drop = max(1, len(drop_candidates))
                    actual_drop_n = min(desired_drop_n, max_drop)
                    perm = torch.randperm(len(drop_candidates), generator=generator).tolist()
                    drop_set = {drop_candidates[j] for j in perm[:actual_drop_n]}

                seen_drop_keys.add(tuple(sorted(drop_set)))
                keep = set(all_modalities) - drop_set
                keep.update(protected_modalities)
                if len(keep) == 0:
                    keep = set(protected_modalities)
                    if len(keep) == 0:
                        fallback_idx = int(torch.randint(len(drop_candidates), (1,), generator=generator).item())
                        keep = {drop_candidates[fallback_idx]}
                out.append(keep)
            return out

        def _apply_protected(keep: set[str]) -> set[str]:
            keep = {m for m in keep if m in all_modalities}
            keep.update(protected_modalities)
            return keep

        def _random_drop_from_subset(
            base_keep: set[str],
            drop_from: list[str],
            num_views: int,
            min_drop: int,
            max_drop: int,
        ) -> list[set[str]]:
            out: list[set[str]] = []
            requested = max(0, num_views)
            if requested == 0:
                return out

            keep_base = _apply_protected(base_keep)
            subset = [m for m in drop_from if m in all_modalities and m not in keep_base]
            if len(subset) == 0:
                return [keep_base] if len(keep_base) > 0 else [set(all_modalities)]

            drop_sizes = [size for size in range(max(1, min_drop), max(1, max_drop) + 1) if size <= len(subset)]
            if len(drop_sizes) == 0:
                drop_sizes = [1]
            drop_pool = [set(combo) for size in drop_sizes for combo in combinations(subset, size)]
            seen_drop_keys: set[tuple[str, ...]] = set()

            for _ in range(requested):
                available = [drop for drop in drop_pool if tuple(sorted(drop)) not in seen_drop_keys]
                if len(available) > 0:
                    choice_idx = int(torch.randint(len(available), (1,), generator=generator).item())
                    drop_set = set(available[choice_idx])
                else:
                    drop_n = drop_sizes[int(torch.randint(len(drop_sizes), (1,), generator=generator).item())]
                    perm = torch.randperm(len(subset), generator=generator).tolist()
                    drop_set = {subset[j] for j in perm[:drop_n]}

                seen_drop_keys.add(tuple(sorted(drop_set)))
                keep = _apply_protected(set(all_modalities) - drop_set)
                keep.update(keep_base)
                out.append(_apply_protected(keep))
            return out

        def _keep_from_spec(spec: Any) -> list[set[str]]:
            if isinstance(spec, dict):
                spec_type = str(spec.get("type", "")).strip().lower()
                if spec_type == "missing_any_one_or_two_modalities":
                    n = int(spec.get("num_views", self.stage2_num_missing_views))
                    return _missing_any_one_or_two(n)
                if spec_type in {"drop_random_non_core_modalities", "drop_random_modalities"}:
                    base_keep = {str(x).strip().lower() for x in spec.get("keep", []) if str(x).strip()}
                    drop_from = [str(x).strip().lower() for x in spec.get("drop_from", []) if str(x).strip()]
                    if len(drop_from) == 0:
                        drop_from = [m for m in all_modalities if m not in base_keep]
                    return _random_drop_from_subset(
                        base_keep=base_keep,
                        drop_from=drop_from,
                        num_views=int(spec.get("num_views", 1)),
                        min_drop=int(spec.get("min_drop", 1)),
                        max_drop=int(spec.get("max_drop", spec.get("min_drop", 1))),
                    )
                if "keep" in spec:
                    keep = {str(x).strip().lower() for x in spec.get("keep", [])}
                    keep = _apply_protected(keep)
                    return [keep] if len(keep) > 0 else []
                if "drop" in spec:
                    drop = {str(x).strip().lower() for x in spec.get("drop", [])}
                    keep = _apply_protected(set(all_modalities) - drop)
                    return [keep] if len(keep) > 0 else []
                return []

            text = str(spec).strip().lower()
            if text == "":
                return []
            if text == "all_modalities_present":
                return [set(all_modalities)]
            if text == "eeg_only":
                return [_apply_protected({"eeg"})] if "eeg" in all_modalities else []
            if text in {"eeg_plus_eog", "eeg+eog"}:
                keep = {m for m in ["eeg", "eog"] if m in all_modalities}
                keep = _apply_protected(keep)
                return [keep] if len(keep) > 0 else []
            if text == "missing_any_one_or_two_modalities":
                return _missing_any_one_or_two(self.stage2_num_missing_views)
            if text.startswith("keep:"):
                keep = {x.strip().lower() for x in text.replace("keep:", "", 1).split(",") if x.strip()}
                keep = _apply_protected(keep)
                return [keep] if len(keep) > 0 else []
            if text.startswith("drop:"):
                drop = {x.strip().lower() for x in text.replace("drop:", "", 1).split(",") if x.strip()}
                keep = _apply_protected(set(all_modalities) - drop)
                return [keep] if len(keep) > 0 else []
            tokens = {x.strip().lower() for x in text.split(",") if x.strip()}
            keep = _apply_protected(tokens)
            return [keep] if len(keep) > 0 else []

        keep_sets: list[set[str]] = []
        specs = self.stage2_local_views
        if not isinstance(specs, list):
            specs = [specs]
        for spec in specs:
            keep_sets.extend(_keep_from_spec(spec))
        if len(keep_sets) == 0:
            keep_sets = [{"eeg"}] if "eeg" in all_modalities else [set(all_modalities)]

        dedup = []
        seen = set()
        for keep in keep_sets:
            key = tuple(sorted(keep))
            if key not in seen:
                dedup.append(keep)
                seen.add(key)
        return dedup

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        for key in batch["modalities"]:
            batch["modalities"][key] = batch["modalities"][key].to(self.device, non_blocking=True)
        for key in batch["modality_mask"]:
            batch["modality_mask"][key] = batch["modality_mask"][key].to(self.device, non_blocking=True)
        if "channel_mask" in batch:
            for key in batch["channel_mask"]:
                batch["channel_mask"][key] = batch["channel_mask"][key].to(self.device, non_blocking=True)
        if "seq_padding_mask" in batch:
            batch["seq_padding_mask"] = batch["seq_padding_mask"].to(self.device, non_blocking=True)
        if "seq_lengths" in batch:
            batch["seq_lengths"] = batch["seq_lengths"].to(self.device, non_blocking=True)
        batch["labels"] = batch["labels"].to(self.device, non_blocking=True)
        batch["dataset_ids"] = batch["dataset_ids"].to(self.device, non_blocking=True)
        return batch

    def _resolve_view_dropout_mode_probs(self) -> dict[str, float]:
        raw = self.view_dropout_cfg.get("mode_probs", None)
        if isinstance(raw, dict) and len(raw) > 0:
            probs = {
                "none": float(raw.get("none", 0.0)),
                "channel": float(raw.get("channel", raw.get("channel_only", 0.0))),
                "modality": float(raw.get("modality", raw.get("modality_only", 0.0))),
                "both": float(raw.get("both", 0.0)),
            }
        else:
            p_mod = min(1.0, max(0.0, float(self.view_dropout_cfg.get("random_modality_drop_prob", 0.0))))
            p_ch = min(1.0, max(0.0, float(self.view_dropout_cfg.get("random_channel_drop_prob", 0.0))))
            probs = {
                "none": (1.0 - p_mod) * (1.0 - p_ch),
                "channel": (1.0 - p_mod) * p_ch,
                "modality": p_mod * (1.0 - p_ch),
                "both": p_mod * p_ch,
            }

        probs = {key: max(0.0, value) for key, value in probs.items()}
        total = sum(probs.values())
        if total <= 0:
            return {"none": 1.0, "channel": 0.0, "modality": 0.0, "both": 0.0}
        return {key: value / total for key, value in probs.items()}

    def _sample_view_dropout_masks(
        self,
        shape: torch.Size,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self._resolve_view_dropout_mode_probs()
        r = torch.rand(shape, device=device)
        p_none = probs["none"]
        p_channel = probs["channel"]
        p_modality = probs["modality"]
        channel_mask = (r >= p_none) & (r < p_none + p_channel)
        modality_mask = (r >= p_none + p_channel) & (r < p_none + p_channel + p_modality)
        both_mask = r >= (p_none + p_channel + p_modality)
        return modality_mask | both_mask, channel_mask | both_mask

    def _apply_modality_dropout_positions(
        self,
        batch: dict[str, Any],
        target_mask: torch.Tensor,
    ) -> None:
        mod_names = list(batch["modalities"].keys())
        if not mod_names or not target_mask.any():
            return

        positions = torch.nonzero(target_mask, as_tuple=False)
        protected = self.view_dropout_protected_modalities
        for pos in positions:
            coord = tuple(int(x.item()) for x in pos)
            present = [
                mod_name
                for mod_name in mod_names
                if bool(batch["modality_mask"][mod_name][coord].item())
            ]
            if len(present) <= 1:
                continue
            candidates = [mod_name for mod_name in present if mod_name not in protected]
            if len(candidates) == 0:
                continue
            selected = candidates[int(torch.randint(len(candidates), (1,), device=target_mask.device).item())]
            batch["modalities"][selected][coord] = 0.0
            batch["modality_mask"][selected][coord] = False
            c_mask = batch.get("channel_mask", {}).get(selected, None)
            if c_mask is not None:
                c_mask[coord] = False

    def _apply_channel_dropout_positions(
        self,
        batch: dict[str, Any],
        target_mask: torch.Tensor,
    ) -> None:
        if not target_mask.any():
            return

        channel_cfg = self.view_dropout_cfg.get("channel_drop", {})
        candidate_modalities = {
            str(x).strip().lower()
            for x in channel_cfg.get("candidate_modalities", [])
            if str(x).strip()
        }
        protected_channel_modalities = {
            str(x).strip().lower()
            for x in channel_cfg.get("protected_modalities", [])
            if str(x).strip()
        }
        min_keep = max(1, int(channel_cfg.get("min_channels_to_keep", 1)))
        min_drop = max(1, int(channel_cfg.get("num_channels_min", channel_cfg.get("num_channels", 1))))
        max_drop = max(min_drop, int(channel_cfg.get("num_channels_max", min_drop)))

        mod_names = [
            mod_name
            for mod_name in batch["modalities"].keys()
            if (not candidate_modalities or mod_name in candidate_modalities)
            and mod_name not in protected_channel_modalities
        ]
        if not mod_names:
            return

        positions = torch.nonzero(target_mask, as_tuple=False)
        for pos in positions:
            coord = tuple(int(x.item()) for x in pos)
            eligible: list[tuple[str, torch.Tensor]] = []
            for mod_name in mod_names:
                if not bool(batch["modality_mask"][mod_name][coord].item()):
                    continue
                x = batch["modalities"][mod_name]
                channels = int(x.shape[-2])
                c_mask = batch.get("channel_mask", {}).get(mod_name, None)
                if c_mask is None:
                    valid_channels = torch.ones(channels, dtype=torch.bool, device=x.device)
                else:
                    valid_channels = c_mask[coord].to(dtype=torch.bool, device=x.device)
                if int(valid_channels.sum().item()) > min_keep:
                    eligible.append((mod_name, valid_channels))

            if not eligible:
                continue
            mod_name, valid_channels = eligible[int(torch.randint(len(eligible), (1,), device=target_mask.device).item())]
            valid_idx = torch.where(valid_channels)[0]
            max_allowed = max(0, int(valid_idx.numel()) - min_keep)
            if max_allowed <= 0:
                continue
            n_drop_high = min(max_drop, max_allowed)
            n_drop_low = min(min_drop, n_drop_high)
            n_drop = int(torch.randint(n_drop_low, n_drop_high + 1, (1,), device=target_mask.device).item())
            perm = torch.randperm(valid_idx.numel(), device=target_mask.device)
            drop_channels = valid_idx[perm[:n_drop]]

            x = batch["modalities"][mod_name]
            if len(coord) == 1:
                x[coord[0], drop_channels, :] = 0.0
            else:
                x[coord[0], coord[1], drop_channels, :] = 0.0
            c_mask = batch.get("channel_mask", {}).get(mod_name, None)
            if c_mask is not None:
                if len(coord) == 1:
                    c_mask[coord[0], drop_channels] = False
                else:
                    c_mask[coord[0], coord[1], drop_channels] = False

    def _apply_view_dropout(self, batch: dict[str, Any], training: bool) -> None:
        if not training:
            return
        if not bool(self.view_dropout_cfg.get("enabled", False)):
            return
        first_mask = next(iter(batch["modality_mask"].values()))
        modality_mask, channel_mask = self._sample_view_dropout_masks(first_mask.shape, first_mask.device)
        self._apply_modality_dropout_positions(batch=batch, target_mask=modality_mask)
        self._apply_channel_dropout_positions(batch=batch, target_mask=channel_mask)

    def _sigreg(self, proj: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
        x = proj.float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"SIGReg expects [V,B,K] or [B,K], got {tuple(x.shape)}")

        num_slices = int(cfg.get("num_slices", 256))
        t_max = float(cfg.get("t_max", 3.0))
        n_points = int(cfg.get("n_points", 17))
        num_slices = max(1, num_slices)
        n_points = max(3, n_points)

        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(self.global_step))
        A = torch.randn((x.size(-1), num_slices), device=x.device, generator=generator, dtype=x.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)

        t = torch.linspace(0.0, t_max, n_points, device=x.device, dtype=x.dtype)
        dt = t_max / float(max(1, n_points - 1))
        weights = torch.full((n_points,), 2.0 * dt, device=x.device, dtype=x.dtype)
        weights[0] = dt
        weights[-1] = dt
        phi = torch.exp(-0.5 * t.square())
        weights = weights * phi

        x_t = (x @ A).unsqueeze(-1) * t
        cos_m = _ddp_all_reduce(x_t.cos().mean(dim=-3))
        sin_m = _ddp_all_reduce(x_t.sin().mean(dim=-3))
        err = (cos_m - phi).square() + sin_m.square()
        statistic = (err @ weights) * x.size(-2)
        return statistic.mean()

    def _lejepa_loss_from_embeddings(
        self,
        global_embeddings: list[torch.Tensor],
        all_embeddings: list[torch.Tensor],
        lamb: float,
        sigreg_cfg: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(global_embeddings) == 0 or len(all_embeddings) == 0:
            raise ValueError("LeJEPA requires non-empty global and all-view embeddings.")

        g = torch.stack(global_embeddings, dim=0)
        a = torch.stack(all_embeddings, dim=0)
        center = g.mean(dim=0)
        pred_loss = (center.unsqueeze(0) - a).square().mean()
        sigreg_loss = self._sigreg(a, cfg=sigreg_cfg)
        total = (1.0 - lamb) * pred_loss + lamb * sigreg_loss
        return total, pred_loss, sigreg_loss

    def _compute_stage1_eeg_jepa_loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        core_model = _unwrap_model(self.model)
        eeg = batch["modalities"].get("eeg")
        if eeg is None:
            raise ValueError("Stage1 EEG JEPA requires EEG modality in batch.")
        channel_mask = batch.get("channel_mask", {}).get("eeg", None)

        global_views = [self._augment_eeg_weak(eeg, channel_mask=channel_mask) for _ in range(self.stage1_num_global_views)]
        local_views = self._build_stage1_local_views(eeg=eeg, channel_mask=channel_mask)
        all_views = global_views + local_views

        global_embs = [core_model.encoders["eeg"](xv) for xv in global_views]
        all_embs = [core_model.encoders["eeg"](xv) for xv in all_views]
        total, pred_loss, sigreg_loss = self._lejepa_loss_from_embeddings(
            global_embeddings=global_embs,
            all_embeddings=all_embs,
            lamb=self.stage1_lambda,
            sigreg_cfg=self.stage1_sigreg_cfg,
        )
        aux = {
            "stage1_pred_loss": pred_loss.detach(),
            "stage1_sigreg_loss": sigreg_loss.detach(),
            "stage1_total_loss": total.detach(),
        }
        return total, None, aux

    def _extract_fused_feature(
        self,
        modalities: dict[str, torch.Tensor],
        modality_mask: dict[str, torch.Tensor],
        channel_mask: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        _, feat = self.model(
            modalities=modalities,
            modality_mask=modality_mask,
            channel_mask=channel_mask,
            return_features=True,
        )
        return feat

    def _compute_stage2_multimodal_pretrain_loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        global_embs = []
        for view_index in range(self.stage2_num_global_views):
            modalities_view, mask_view, channel_view = self._build_stage2_global_view(
                batch=batch,
                training=self.model.training,
                view_index=view_index,
            )
            global_embs.append(
                self._extract_fused_feature(
                    modalities=modalities_view,
                    modality_mask=mask_view,
                    channel_mask=channel_view,
                )
            )

        all_embs = list(global_embs)
        keep_sets = self._build_stage2_keep_sets()
        if (not self._stage2_keep_sets_logged) and is_main_process():
            pretty = [sorted(list(x)) for x in keep_sets]
            self.logger.info("stage2 local keep sets (effective): %s", pretty)
            self._stage2_keep_sets_logged = True

        for keep_set in keep_sets:
            modalities_view, mask_view, channel_view = self._build_stage2_view(
                batch=batch,
                keep_modalities=keep_set,
            )
            all_embs.append(
                self._extract_fused_feature(
                    modalities=modalities_view,
                    modality_mask=mask_view,
                    channel_mask=channel_view,
                )
            )

        total, pred_loss, sigreg_loss = self._lejepa_loss_from_embeddings(
            global_embeddings=global_embs,
            all_embeddings=all_embs,
            lamb=self.stage2_lambda,
            sigreg_cfg=self.stage2_sigreg_cfg,
        )
        aux = {
            "stage2_pred_loss": pred_loss.detach(),
            "stage2_sigreg_loss": sigreg_loss.detach(),
            "stage2_total_loss": total.detach(),
        }
        return total, None, aux

    def _compute_supervised_ce_loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        try:
            logits = self.model(
                modalities=batch["modalities"],
                modality_mask=batch["modality_mask"],
                channel_mask=batch.get("channel_mask"),
                seq_padding_mask=batch.get("seq_padding_mask"),
                task_name=self.downstream_task,
            )
        except KeyError as exc:
            core_model = _unwrap_model(self.model)
            supported = []
            if hasattr(core_model, "get_supported_tasks"):
                try:
                    supported = list(core_model.get_supported_tasks())
                except Exception:
                    supported = []
            raise NotImplementedError(
                f"Unsupported downstream task '{self.downstream_task}'. "
                f"Supported tasks: {supported}"
            ) from exc
        if logits.ndim == 3 and batch["labels"].ndim == 2:
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]),
                batch["labels"].reshape(-1),
            )
        else:
            loss = self.criterion(logits, batch["labels"])
        aux = {"ce_loss": loss.detach()}
        return loss, logits, aux

    def _compute_objective(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        if _is_stage1_eeg_jepa(self.stage):
            return self._compute_stage1_eeg_jepa_loss(batch=batch)
        if _is_stage2_multimodal_pretrain(self.stage):
            return self._compute_stage2_multimodal_pretrain_loss(batch=batch)
        return self._compute_supervised_ce_loss(batch=batch)

    def _sync_loss_and_count(self, loss_sum: torch.Tensor, sample_count: torch.Tensor) -> tuple[float, int]:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        avg_loss = (loss_sum / sample_count.clamp_min(1.0)).item()
        return avg_loss, int(sample_count.item())

    def _sync_confusion(self, confusion: torch.Tensor | None) -> torch.Tensor | None:
        if confusion is None:
            return None
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
        return confusion

    def _log_epoch_metrics(self, phase: str, epoch: int, metrics: dict[str, Any]) -> None:
        if not is_main_process():
            return
        self.logger.info(
            "%s | stage=%s | epoch=%d | loss=%.6f | acc=%.4f | macro_f1=%.4f",
            phase,
            self.stage,
            epoch,
            metrics["loss"],
            metrics["accuracy"],
            metrics["macro_f1"],
        )

    def _run_epoch(self, loader: Any, epoch: int, training: bool) -> dict[str, Any]:
        if loader is None:
            return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

        if training:
            self.model.train()
            if self._use_multi_view_pretrain_objective():
                self._set_batch_norm_eval()
        else:
            self.model.eval()

        grad_accum_steps = int(self.training_cfg.get("grad_accum_steps", 1))
        log_interval = int(self.training_cfg.get("log_interval", 20))
        max_grad_norm = float(self.training_cfg.get("max_grad_norm", 1.0))
        use_amp = bool(self.training_cfg.get("use_amp", True)) and self.device.type == "cuda"

        loss_sum = torch.zeros(1, device=self.device, dtype=torch.float64)
        sample_count = torch.zeros(1, device=self.device, dtype=torch.float64)
        confusion = (
            torch.zeros((self.num_classes, self.num_classes), device=self.device, dtype=torch.float64)
            if self.num_classes is not None
            else None
        )
        aux_sums: dict[str, torch.Tensor] = {}

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        start_time = time.time()
        for step, raw_batch in enumerate(loader):
            batch = self._move_batch(raw_batch)
            self._apply_view_dropout(batch, training=training)

            with torch.autocast(device_type=self.device.type, enabled=use_amp, dtype=self.amp_dtype):
                loss, logits_for_metrics, aux = self._compute_objective(batch=batch)

            if training:
                scaled_loss = loss / max(1, grad_accum_steps)
                if self.scaler is not None and use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                need_step = ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(loader))
                if need_step:
                    if max_grad_norm > 0:
                        if self.scaler is not None and use_amp:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer_step_ran = False
                    if self.scaler is not None and use_amp:
                        prev_scale = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        # GradScaler may skip optimizer.step() on inf/NaN gradients.
                        # Only advance scheduler/global_step when a parameter update happened.
                        optimizer_step_ran = self.scaler.get_scale() >= prev_scale
                    else:
                        self.optimizer.step()
                        optimizer_step_ran = True
                    if optimizer_step_ran:
                        if self.scheduler is not None:
                            self.scheduler.step()
                        self.global_step += 1
                    self.optimizer.zero_grad(set_to_none=True)

            if logits_for_metrics is not None:
                if self.num_classes is None:
                    self.num_classes = int(logits_for_metrics.shape[-1])
                if confusion is None:
                    confusion = torch.zeros((self.num_classes, self.num_classes), device=self.device, dtype=torch.float64)
                if logits_for_metrics.ndim == 3 and batch["labels"].ndim == 2:
                    preds = torch.argmax(logits_for_metrics, dim=-1)
                    targets = batch["labels"]
                    if "seq_padding_mask" in batch:
                        valid = ~batch["seq_padding_mask"]
                    else:
                        valid = targets != self.sequence_label_pad_value
                    preds = preds[valid]
                    targets = targets[valid]
                else:
                    preds = torch.argmax(logits_for_metrics, dim=1)
                    targets = batch["labels"]
                confusion = update_confusion(confusion, preds=preds, targets=targets, num_classes=self.num_classes)

            if batch["labels"].ndim == 2:
                if "seq_padding_mask" in batch:
                    valid_tokens = (~batch["seq_padding_mask"]) & (batch["labels"] != self.sequence_label_pad_value)
                    bs = int(valid_tokens.sum().item())
                else:
                    bs = int((batch["labels"] != self.sequence_label_pad_value).sum().item())
            else:
                bs = batch["labels"].shape[0]
            loss_sum += loss.detach().to(dtype=torch.float64) * bs
            sample_count += bs
            for key, value in aux.items():
                val = value.detach()
                if val.ndim > 0:
                    val = val.mean()
                if key not in aux_sums:
                    aux_sums[key] = torch.zeros(1, device=self.device, dtype=torch.float64)
                aux_sums[key] += val.to(dtype=torch.float64) * bs

            if training and is_main_process() and ((step + 1) % log_interval == 0):
                elapsed = max(1e-6, time.time() - start_time)
                seen = int(sample_count.item())
                aux_msg = " ".join([f"{k}={float(v):.4f}" for k, v in aux.items()])
                self.logger.info(
                    "train step=%d/%d | epoch=%d | loss=%.6f | %s | samples=%d | speed=%.2f sample/s",
                    step + 1,
                    len(loader),
                    epoch,
                    float(loss.item()),
                    aux_msg,
                    seen,
                    seen / elapsed,
                )

        avg_loss, _ = self._sync_loss_and_count(loss_sum=loss_sum, sample_count=sample_count)
        confusion = self._sync_confusion(confusion)
        metrics = (
            metrics_from_confusion(confusion)
            if confusion is not None
            else {"accuracy": 0.0, "macro_f1": 0.0}
        )
        metrics["loss"] = avg_loss
        metrics["global_step"] = self.global_step

        denom = sample_count.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(denom, op=dist.ReduceOp.SUM)
        for key, value_sum in aux_sums.items():
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(value_sum, op=dist.ReduceOp.SUM)
            metrics[key] = (value_sum / denom.clamp_min(1.0)).item()
        return metrics

    def train_one_epoch(self, train_loader: Any, epoch: int) -> dict[str, Any]:
        metrics = self._run_epoch(loader=train_loader, epoch=epoch, training=True)
        self._log_epoch_metrics("train", epoch, metrics)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: Any, epoch: int, split_name: str = "val") -> dict[str, Any]:
        metrics = self._run_epoch(loader=loader, epoch=epoch, training=False)
        self._log_epoch_metrics(split_name, epoch, metrics)
        return metrics
