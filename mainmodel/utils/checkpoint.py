from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(state: dict[str, Any], output_dir: str, filename: str) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    torch.save(state, path)
    return str(path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model_state = checkpoint.get("model", checkpoint)
    missing, unexpected = _unwrap_model(model).load_state_dict(model_state, strict=strict)
    metadata = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "epoch": checkpoint.get("epoch", -1),
        "best_metric": checkpoint.get("best_metric", float("-inf")),
        "global_step": checkpoint.get("global_step", 0),
    }

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return metadata


def find_latest_checkpoint(output_dir: str) -> str | None:
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return None
    candidates = sorted(out_dir.glob("last_epoch_*.pt"))
    if not candidates:
        fallback = out_dir / "last.pt"
        return str(fallback) if fallback.exists() else None
    return str(candidates[-1])

