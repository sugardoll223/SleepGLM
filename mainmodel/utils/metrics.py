from __future__ import annotations

from typing import Any

import torch


def update_confusion(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if confusion is None:
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=preds.device)

    preds = preds.view(-1).long()
    targets = targets.view(-1).long()
    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]
    bincount = torch.bincount(
        targets * num_classes + preds,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    confusion = confusion + bincount.to(confusion.dtype)
    return confusion


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    accuracy = (correct / total) if total > 0 else 0.0

    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    macro_f1 = f1.mean().item()

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": [x.item() for x in f1],
    }

