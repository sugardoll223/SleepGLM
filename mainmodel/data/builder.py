from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .collate import SleepCollator
from .h5_dataset import SleepH5Dataset


def _resolve_path_from_runtime(
    raw_path: str,
    config_dir: Path | None,
    must_exist: bool,
) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        resolved = p.resolve()
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")
        return resolved

    candidates: list[Path] = [(Path.cwd() / p).resolve()]
    if config_dir is not None:
        candidates.append((config_dir / p).resolve())
        candidates.append((config_dir.parent / p).resolve())

    for cand in candidates:
        if cand.exists():
            return cand

    fallback = candidates[-1] if config_dir is not None else candidates[0]
    if must_exist:
        tried = ", ".join(str(x) for x in candidates)
        raise FileNotFoundError(f"Path does not exist: {raw_path}. Tried: {tried}")
    return fallback


def _resolve_split_manifest_files(cfg: dict[str, Any]) -> tuple[list[str], list[str], list[str]] | None:
    data_cfg = cfg["data"]
    split_file = str(data_cfg.get("split_file", "")).strip()
    if not split_file:
        return None

    runtime_cfg = cfg.get("_runtime", {})
    config_dir_raw = str(runtime_cfg.get("config_dir", "")).strip()
    config_dir = Path(config_dir_raw).expanduser().resolve() if config_dir_raw else None

    split_path = _resolve_path_from_runtime(split_file, config_dir=config_dir, must_exist=True)
    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    files_section = payload.get("files", {})
    if not isinstance(files_section, dict):
        raise ValueError(f"Invalid split manifest format in {split_path}: missing 'files' dict")

    split_root_dir = str(data_cfg.get("split_root_dir", "")).strip()
    split_root = (
        _resolve_path_from_runtime(split_root_dir, config_dir=config_dir, must_exist=False)
        if split_root_dir
        else split_path.parent
    )

    def _to_abs_list(items: Any) -> list[str]:
        if not isinstance(items, list):
            return []
        out: list[str] = []
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            p = Path(text)
            if not p.is_absolute():
                p = split_root / p
            out.append(str(p.resolve()))
        return out

    train_files = _to_abs_list(files_section.get("train", []))
    val_files = _to_abs_list(files_section.get("val", []))
    test_files = _to_abs_list(files_section.get("test", []))
    return train_files, val_files, test_files


def _build_one_loader(
    dataset: SleepH5Dataset | None,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    shuffle: bool,
    drop_last: bool,
    data_cfg: dict[str, Any],
    collator: SleepCollator,
) -> DataLoader | None:
    if dataset is None or len(dataset) == 0:
        return None

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    persistent_workers = bool(data_cfg.get("persistent_workers", True)) and num_workers > 0
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "drop_last": drop_last,
        "persistent_workers": persistent_workers,
        "collate_fn": collator,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return DataLoader(**loader_kwargs)


def build_dataloaders(cfg: dict[str, Any], distributed: bool) -> dict[str, Any]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    collator = SleepCollator(data_cfg=data_cfg, model_cfg=model_cfg)

    train_files = list(data_cfg.get("train_files", []))
    val_files = list(data_cfg.get("val_files", []))
    test_files = list(data_cfg.get("test_files", []))
    split_manifest_files = _resolve_split_manifest_files(cfg)
    use_manifest_split = split_manifest_files is not None
    if split_manifest_files is not None:
        train_files, val_files, test_files = split_manifest_files

    train_set = SleepH5Dataset(
        files=train_files,
        split_name=(None if use_manifest_split else data_cfg.get("train_split_name", "train")),
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    ) if train_files else None
    val_set = SleepH5Dataset(
        files=val_files,
        split_name=(None if use_manifest_split else data_cfg.get("val_split_name", "val")),
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    ) if val_files else None
    test_set = SleepH5Dataset(
        files=test_files,
        split_name=(None if use_manifest_split else data_cfg.get("test_split_name", "test")),
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    ) if test_files else None

    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader = _build_one_loader(
        dataset=train_set,
        batch_size=int(data_cfg.get("train_batch_size", 32)),
        num_workers=num_workers,
        distributed=distributed,
        shuffle=True,
        drop_last=bool(data_cfg.get("drop_last", True)),
        data_cfg=data_cfg,
        collator=collator,
    )
    val_loader = _build_one_loader(
        dataset=val_set,
        batch_size=int(data_cfg.get("val_batch_size", 32)),
        num_workers=num_workers,
        distributed=distributed,
        shuffle=False,
        drop_last=False,
        data_cfg=data_cfg,
        collator=collator,
    )
    test_loader = _build_one_loader(
        dataset=test_set,
        batch_size=int(data_cfg.get("test_batch_size", data_cfg.get("val_batch_size", 32))),
        num_workers=num_workers,
        distributed=distributed,
        shuffle=False,
        drop_last=False,
        data_cfg=data_cfg,
        collator=collator,
    )

    return {
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
