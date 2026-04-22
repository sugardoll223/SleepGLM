from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mainmodel.data.builder import build_dataloaders
from mainmodel.engine.trainer import DDPTrainer, build_optimizer, build_scheduler
from mainmodel.models import Model
from mainmodel.utils import (
    cleanup_distributed,
    dump_config,
    find_latest_checkpoint,
    init_distributed,
    is_main_process,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logger,
)


def _is_finetune_stage(stage: str) -> bool:
    normalized = str(stage).strip().lower()
    return normalized in {"finetune", "stage3_finetune", "stage3_downstream_finetune"}


def _is_stage2_pretrain_stage(stage: str) -> bool:
    normalized = str(stage).strip().lower()
    return normalized in {"pretrain", "stage2_multimodal_pretrain", "stage2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model training entrypoint (DDP-ready).")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Override config with dotted key format, e.g. training.epochs=30",
    )
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation.")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path for eval-only.")
    return parser.parse_args()


def _prepare_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _maybe_sync_bn(cfg: dict[str, Any], model: torch.nn.Module, distributed: bool) -> torch.nn.Module:
    if distributed and bool(cfg.get("experiment", {}).get("use_sync_bn", False)):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def _save_runtime_config(cfg: dict[str, Any], output_dir: str) -> None:
    dump_config(cfg, os.path.join(output_dir, "resolved_config.yaml"))


def _resolve_eval_checkpoint(args_checkpoint: str, output_dir: str) -> str:
    ckpt = args_checkpoint.strip()
    if ckpt:
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {ckpt}")
        return ckpt

    best_path = os.path.join(output_dir, "best.pt")
    if os.path.exists(best_path):
        return best_path
    last_path = os.path.join(output_dir, "last.pt")
    if os.path.exists(last_path):
        return last_path
    raise ValueError(
        "Eval-only requires a checkpoint. Please pass --checkpoint, "
        "or ensure outputs contain best.pt/last.pt."
    )


def _wrap_ddp(
    model: torch.nn.Module,
    distributed: bool,
    local_rank: int,
    find_unused_parameters: bool,
) -> torch.nn.Module:
    if not distributed:
        return model
    if torch.cuda.is_available():
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return DDP(model, find_unused_parameters=find_unused_parameters)


def _load_pretrained_if_needed(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    logger: Any,
) -> None:
    stage = cfg.get("experiment", {}).get("stage", "stage2_multimodal_pretrain")
    if not _is_finetune_stage(stage):
        return
    ckpt_path = str(cfg.get("finetune", {}).get("pretrained_checkpoint", "")).strip()
    if not ckpt_path:
        return
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"finetune.pretrained_checkpoint not found: {ckpt_path}")
    meta = load_checkpoint(ckpt_path, model=model, strict=False, map_location="cpu")
    logger.info(
        "Loaded pretrained checkpoint: %s | missing=%d | unexpected=%d",
        ckpt_path,
        len(meta["missing_keys"]),
        len(meta["unexpected_keys"]),
    )


def _load_stage2_eeg_init_if_needed(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    logger: Any,
) -> None:
    stage = cfg.get("experiment", {}).get("stage", "stage2_multimodal_pretrain")
    if not _is_stage2_pretrain_stage(stage):
        return

    stage2_cfg = cfg.get("training", {}).get("stage2_multimodal", {})
    init_enabled = bool(stage2_cfg.get("init_eeg_encoder_from_stage1", False))
    if not init_enabled:
        logger.info("Stage2 EEG encoder initialization from stage1 is disabled.")
        return

    ckpt_path = str(stage2_cfg.get("stage1_checkpoint", "")).strip()
    if not ckpt_path:
        raise ValueError(
            "training.stage2_multimodal.init_eeg_encoder_from_stage1=true "
            "requires training.stage2_multimodal.stage1_checkpoint."
        )
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"stage1 checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    source_state = checkpoint.get("model", checkpoint)

    target_model = model.module if hasattr(model, "module") else model
    target_state = target_model.state_dict()
    loaded = 0
    skipped = 0
    for key, value in source_state.items():
        if not key.startswith("encoders.eeg."):
            continue
        if key not in target_state:
            skipped += 1
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            skipped += 1
            continue
        target_state[key] = value
        loaded += 1

    target_model.load_state_dict(target_state, strict=False)
    logger.info(
        "Stage2 EEG encoder initialized from %s | loaded=%d | skipped=%d",
        ckpt_path,
        loaded,
        skipped,
    )


def _maybe_resume(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    output_dir: str,
    logger: Any,
) -> tuple[int, float, int]:
    resume_ckpt = str(cfg.get("finetune", {}).get("resume_checkpoint", "")).strip()
    if not resume_ckpt:
        auto_ckpt = find_latest_checkpoint(output_dir)
        resume_ckpt = auto_ckpt or ""

    if not resume_ckpt:
        return 0, float("-inf"), 0

    if not Path(resume_ckpt).exists():
        logger.warning("Resume checkpoint not found: %s", resume_ckpt)
        return 0, float("-inf"), 0

    meta = load_checkpoint(
        resume_ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        strict=True,
        map_location="cpu",
    )
    start_epoch = int(meta.get("epoch", -1)) + 1
    best_metric = float(meta.get("best_metric", float("-inf")))
    global_step = int(meta.get("global_step", 0))
    logger.info(
        "Resumed from %s | start_epoch=%d | best_metric=%.6f | global_step=%d",
        resume_ckpt,
        start_epoch,
        best_metric,
        global_step,
    )
    return start_epoch, best_metric, global_step


def run_training(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    dist_ctx = init_distributed()
    distributed = dist_ctx["distributed"]
    rank = dist_ctx["rank"]
    local_rank = dist_ctx["local_rank"]
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    deterministic = bool(cfg.get("training", {}).get("deterministic", False))
    set_seed(seed + rank, deterministic=deterministic)

    device = _prepare_device(local_rank)
    output_dir = str(cfg.get("experiment", {}).get("output_dir", "outputs/default"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir=output_dir, rank=rank, name="train")
    stage = cfg.get("experiment", {}).get("stage", "stage2_multimodal_pretrain")
    if is_main_process():
        _save_runtime_config(cfg, output_dir)
        logger.info("Resolved config saved to %s", os.path.join(output_dir, "resolved_config.yaml"))

    loaders = build_dataloaders(cfg, distributed=distributed)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    model = Model(cfg["model"])
    model = _maybe_sync_bn(cfg, model, distributed=distributed)
    model = model.to(device)

    _load_stage2_eeg_init_if_needed(cfg, model, logger)
    _load_pretrained_if_needed(cfg, model, logger)

    find_unused = bool(cfg.get("training", {}).get("find_unused_parameters", False))
    model = _wrap_ddp(model, distributed=distributed, local_rank=local_rank, find_unused_parameters=find_unused)

    if bool(cfg.get("training", {}).get("use_torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = build_optimizer(cfg, model)
    scheduler, _ = build_scheduler(cfg, optimizer, train_loader_len=len(train_loader) if train_loader is not None else 0)
    use_amp = bool(cfg.get("training", {}).get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    trainer = DDPTrainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        logger=logger,
        device=device,
        output_dir=output_dir,
    )

    start_epoch, best_metric, global_step = _maybe_resume(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        output_dir=output_dir,
        logger=logger,
    )
    trainer.global_step = global_step

    if args.eval_only:
        ckpt = _resolve_eval_checkpoint(args.checkpoint, output_dir=output_dir)
        load_checkpoint(ckpt, model=model, strict=True, map_location="cpu")
        if is_main_process():
            logger.info("Loaded checkpoint for eval-only: %s", ckpt)
        target_loader = test_loader if test_loader is not None else val_loader
        if target_loader is None:
            raise ValueError("Eval-only requires data.val_files or data.test_files (or split_file).")
        metrics = trainer.evaluate(target_loader, epoch=start_epoch, split_name="eval")
        if is_main_process():
            logger.info("Eval metrics: %s", metrics)
        cleanup_distributed()
        return

    if train_loader is None:
        raise ValueError("No training data found. Please set data.train_files in config.")

    epochs = int(cfg["training"].get("epochs", 1))
    freeze_epochs = int(cfg.get("finetune", {}).get("freeze_backbone_epochs", 0))
    val_interval = int(cfg["training"].get("val_interval", 1))
    save_interval = int(cfg["training"].get("save_interval", 1))
    for epoch in range(start_epoch, epochs):
        if distributed and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if _is_finetune_stage(stage):
            core = model.module if hasattr(model, "module") else model
            core.freeze_backbone(epoch < freeze_epochs)
            if is_main_process():
                logger.info("Epoch %d backbone frozen: %s", epoch, str(epoch < freeze_epochs))

        train_metrics = trainer.train_one_epoch(train_loader, epoch=epoch)

        val_metrics = None
        if val_loader is not None and ((epoch + 1) % max(1, val_interval) == 0):
            val_metrics = trainer.evaluate(val_loader, epoch=epoch, split_name="val")

        if not is_main_process():
            continue

        state = {
            "epoch": epoch,
            "global_step": trainer.global_step,
            "best_metric": best_metric,
            "model": (model.module if hasattr(model, "module") else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": cfg,
        }

        if (epoch + 1) % max(1, save_interval) == 0:
            save_checkpoint(state, output_dir=output_dir, filename=f"last_epoch_{epoch:04d}.pt")
            save_checkpoint(state, output_dir=output_dir, filename="last.pt")

        if val_metrics is not None:
            metric = float(val_metrics.get("macro_f1", float("-inf")))
            if metric > best_metric:
                best_metric = metric
                state["best_metric"] = best_metric
                save_checkpoint(state, output_dir=output_dir, filename="best.pt")
                logger.info("Best checkpoint updated at epoch %d with macro_f1=%.6f", epoch, best_metric)

    if test_loader is not None:
        if is_main_process():
            logger.info("Final test evaluation starts.")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        best_path = os.path.join(output_dir, "best.pt")
        if os.path.exists(best_path):
            load_checkpoint(best_path, model=model, strict=True, map_location="cpu")
        test_metrics = trainer.evaluate(test_loader, epoch=epochs, split_name="test")
        if is_main_process():
            logger.info("Test metrics: %s", test_metrics)

    cleanup_distributed()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    run_training(cfg, args)


if __name__ == "__main__":
    main()

