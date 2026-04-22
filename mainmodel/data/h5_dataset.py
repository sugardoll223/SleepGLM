from __future__ import annotations

import argparse
import hashlib
import re
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _decode_scalar(value.item())
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_int(value: Any, default: int = 0) -> int:
    value = _decode_scalar(value)
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default
    return default


def _to_str(value: Any) -> str:
    value = _decode_scalar(value)
    return "" if value is None else str(value)


def _to_str_list(value: Any) -> list[str]:
    value = _decode_scalar(value)
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        out: list[str] = []
        for item in value.reshape(-1):
            out.extend(_to_str_list(item))
        return out
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_to_str_list(item))
        return out
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        if "," in value:
            parts = [x.strip() for x in value.split(",")]
            return [x for x in parts if x]
        raw = value.strip()
        return [raw] if raw else []
    return [str(value)]


def _normalize_dataset_name(name: str) -> str:
    if not name:
        return ""
    normalized = re.sub(r"[\s_\-]+", "", name).upper()

    alias_map = {
        "HSP": ["HSP"],
        "SHHS": ["SHHS", "SHHS1", "SHHS2"],
        "MROS": ["MROS"],
        "MESA": ["MESA"],
        "PHYSIONET2018": ["PHYSIONET2018", "CINC2018"],
        "CAP": ["CAP"],
        "SLEEPEDF": ["SLEEPEDF", "SLEEPEDFSC", "SLEEPEDFST"],
        "ISRUC": ["ISRUC", "ISRUCSLEEP"],
        "WSC": ["WSC", "WISCONSIN", "WISCONSINSLEEPCOHORT"],
    }
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias in normalized:
                return canonical
    return normalized


def _norm_token(text: str) -> str:
    return re.sub(r"[\s_\-:/]+", "", text).upper()


class SleepH5Dataset(Dataset):
    """
    Generic sleep H5 dataset loader.

    Supported layouts:
    1) Array layout:
       /eeg [N, C, T], /label [N], ...
    2) Sample-group layout:
       /samples/<sample_id>/eeg [C, T], /samples/<sample_id>/label, ...
    3) Continuous PSG layout:
       /signals/<modality_or_channel_group>/<channel_name> [TotalSamples]
       /hypnogram [N_epochs]
    """

    def __init__(
        self,
        files: list[str],
        split_name: str | None,
        data_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.files = [str(Path(p).expanduser().resolve()) for p in files]
        self.split_name = split_name.lower() if split_name else None
        self.requested_split_name = self.split_name

        self.split_key = data_cfg.get("split_key", "split")
        self.sample_group_key = data_cfg.get("sample_group_key", "samples")
        self.label_keys = list(data_cfg.get("label_key_candidates", ["label", "labels", "y"]))
        self.dataset_id_keys = list(data_cfg.get("dataset_id_key_candidates", ["dataset_id"]))
        self.dataset_name_keys = list(data_cfg.get("dataset_name_key_candidates", ["dataset_name", "dataset"]))
        self.subject_id_keys = list(
            data_cfg.get(
                "subject_id_key_candidates",
                ["subject_id", "subject", "subject_name", "patient_id", "participant_id"],
            )
        )
        self.dataset_vocab = list(data_cfg.get("dataset_vocab", []))
        self.dataset_to_id = {_normalize_dataset_name(name): idx for idx, name in enumerate(self.dataset_vocab)}
        self.file_dataset_name_overrides = data_cfg.get("file_dataset_name_overrides", {})

        self.modality_key_map = data_cfg.get("modality_keys", {})
        self.modality_names = list(model_cfg.get("modalities", {}).keys())
        self.modality_channel_name_keys = data_cfg.get("modality_channel_name_keys", {})
        self.strict_modality = bool(data_cfg.get("strict_modality", False))
        self.return_sequence = bool(data_cfg.get("return_sequence", False))

        self.use_subject_split = bool(data_cfg.get("split_by_subject", False))
        self.subject_split_seed = int(data_cfg.get("subject_split_seed", 42))
        self.subject_split_ratios = self._parse_subject_split_ratios(
            data_cfg.get("subject_split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
        )
        self.subject_split_regex = str(
            data_cfg.get(
                "subject_id_from_filename_regex",
                r"^([A-Za-z]{2}\d{3})\d[A-Za-z]\d$",
            )
        )
        self.subject_split_regex_group = int(data_cfg.get("subject_id_from_filename_regex_group", 1))
        self._subject_regex = re.compile(self.subject_split_regex)
        self._sleepedf_subject_regex = re.compile(r"^([A-Za-z]{2}\d{3})\d[A-Za-z]\d$")
        self._warned_missing_subject = False
        self._split_alias_to_canonical = {
            "train": "train",
            "training": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "dev": "val",
            "test": "test",
            "testing": "test",
            str(data_cfg.get("train_split_name", "train")).strip().lower(): "train",
            str(data_cfg.get("val_split_name", "val")).strip().lower(): "val",
            str(data_cfg.get("test_split_name", "test")).strip().lower(): "test",
        }
        self.canonical_split_name = self._canonicalize_split_name(self.split_name)

        self.epoch_seconds = int(data_cfg.get("epoch_seconds", 30))
        self.cont_signal_group_key = str(data_cfg.get("continuous_signal_group_key", "signals"))
        self.cont_label_key = str(data_cfg.get("continuous_label_key", "hypnogram"))
        self.continuous_channel_aliases = data_cfg.get(
            "continuous_channel_aliases",
            {
                "eeg": ["EEG", "F3", "F4", "C3", "C4", "O1", "O2", "FPZ", "PZ"],
                "eog": ["EOG", "LOC", "ROC", "E1", "E2"],
                "emg": ["EMG", "CHIN", "LEG", "SUBMENTAL"],
                "ecg": ["ECG", "EKG"],
                "airflow": ["AIRFLOW", "NASAL", "FLOW"],
                "thoracoabdominal": ["THORAX", "ABDOMEN", "RESP"],
                "spo2": ["SPO2", "SAO2"],
                "ppg": ["PPG", "PLETH"],
            },
        )
        self.modality_sample_rates = {
            mod_name: int(model_cfg.get("modalities", {}).get(mod_name, {}).get("sample_rate", 1))
            for mod_name in self.modality_names
        }

        self.index: list[dict[str, Any]] = []
        self.file_dataset_name_hints: dict[int, str] = {}
        self.file_subject_hints: dict[int, str] = {}
        self._worker_handles: dict[int, dict[int, h5py.File]] = {}
        self._build_index()

    @staticmethod
    def _parse_subject_split_ratios(cfg_value: Any) -> tuple[float, float, float]:
        if isinstance(cfg_value, dict):
            train_r = float(cfg_value.get("train", 0.8))
            val_r = float(cfg_value.get("val", 0.1))
            test_r = float(cfg_value.get("test", 0.1))
        elif isinstance(cfg_value, (list, tuple)) and len(cfg_value) == 3:
            train_r = float(cfg_value[0])
            val_r = float(cfg_value[1])
            test_r = float(cfg_value[2])
        else:
            train_r, val_r, test_r = 0.8, 0.1, 0.1

        train_r = max(0.0, train_r)
        val_r = max(0.0, val_r)
        test_r = max(0.0, test_r)
        total = train_r + val_r + test_r
        if total <= 0:
            return 0.8, 0.1, 0.1
        return train_r / total, val_r / total, test_r / total

    def _canonicalize_split_name(self, split_name: str | None) -> str | None:
        if split_name is None:
            return None
        return self._split_alias_to_canonical.get(str(split_name).strip().lower(), None)

    def _subject_hash_unit_interval(self, subject_id: str) -> float:
        key = f"{self.subject_split_seed}:{subject_id}".encode("utf-8", errors="ignore")
        digest = hashlib.md5(key).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return value / float(2**64 - 1)

    def _subject_to_split(self, subject_id: str) -> str:
        train_r, val_r, _ = self.subject_split_ratios
        score = self._subject_hash_unit_interval(subject_id)
        if score < train_r:
            return "train"
        if score < (train_r + val_r):
            return "val"
        return "test"

    def _match_subject_split(self, subject_id: str, fallback_split: str | None = None) -> bool:
        if not self.use_subject_split:
            return True
        if self.canonical_split_name is None:
            return True

        if subject_id:
            return self._subject_to_split(subject_id) == self.canonical_split_name

        if fallback_split is not None:
            fallback = self._canonicalize_split_name(fallback_split)
            if fallback is not None:
                return fallback == self.canonical_split_name

        if not self._warned_missing_subject:
            warnings.warn(
                "split_by_subject=True but subject_id is missing for some samples. "
                "These samples are skipped to avoid leakage.",
                RuntimeWarning,
            )
            self._warned_missing_subject = True
        return False

    @staticmethod
    def _clean_subject_id(raw: Any) -> str:
        text = _to_str(raw).strip()
        if not text:
            return ""
        return text.upper()

    @staticmethod
    def _clean_record_token(raw: Any) -> str:
        token = _to_str(raw).strip()
        if not token:
            return ""
        token = Path(token).stem.upper()
        token = token.replace("-PSG", "").replace("-HYPNOGRAM", "")
        return token

    def _infer_subject_from_text(self, raw: Any) -> str:
        token = self._clean_record_token(raw)
        if not token:
            return ""

        match = self._subject_regex.match(token)
        if match is not None:
            try:
                return self._clean_subject_id(match.group(self.subject_split_regex_group))
            except IndexError:
                pass

        sleepedf_match = self._sleepedf_subject_regex.match(token)
        if sleepedf_match is not None:
            return self._clean_subject_id(sleepedf_match.group(1))

        return ""

    def _read_root_scalar(self, h5: h5py.File, keys: list[str], default: Any = "") -> Any:
        for key in keys:
            if key in h5:
                ds = h5[key]
                if isinstance(ds, h5py.Dataset):
                    return _decode_scalar(ds[()])
                return _decode_scalar(ds)
            if key in h5.attrs:
                return _decode_scalar(h5.attrs[key])
        return default

    def _resolve_file_subject_hint(self, h5: h5py.File, file_path: str) -> str:
        raw = self._read_root_scalar(h5, self.subject_id_keys, default="")
        subject_id = self._clean_subject_id(raw)
        if subject_id:
            return subject_id

        for key in ["record_name", "record_id", "source_psg", "source_hypnogram"]:
            raw_token = self._read_root_scalar(h5, [key], default="")
            subject_id = self._infer_subject_from_text(raw_token)
            if subject_id:
                return subject_id

        return self._infer_subject_from_text(Path(file_path).stem)

    def _resolve_subject_id(self, h5: h5py.File, item: dict[str, Any]) -> str:
        file_hint = self.file_subject_hints.get(item["file_idx"], "")

        if item["layout"] in ("array", "continuous"):
            sample_idx = int(item["sample_idx"])
            raw = self._read_scalar_from_array(h5, self.subject_id_keys, sample_idx, default="")
            subject_id = self._clean_subject_id(raw)
            if subject_id:
                return subject_id

            for key in self.subject_id_keys:
                if key in h5 and isinstance(h5[key], h5py.Dataset) and h5[key].ndim == 0:
                    root_subject = self._clean_subject_id(h5[key][()])
                    if root_subject:
                        return root_subject
                if key in h5.attrs:
                    root_subject = self._clean_subject_id(h5.attrs[key])
                    if root_subject:
                        return root_subject

            for key in ["record_name", "record_id", "source_psg", "source_hypnogram"]:
                record_value = self._read_scalar_from_array(h5, [key], sample_idx, default="")
                subject_id = self._infer_subject_from_text(record_value)
                if subject_id:
                    return subject_id
            return file_hint

        group = h5[item["group_path"]]
        raw = self._read_scalar_from_group(group, self.subject_id_keys, default="")
        subject_id = self._clean_subject_id(raw)
        if subject_id:
            return subject_id
        for key in ["record_name", "record_id", "source_psg", "source_hypnogram"]:
            record_value = self._read_scalar_from_group(group, [key], default="")
            subject_id = self._infer_subject_from_text(record_value)
            if subject_id:
                return subject_id
        return file_hint

    def _build_index(self) -> None:
        for file_idx, file_path in enumerate(self.files):
            with h5py.File(file_path, "r") as h5:
                self.file_dataset_name_hints[file_idx] = self._resolve_file_dataset_name(h5, file_path)
                self.file_subject_hints[file_idx] = self._resolve_file_subject_hint(h5, file_path)
                layout, label_key = self._detect_layout(h5)
                if layout == "array":
                    if self.return_sequence:
                        self._index_array_layout_sequence(h5, file_idx, label_key)
                    else:
                        self._index_array_layout(h5, file_idx, label_key)
                elif layout == "sample_group":
                    if self.return_sequence:
                        raise NotImplementedError(
                            "return_sequence=True is not supported for sample-group layout yet. "
                            "Use array/continuous layout or set data.return_sequence=false."
                        )
                    self._index_group_layout(h5, file_idx)
                elif layout == "continuous":
                    if self.return_sequence:
                        self._index_continuous_layout_sequence(h5, file_idx)
                    else:
                        self._index_continuous_layout(h5, file_idx)
                else:
                    raise ValueError(
                        f"Unsupported H5 layout in file: {file_path}. "
                        "Need array layout, sample-group layout, or continuous layout."
                    )

    def _find_key_ci(self, container: h5py.File | h5py.Group, candidates: list[str]) -> str | None:
        existing = {k.upper(): k for k in container.keys()}
        for cand in candidates:
            key = existing.get(cand.upper())
            if key is not None:
                return key
        return None

    def _detect_layout(self, h5: h5py.File) -> tuple[str, str]:
        label_key = self._find_existing_key(h5, self.label_keys)
        if label_key is not None:
            return "array", label_key

        if self.sample_group_key in h5 and isinstance(h5[self.sample_group_key], h5py.Group):
            return "sample_group", ""

        if self.cont_label_key in h5 and self.cont_signal_group_key in h5 and isinstance(h5[self.cont_signal_group_key], h5py.Group):
            return "continuous", self.cont_label_key

        root_groups = [k for k, v in h5.items() if isinstance(v, h5py.Group)]
        for group_name in root_groups:
            group = h5[group_name]
            if self._find_existing_key(group, self.label_keys) is not None:
                return "sample_group", ""
        return "unknown", ""

    def _find_existing_key(self, group: h5py.Group | h5py.File, candidates: list[str]) -> str | None:
        for key in candidates:
            if key in group:
                return key
        return None

    def _index_array_layout(self, h5: h5py.File, file_idx: int, label_key: str) -> None:
        labels = h5[label_key]
        n_samples = int(labels.shape[0])
        has_split = self.split_key in h5 and getattr(h5[self.split_key], "shape", (0,))[0] == n_samples

        for sample_idx in range(n_samples):
            split_value = ""
            if self.split_name and has_split:
                split_value = _to_str(h5[self.split_key][sample_idx]).lower()

            subject_id = self._resolve_subject_id(
                h5,
                {
                    "layout": "array",
                    "file_idx": file_idx,
                    "sample_idx": sample_idx,
                },
            )
            if self.use_subject_split:
                if not self._match_subject_split(subject_id, fallback_split=split_value):
                    continue
            elif self.split_name and has_split and split_value != self.split_name:
                continue
            self.index.append(
                {
                    "layout": "array",
                    "file_idx": file_idx,
                    "sample_idx": sample_idx,
                    "label_key": label_key,
                    "subject_id": subject_id,
                }
            )

    def _index_array_layout_sequence(self, h5: h5py.File, file_idx: int, label_key: str) -> None:
        labels = h5[label_key]
        n_samples = int(labels.shape[0])
        has_split = self.split_key in h5 and getattr(h5[self.split_key], "shape", (0,))[0] == n_samples

        sample_indices: list[int] = []
        subject_id_hint = ""
        for sample_idx in range(n_samples):
            split_value = ""
            if self.split_name and has_split:
                split_value = _to_str(h5[self.split_key][sample_idx]).lower()

            subject_id = self._resolve_subject_id(
                h5,
                {
                    "layout": "array",
                    "file_idx": file_idx,
                    "sample_idx": sample_idx,
                },
            )
            if self.use_subject_split:
                if not self._match_subject_split(subject_id, fallback_split=split_value):
                    continue
            elif self.split_name and has_split and split_value != self.split_name:
                continue

            label = _to_int(labels[sample_idx], default=-1)
            if not self._is_label_valid(label):
                continue
            sample_indices.append(sample_idx)
            if not subject_id_hint and subject_id:
                subject_id_hint = subject_id

        if len(sample_indices) == 0:
            return
        if not subject_id_hint:
            subject_id_hint = self.file_subject_hints.get(file_idx, "")

        self.index.append(
            {
                "layout": "array_sequence",
                "file_idx": file_idx,
                "sample_indices": sample_indices,
                "label_key": label_key,
                "subject_id": subject_id_hint,
            }
        )

    def _index_group_layout(self, h5: h5py.File, file_idx: int) -> None:
        if self.sample_group_key in h5 and isinstance(h5[self.sample_group_key], h5py.Group):
            parent = h5[self.sample_group_key]
            paths = [f"/{self.sample_group_key}/{key}" for key in parent.keys()]
        else:
            paths = [f"/{key}" for key, value in h5.items() if isinstance(value, h5py.Group)]

        for group_path in paths:
            group = h5[group_path]
            if self._find_existing_key(group, self.label_keys) is None:
                continue
            split_value = ""
            has_split_key = (self.split_key in group) or (self.split_key in group.attrs)
            if self.split_name and has_split_key:
                split_value = str(self._read_scalar_from_group(group, [self.split_key], default="")).lower()

            subject_id = self._resolve_subject_id(
                h5,
                {
                    "layout": "sample_group",
                    "file_idx": file_idx,
                    "group_path": group_path,
                },
            )
            if self.use_subject_split:
                if not self._match_subject_split(subject_id, fallback_split=split_value):
                    continue
            elif self.split_name and has_split_key and split_value != self.split_name:
                continue
            self.index.append(
                {
                    "layout": "sample_group",
                    "file_idx": file_idx,
                    "group_path": group_path,
                    "subject_id": subject_id,
                }
            )

    def _get_signal_group(self, h5: h5py.File) -> h5py.Group | None:
        if self.cont_signal_group_key in h5 and isinstance(h5[self.cont_signal_group_key], h5py.Group):
            return h5[self.cont_signal_group_key]
        key = self._find_key_ci(h5, [self.cont_signal_group_key])
        if key is not None and isinstance(h5[key], h5py.Group):
            return h5[key]
        return None

    def _is_label_valid(self, label: int) -> bool:
        return label >= 0

    def _continuous_epoch_capacity(self, h5: h5py.File) -> int:
        signal_group = self._get_signal_group(h5)
        if signal_group is None:
            return 0

        min_epochs = None
        for mod_name in self.modality_names:
            channels = self._collect_continuous_channels(signal_group, mod_name)
            if len(channels) == 0:
                continue
            sample_rate = max(1, int(self.modality_sample_rates.get(mod_name, 1)))
            epoch_points = sample_rate * self.epoch_seconds
            if epoch_points <= 0:
                continue
            for _, ds in channels:
                total_points = self._dataset_total_points(ds)
                epochs = total_points // epoch_points
                min_epochs = epochs if min_epochs is None else min(min_epochs, epochs)
        return int(min_epochs or 0)

    @staticmethod
    def _dataset_total_points(ds: h5py.Dataset) -> int:
        if ds.ndim == 0:
            return 0
        if ds.ndim == 1:
            return int(ds.shape[0])
        # Flatten higher dimensions as a robust fallback.
        return int(np.prod(ds.shape))

    def _index_continuous_layout(self, h5: h5py.File, file_idx: int) -> None:
        labels = h5[self.cont_label_key]
        n_labels = int(labels.shape[0])
        n_cap = self._continuous_epoch_capacity(h5)
        n_samples = min(n_labels, n_cap) if n_cap > 0 else n_labels

        has_split = self.split_key in h5 and getattr(h5[self.split_key], "shape", (0,))[0] == n_labels
        for sample_idx in range(n_samples):
            split_value = ""
            if self.split_name and has_split:
                split_value = _to_str(h5[self.split_key][sample_idx]).lower()

            subject_id = self._resolve_subject_id(
                h5,
                {
                    "layout": "continuous",
                    "file_idx": file_idx,
                    "sample_idx": sample_idx,
                },
            )
            if self.use_subject_split:
                if not self._match_subject_split(subject_id, fallback_split=split_value):
                    continue
            elif self.split_name and has_split and split_value != self.split_name:
                continue
            label = _to_int(labels[sample_idx], default=-1)
            if not self._is_label_valid(label):
                continue
            self.index.append(
                {
                    "layout": "continuous",
                    "file_idx": file_idx,
                    "sample_idx": sample_idx,
                    "label_key": self.cont_label_key,
                    "subject_id": subject_id,
                }
            )

    def _index_continuous_layout_sequence(self, h5: h5py.File, file_idx: int) -> None:
        labels = h5[self.cont_label_key]
        n_labels = int(labels.shape[0])
        n_cap = self._continuous_epoch_capacity(h5)
        n_samples = min(n_labels, n_cap) if n_cap > 0 else n_labels
        has_split = self.split_key in h5 and getattr(h5[self.split_key], "shape", (0,))[0] == n_labels

        sample_indices: list[int] = []
        subject_id_hint = ""
        for sample_idx in range(n_samples):
            split_value = ""
            if self.split_name and has_split:
                split_value = _to_str(h5[self.split_key][sample_idx]).lower()

            subject_id = self._resolve_subject_id(
                h5,
                {
                    "layout": "continuous",
                    "file_idx": file_idx,
                    "sample_idx": sample_idx,
                },
            )
            if self.use_subject_split:
                if not self._match_subject_split(subject_id, fallback_split=split_value):
                    continue
            elif self.split_name and has_split and split_value != self.split_name:
                continue

            label = _to_int(labels[sample_idx], default=-1)
            if not self._is_label_valid(label):
                continue
            sample_indices.append(sample_idx)
            if not subject_id_hint and subject_id:
                subject_id_hint = subject_id

        if len(sample_indices) == 0:
            return
        if not subject_id_hint:
            subject_id_hint = self.file_subject_hints.get(file_idx, "")

        self.index.append(
            {
                "layout": "continuous_sequence",
                "file_idx": file_idx,
                "sample_indices": sample_indices,
                "label_key": self.cont_label_key,
                "subject_id": subject_id_hint,
            }
        )

    def _get_worker_file(self, file_idx: int) -> h5py.File:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        handle_map = self._worker_handles.setdefault(worker_id, {})
        if file_idx not in handle_map:
            handle_map[file_idx] = h5py.File(self.files[file_idx], "r")
        return handle_map[file_idx]

    def _read_scalar_from_array(self, h5: h5py.File, keys: list[str], index: int, default: Any) -> Any:
        for key in keys:
            if key not in h5:
                continue
            ds = h5[key]
            if len(getattr(ds, "shape", ())) == 0:
                return _decode_scalar(ds[()])
            if ds.shape[0] > index:
                return _decode_scalar(ds[index])
        return default

    def _read_scalar_from_group(self, group: h5py.Group, keys: list[str], default: Any) -> Any:
        for key in keys:
            if key in group:
                ds = group[key]
                if isinstance(ds, h5py.Dataset):
                    return _decode_scalar(ds[()])
                return _decode_scalar(ds)
            if key in group.attrs:
                return _decode_scalar(group.attrs[key])
        return default

    def _channel_name_candidates(self, modality_name: str) -> list[str]:
        cfg_candidates = self.modality_channel_name_keys.get(modality_name, [])
        auto_candidates = [
            f"{modality_name}_channel_names",
            f"{modality_name}_channels",
            f"{modality_name}_ch_names",
            f"{modality_name}_channel_labels",
        ]
        out: list[str] = []
        for key in list(cfg_candidates) + auto_candidates:
            if key not in out:
                out.append(key)
        return out

    def _read_channel_names(
        self,
        container: h5py.Group | h5py.File,
        modality_name: str,
        sample_idx: int | None = None,
    ) -> list[str]:
        for key in self._channel_name_candidates(modality_name):
            if key in container:
                ds = container[key]
                value = None
                if isinstance(ds, h5py.Dataset):
                    shape = getattr(ds, "shape", ())
                    if sample_idx is not None and len(shape) >= 2 and shape[0] > sample_idx:
                        value = ds[sample_idx]
                    elif sample_idx is not None and len(shape) == 1 and shape[0] > sample_idx and shape[0] > 64:
                        value = ds[sample_idx]
                    else:
                        value = ds[()]
                else:
                    value = ds
                names = _to_str_list(value)
                if names:
                    return names

            if key in container.attrs:
                names = _to_str_list(container.attrs[key])
                if names:
                    return names
        return []

    def _resolve_file_dataset_name(self, h5: h5py.File, file_path: str) -> str:
        for key in self.dataset_name_keys:
            if key in h5:
                raw = _decode_scalar(h5[key][()])
                name = _to_str(raw)
                if name:
                    return _normalize_dataset_name(name)
            if key in h5.attrs:
                name = _to_str(h5.attrs[key])
                if name:
                    return _normalize_dataset_name(name)

        basename = Path(file_path).name
        basename_norm = basename.lower()
        for pattern, mapped_name in self.file_dataset_name_overrides.items():
            if pattern.lower() in basename_norm:
                return _normalize_dataset_name(str(mapped_name))

        return _normalize_dataset_name(Path(file_path).stem)

    def _resolve_dataset_name(self, h5: h5py.File, item: dict[str, Any]) -> str:
        file_hint = self.file_dataset_name_hints.get(item["file_idx"], "")
        if item["layout"] in ("array", "continuous"):
            sample_idx = int(item["sample_idx"])
            raw = self._read_scalar_from_array(h5, self.dataset_name_keys, sample_idx, default="")
            name = _to_str(raw)
            if name:
                return _normalize_dataset_name(name)
            for key in self.dataset_name_keys:
                if key in h5 and isinstance(h5[key], h5py.Dataset) and h5[key].ndim == 0:
                    root_name = _to_str(h5[key][()])
                    if root_name:
                        return _normalize_dataset_name(root_name)
                if key in h5.attrs:
                    root_name = _to_str(h5.attrs[key])
                    if root_name:
                        return _normalize_dataset_name(root_name)
            return file_hint

        group = h5[item["group_path"]]
        raw = self._read_scalar_from_group(group, self.dataset_name_keys, default="")
        name = _to_str(raw)
        if name:
            return _normalize_dataset_name(name)
        return file_hint

    def _resolve_dataset_id(self, h5: h5py.File, item: dict[str, Any], dataset_name: str) -> int:
        default = item["file_idx"]
        if item["layout"] in ("array", "continuous"):
            sample_idx = int(item["sample_idx"])
            raw_id = self._read_scalar_from_array(h5, self.dataset_id_keys, sample_idx, default=None)
        else:
            group = h5[item["group_path"]]
            raw_id = self._read_scalar_from_group(group, self.dataset_id_keys, default=None)

        if raw_id is not None:
            parsed = _to_int(raw_id, default=-1)
            if parsed >= 0:
                return parsed

        norm_name = _normalize_dataset_name(dataset_name)
        if norm_name in self.dataset_to_id:
            return int(self.dataset_to_id[norm_name])
        return int(default)

    def _read_modality_array(
        self,
        container: h5py.Group | h5py.File,
        aliases: list[str],
        sample_idx: int | None = None,
    ) -> np.ndarray | None:
        for key in aliases:
            if key not in container:
                continue
            ds = container[key]
            if sample_idx is None:
                array = np.asarray(ds)
            else:
                if getattr(ds, "shape", (0,))[0] <= sample_idx:
                    continue
                array = np.asarray(ds[sample_idx])
            return array.astype(np.float32, copy=False)
        return None

    def _convert_modality(self, arr: np.ndarray | None, modality_name: str) -> torch.Tensor | None:
        if arr is None:
            return None

        tensor = torch.from_numpy(arr)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)

        if tensor.ndim != 2:
            if self.strict_modality:
                raise ValueError(f"Invalid modality shape for '{modality_name}': {tuple(tensor.shape)}")
            return None
        return tensor.float()

    def _modality_alias_match(self, channel_name: str, modality_name: str) -> bool:
        norm_channel = _norm_token(channel_name)
        aliases = [str(x) for x in self.continuous_channel_aliases.get(modality_name, [])]
        if len(aliases) == 0:
            return False
        for alias in aliases:
            if _norm_token(alias) in norm_channel:
                return True
        return False

    def _collect_continuous_channels(
        self,
        signal_group: h5py.Group,
        modality_name: str,
    ) -> list[tuple[str, h5py.Dataset]]:
        collected: list[tuple[str, h5py.Dataset]] = []

        aliases = [modality_name] + list(self.modality_key_map.get(modality_name, []))
        key = self._find_key_ci(signal_group, aliases)
        if key is not None:
            obj = signal_group[key]
            if isinstance(obj, h5py.Group):
                for ch_name in obj.keys():
                    ch_ds = obj[ch_name]
                    if isinstance(ch_ds, h5py.Dataset):
                        collected.append((ch_name, ch_ds))
            elif isinstance(obj, h5py.Dataset):
                collected.append((key, obj))

        # Demo special-case: ECG/EMG may both be stored under signals/emg.
        if len(collected) == 0 and modality_name in {"ecg", "emg"}:
            emg_key = self._find_key_ci(signal_group, ["emg", "EMG"])
            if emg_key is not None and isinstance(signal_group[emg_key], h5py.Group):
                emg_group = signal_group[emg_key]
                for ch_name in emg_group.keys():
                    ch_ds = emg_group[ch_name]
                    if not isinstance(ch_ds, h5py.Dataset):
                        continue
                    if self._modality_alias_match(ch_name, modality_name):
                        collected.append((ch_name, ch_ds))

        # Last fallback: scan direct channel datasets under /signals.
        if len(collected) == 0:
            for ch_name in signal_group.keys():
                ch_ds = signal_group[ch_name]
                if not isinstance(ch_ds, h5py.Dataset):
                    continue
                if self._modality_alias_match(ch_name, modality_name):
                    collected.append((ch_name, ch_ds))

        # Deduplicate by channel name.
        dedup: dict[str, h5py.Dataset] = {}
        for ch_name, ds in collected:
            if ch_name not in dedup:
                dedup[ch_name] = ds
        return list(dedup.items())

    def _slice_signal_epoch(
        self,
        ds: h5py.Dataset,
        sample_idx: int,
        start: int,
        end: int,
        target_len: int,
    ) -> np.ndarray:
        if ds.ndim == 1:
            if start >= ds.shape[0]:
                return np.zeros((target_len,), dtype=np.float32)
            arr = np.asarray(ds[start:min(end, ds.shape[0])], dtype=np.float32)
        elif ds.ndim == 2:
            # Case A: [N_epoch, T_epoch]
            if ds.shape[0] > sample_idx and ds.shape[1] <= target_len * 4:
                arr = np.asarray(ds[sample_idx], dtype=np.float32).reshape(-1)
            else:
                # Fallback: flatten as a continuous stream.
                flat = np.asarray(ds).reshape(-1)
                if start >= flat.shape[0]:
                    return np.zeros((target_len,), dtype=np.float32)
                arr = flat[start:min(end, flat.shape[0])].astype(np.float32, copy=False)
        else:
            flat = np.asarray(ds).reshape(-1).astype(np.float32, copy=False)
            if start >= flat.shape[0]:
                return np.zeros((target_len,), dtype=np.float32)
            arr = flat[start:min(end, flat.shape[0])]

        if arr.shape[0] < target_len:
            pad = np.zeros((target_len - arr.shape[0],), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > target_len:
            arr = arr[:target_len]
        return arr.astype(np.float32, copy=False)

    def _read_continuous_modality_epoch(
        self,
        h5: h5py.File,
        sample_idx: int,
        modality_name: str,
    ) -> tuple[torch.Tensor | None, list[str]]:
        signal_group = self._get_signal_group(h5)
        if signal_group is None:
            return None, []

        channels = self._collect_continuous_channels(signal_group, modality_name)
        if len(channels) == 0:
            return None, []

        sample_rate = max(1, int(self.modality_sample_rates.get(modality_name, 1)))
        target_len = sample_rate * self.epoch_seconds
        start = sample_idx * target_len
        end = start + target_len

        arrs: list[np.ndarray] = []
        channel_names: list[str] = []
        for ch_name, ds in channels:
            arr = self._slice_signal_epoch(ds, sample_idx=sample_idx, start=start, end=end, target_len=target_len)
            arrs.append(arr)
            channel_names.append(ch_name)

        if len(arrs) == 0:
            return None, []
        mod_arr = np.stack(arrs, axis=0).astype(np.float32, copy=False)
        return self._convert_modality(mod_arr, modality_name), channel_names

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.index[idx]
        h5 = self._get_worker_file(item["file_idx"])
        file_name = Path(self.files[item["file_idx"]]).name
        subject_id = self._clean_subject_id(item.get("subject_id", ""))

        modalities: dict[str, torch.Tensor | None] = {}
        channel_names: dict[str, list[str]] = {}

        if item["layout"] == "array":
            sample_idx = int(item["sample_idx"])
            label = _to_int(self._read_scalar_from_array(h5, [item["label_key"]], sample_idx, default=0), default=0)

            for mod_name in self.modality_names:
                aliases = self.modality_key_map.get(mod_name, [mod_name])
                arr = self._read_modality_array(h5, aliases, sample_idx=sample_idx)
                modalities[mod_name] = self._convert_modality(arr, mod_name)
                channel_names[mod_name] = self._read_channel_names(h5, mod_name, sample_idx=sample_idx)

            dataset_name = self._resolve_dataset_name(h5, item)
            dataset_id = self._resolve_dataset_id(h5, item, dataset_name)
            sample_id = f"{file_name}:{sample_idx}"
        elif item["layout"] == "array_sequence":
            sample_indices = [int(x) for x in item.get("sample_indices", [])]
            labels: list[int] = []
            seq_modalities: dict[str, list[torch.Tensor | None]] = {k: [] for k in self.modality_names}
            seq_channel_names: dict[str, list[list[str]]] = {k: [] for k in self.modality_names}

            for sample_idx in sample_indices:
                labels.append(
                    _to_int(self._read_scalar_from_array(h5, [item["label_key"]], sample_idx, default=0), default=0)
                )
                for mod_name in self.modality_names:
                    aliases = self.modality_key_map.get(mod_name, [mod_name])
                    arr = self._read_modality_array(h5, aliases, sample_idx=sample_idx)
                    seq_modalities[mod_name].append(self._convert_modality(arr, mod_name))
                    seq_channel_names[mod_name].append(self._read_channel_names(h5, mod_name, sample_idx=sample_idx))

            first_idx = sample_indices[0]
            item_for_meta = {
                "layout": "array",
                "file_idx": item["file_idx"],
                "sample_idx": first_idx,
            }
            dataset_name = self._resolve_dataset_name(h5, item_for_meta)
            dataset_id = self._resolve_dataset_id(h5, item_for_meta, dataset_name)
            sample_id = f"{file_name}:seq:{first_idx}-{sample_indices[-1]}"

            if not subject_id:
                subject_id = self._resolve_subject_id(
                    h5,
                    {
                        "layout": "array",
                        "file_idx": item["file_idx"],
                        "sample_idx": first_idx,
                    },
                )
            return {
                "modalities": seq_modalities,
                "channel_names": seq_channel_names,
                "labels": labels,
                "seq_len": len(labels),
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "subject_id": subject_id,
                "sample_id": sample_id,
            }
        elif item["layout"] == "sample_group":
            group = h5[item["group_path"]]
            label = _to_int(self._read_scalar_from_group(group, self.label_keys, default=0), default=0)

            for mod_name in self.modality_names:
                aliases = self.modality_key_map.get(mod_name, [mod_name])
                arr = self._read_modality_array(group, aliases, sample_idx=None)
                modalities[mod_name] = self._convert_modality(arr, mod_name)
                channel_names[mod_name] = self._read_channel_names(group, mod_name, sample_idx=None)

            dataset_name = self._resolve_dataset_name(h5, item)
            dataset_id = self._resolve_dataset_id(h5, item, dataset_name)
            sample_id = f"{file_name}:{item['group_path']}"
        elif item["layout"] == "continuous_sequence":
            sample_indices = [int(x) for x in item.get("sample_indices", [])]
            labels: list[int] = []
            seq_modalities: dict[str, list[torch.Tensor | None]] = {k: [] for k in self.modality_names}
            seq_channel_names: dict[str, list[list[str]]] = {k: [] for k in self.modality_names}

            for sample_idx in sample_indices:
                labels.append(
                    _to_int(self._read_scalar_from_array(h5, [self.cont_label_key], sample_idx, default=0), default=0)
                )
                for mod_name in self.modality_names:
                    mod_tensor, mod_channel_names = self._read_continuous_modality_epoch(
                        h5,
                        sample_idx=sample_idx,
                        modality_name=mod_name,
                    )
                    seq_modalities[mod_name].append(mod_tensor)
                    seq_channel_names[mod_name].append(mod_channel_names)

            first_idx = sample_indices[0]
            item_for_meta = {
                "layout": "continuous",
                "file_idx": item["file_idx"],
                "sample_idx": first_idx,
            }
            dataset_name = self._resolve_dataset_name(h5, item_for_meta)
            dataset_id = self._resolve_dataset_id(h5, item_for_meta, dataset_name)
            sample_id = f"{file_name}:seq:{first_idx}-{sample_indices[-1]}"

            if not subject_id:
                subject_id = self._resolve_subject_id(
                    h5,
                    {
                        "layout": "continuous",
                        "file_idx": item["file_idx"],
                        "sample_idx": first_idx,
                    },
                )
            return {
                "modalities": seq_modalities,
                "channel_names": seq_channel_names,
                "labels": labels,
                "seq_len": len(labels),
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "subject_id": subject_id,
                "sample_id": sample_id,
            }
        else:
            sample_idx = int(item["sample_idx"])
            label = _to_int(self._read_scalar_from_array(h5, [self.cont_label_key], sample_idx, default=0), default=0)

            for mod_name in self.modality_names:
                mod_tensor, mod_channel_names = self._read_continuous_modality_epoch(h5, sample_idx=sample_idx, modality_name=mod_name)
                modalities[mod_name] = mod_tensor
                channel_names[mod_name] = mod_channel_names

            dataset_name = self._resolve_dataset_name(h5, item)
            dataset_id = self._resolve_dataset_id(h5, item, dataset_name)
            sample_id = f"{file_name}:{sample_idx}"

        if not subject_id:
            subject_id = self._resolve_subject_id(h5, item)

        return {
            "modalities": modalities,
            "channel_names": channel_names,
            "label": label,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "subject_id": subject_id,
            "sample_id": sample_id,
        }

    def __del__(self) -> None:
        for handle_map in self._worker_handles.values():
            for h5 in handle_map.values():
                try:
                    h5.close()
                except Exception:
                    pass


def _main() -> None:
    parser = argparse.ArgumentParser(description="SleepH5Dataset quick reader test")
    parser.add_argument("--h5", type=str, required=True, help="Path to .h5 file")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config")
    parser.add_argument("--split", type=str, default="", help="Split name, empty means no split filtering")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=2)
    args = parser.parse_args()

    from mainmodel.data.collate import SleepCollator
    from mainmodel.utils.config import load_config

    cfg = load_config(args.config)
    ds = SleepH5Dataset(
        files=[args.h5],
        split_name=(args.split if args.split else None),
        data_cfg=cfg["data"],
        model_cfg=cfg["model"],
    )
    print(f"[Dataset] len={len(ds)}")

    show_n = min(len(ds), max(1, args.num_samples))
    for i in range(show_n):
        item = ds[i]
        if "label" in item:
            print(f"[Sample {i}] id={item['sample_id']} subject={item.get('subject_id', '')} label={item['label']}")
        else:
            print(
                f"[Sample {i}] id={item['sample_id']} subject={item.get('subject_id', '')} "
                f"seq_len={item.get('seq_len', 0)}"
            )
        for mod_name, x in item["modalities"].items():
            ch_names = item["channel_names"].get(mod_name, [])
            if isinstance(x, list):
                one = x[0] if len(x) > 0 else None
                print(
                    f"  - {mod_name}: seq={len(x)} x0_shape={tuple(one.shape) if one is not None else None}, "
                    f"channels0={ch_names[0] if len(ch_names) > 0 else []}"
                )
            else:
                print(f"  - {mod_name}: shape={tuple(x.shape) if x is not None else None}, channels={ch_names}")

    collator = SleepCollator(data_cfg=cfg["data"], model_cfg=cfg["model"])
    loader = DataLoader(ds, batch_size=max(1, args.batch_size), shuffle=False, num_workers=0, collate_fn=collator)
    batch = next(iter(loader))
    print(f"[Batch] labels={tuple(batch['labels'].shape)}")
    for mod_name in batch["modalities"].keys():
        print(
            f"  - {mod_name}: x={tuple(batch['modalities'][mod_name].shape)}, "
            f"modality_mask={tuple(batch['modality_mask'][mod_name].shape)}, "
            f"channel_mask={tuple(batch['channel_mask'][mod_name].shape)}"
        )


if __name__ == "__main__":
    _main()


