from __future__ import annotations

import argparse
import logging
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm


# Sleep stage labels
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
MOVE = 5
UNK = 6

ANN2LABEL = {
    "Sleep stage W": W,
    "Sleep stage 1": N1,
    "Sleep stage 2": N2,
    "Sleep stage 3": N3,
    "Sleep stage 4": N3,  # merge N3 + N4
    "Sleep stage R": REM,
    "Movement time": MOVE,
    "Sleep stage ?": UNK,
}

DEFAULT_MODALITIES = [
    "eeg",
    "eog",
    "emg",
    "ecg",
    "airflow",
    "thoracoabdominal",
    "spo2",
    "ppg",
]

MODALITY_ALIASES = {
    "eeg": ["EEG", "FPZCZ", "PZOZ", "F3", "F4", "C3", "C4", "O1", "O2"],
    "eog": ["EOG", "LOC", "ROC", "LEFTEOG", "RIGHTEOG", "HORIZONTALEOG", "E1", "E2"],
    "emg": ["EMG", "CHIN", "SUBMENTAL", "LEG"],
    "ecg": ["ECG", "EKG"],
    "airflow": ["AIRFLOW", "NASAL", "ORONASAL", "FLOW"],
    "thoracoabdominal": ["THORAX", "ABDOMEN", "RESP", "CHEST", "BELT", "ABD"],
    "spo2": ["SPO2", "SAO2", "OXY"],
    "ppg": ["PPG", "PLETH"],
}

TARGET_FS = {
    "eeg": 100,
    "eog": 100,
    "emg": 100,
    "ecg": 100,
    "airflow": 10,
    "thoracoabdominal": 10,
    "spo2": 10,
    "ppg": 10,
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[\s\-_:/()]+", "", name).upper()


def _infer_modality(channel_name: str, enabled_modalities: set[str]) -> str | None:
    norm_name = _normalize_name(channel_name)
    for modality in DEFAULT_MODALITIES:
        if modality not in enabled_modalities:
            continue
        aliases = MODALITY_ALIASES.get(modality, [])
        if any(alias in norm_name for alias in aliases):
            return modality
    return None


def _pair_key(edf_path: Path) -> str:
    stem = edf_path.stem.upper().replace("-PSG", "").replace("-HYPNOGRAM", "")
    # Sleep-EDF pairing key (e.g. SC4001 from SC4001E0 / SC4001EC)
    return stem[:6]


def _subject_id_from_record_name(record_name: str) -> str:
    token = record_name.upper().replace("-PSG", "").replace("-HYPNOGRAM", "")
    match = re.match(r"^([A-Z]{2}\d{3})\d[A-Z]\d$", token)
    if match is not None:
        return match.group(1)
    # Conservative fallback: keep first 5 chars if it matches SC/ST style,
    # otherwise keep the full token to avoid information loss.
    if re.match(r"^[A-Z]{2}\d{3}.*$", token):
        return token[:5]
    return token


def pair_sleep_edf_files(data_dir: Path) -> list[tuple[Path, Path]]:
    psg_files = sorted(data_dir.glob("*-PSG.edf"))
    hyp_files = sorted(data_dir.glob("*-Hypnogram.edf"))

    if len(psg_files) == 0:
        raise FileNotFoundError(f"No PSG EDF files found in: {data_dir}")
    if len(hyp_files) == 0:
        raise FileNotFoundError(f"No Hypnogram EDF files found in: {data_dir}")

    hyp_by_key: dict[str, list[Path]] = {}
    for hyp in hyp_files:
        hyp_by_key.setdefault(_pair_key(hyp), []).append(hyp)

    pairs: list[tuple[Path, Path]] = []
    used_hyp: set[Path] = set()
    for psg in psg_files:
        key = _pair_key(psg)
        cands = [h for h in hyp_by_key.get(key, []) if h not in used_hyp]
        if not cands:
            raise FileNotFoundError(f"Cannot find matching hypnogram for {psg.name}")
        hyp = sorted(cands)[0]
        used_hyp.add(hyp)
        pairs.append((psg, hyp))
    return pairs


def _bandpass_filter(sig: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    if fs <= 0:
        return sig
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        return sig
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


def _resample_signal(sig: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
    if fs_in <= 0:
        return sig
    if abs(fs_in - fs_out) < 1e-6:
        return sig.astype(np.float32)
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(sig, ratio.numerator, ratio.denominator).astype(np.float32)


def _build_labels(ann_reader: pyedflib.EdfReader, epoch_seconds: int) -> np.ndarray:
    onsets, durations, stages = ann_reader.readAnnotations()
    labels: list[np.ndarray] = []
    cursor_sec = 0

    for onset, duration, stage_raw in zip(onsets, durations, stages):
        onset_sec = int(onset)
        duration_sec = int(duration)
        stage_name = str("".join(stage_raw) if isinstance(stage_raw, (list, tuple, np.ndarray)) else stage_raw)

        if onset_sec > cursor_sec:
            gap_sec = onset_sec - cursor_sec
            gap_epochs = int(round(gap_sec / epoch_seconds))
            if gap_epochs > 0:
                labels.append(np.full(gap_epochs, UNK, dtype=np.int32))
                cursor_sec += gap_epochs * epoch_seconds

        if duration_sec <= 0:
            continue
        label = ANN2LABEL.get(stage_name, UNK)
        n_epochs = int(round(duration_sec / epoch_seconds))
        if n_epochs <= 0:
            continue
        labels.append(np.full(n_epochs, label, dtype=np.int32))
        cursor_sec += n_epochs * epoch_seconds

    if not labels:
        return np.empty((0,), dtype=np.int32)
    return np.concatenate(labels, axis=0)


def _select_sleep_window(y: np.ndarray, epoch_seconds: int, wake_edge_mins: int) -> np.ndarray:
    if y.size == 0:
        return np.empty((0,), dtype=np.int64)
    non_w = np.where(y != W)[0]
    if non_w.size == 0:
        return np.arange(len(y))
    edge_epochs = int((wake_edge_mins * 60) / epoch_seconds)
    start = max(0, int(non_w[0]) - edge_epochs)
    end = min(len(y) - 1, int(non_w[-1]) + edge_epochs)
    return np.arange(start, end + 1)


def _extract_epoch_tensor(
    psg_reader: pyedflib.EdfReader,
    channel_indices: list[int],
    channel_names: list[str],
    modality: str,
    epoch_seconds: int,
    apply_filter: bool,
    low_hz: float,
    high_hz: float,
    max_eeg_channels: int,
) -> tuple[np.ndarray, list[str], int]:
    target_fs = int(TARGET_FS[modality])
    channel_arrays: list[np.ndarray] = []
    kept_names: list[str] = []

    for idx, ch_name in zip(channel_indices, channel_names):
        fs_in = float(psg_reader.getSampleFrequency(idx))
        sig = psg_reader.readSignal(idx).astype(np.float32)

        if apply_filter and modality in {"eeg", "eog", "emg"}:
            sig = _bandpass_filter(sig, fs=fs_in, low_hz=low_hz, high_hz=high_hz)

        sig = _resample_signal(sig, fs_in=fs_in, fs_out=target_fs)
        channel_arrays.append(sig)
        kept_names.append(ch_name)

    if not channel_arrays:
        return np.empty((0, 0, 0), dtype=np.float32), [], target_fs

    if modality == "eeg" and max_eeg_channels > 0:
        channel_arrays = channel_arrays[:max_eeg_channels]
        kept_names = kept_names[:max_eeg_channels]

    min_len = min(arr.shape[0] for arr in channel_arrays)
    epoch_points = int(epoch_seconds * target_fs)
    n_epochs = min_len // epoch_points
    if n_epochs <= 0:
        return np.empty((0, 0, 0), dtype=np.float32), [], target_fs

    trimmed = [arr[: n_epochs * epoch_points] for arr in channel_arrays]
    stacked = np.stack(trimmed, axis=0)  # [C, Total]
    x = stacked.reshape(stacked.shape[0], n_epochs, epoch_points).transpose(1, 0, 2)  # [N, C, T]
    return x.astype(np.float32), kept_names, target_fs


def convert_one_record(
    psg_path: Path,
    hyp_path: Path,
    output_path: Path,
    dataset_name: str,
    enabled_modalities: set[str],
    wake_edge_mins: int,
    drop_move_unknown: bool,
    apply_filter: bool,
    low_hz: float,
    high_hz: float,
    max_eeg_channels: int,
) -> dict[str, Any]:
    psg_reader = pyedflib.EdfReader(str(psg_path))
    ann_reader = pyedflib.EdfReader(str(hyp_path))

    try:
        if psg_reader.getStartdatetime() != ann_reader.getStartdatetime():
            raise ValueError(f"PSG and annotation start time mismatch: {psg_path.name}")

        epoch_seconds = int(psg_reader.datarecord_duration)
        n_records = int(psg_reader.datarecords_in_file)
        if epoch_seconds == 60:
            # Known Sleep-EDF special case, split one 60s record into two 30s epochs.
            epoch_seconds = 30
            n_records = n_records * 2

        labels = _build_labels(ann_reader, epoch_seconds=epoch_seconds)
        labels = labels[:n_records]

        signal_labels = [str(x) for x in psg_reader.getSignalLabels()]
        modality_to_indices: dict[str, list[int]] = {m: [] for m in enabled_modalities}
        modality_to_names: dict[str, list[str]] = {m: [] for m in enabled_modalities}

        for idx, ch_name in enumerate(signal_labels):
            modality = _infer_modality(ch_name, enabled_modalities=enabled_modalities)
            if modality is None:
                continue
            modality_to_indices[modality].append(idx)
            modality_to_names[modality].append(ch_name)

        modality_data: dict[str, np.ndarray] = {}
        modality_names: dict[str, list[str]] = {}
        modality_fs: dict[str, int] = {}
        for modality in DEFAULT_MODALITIES:
            if modality not in enabled_modalities:
                continue
            ch_indices = modality_to_indices.get(modality, [])
            ch_names = modality_to_names.get(modality, [])
            if len(ch_indices) == 0:
                continue
            x_mod, kept_names, fs_mod = _extract_epoch_tensor(
                psg_reader=psg_reader,
                channel_indices=ch_indices,
                channel_names=ch_names,
                modality=modality,
                epoch_seconds=epoch_seconds,
                apply_filter=apply_filter,
                low_hz=low_hz,
                high_hz=high_hz,
                max_eeg_channels=max_eeg_channels,
            )
            if x_mod.shape[0] == 0:
                continue
            modality_data[modality] = x_mod
            modality_names[modality] = kept_names
            modality_fs[modality] = fs_mod

        if "eeg" not in modality_data:
            raise ValueError(f"No EEG channels were extracted from {psg_path.name}")
        if labels.size == 0:
            raise ValueError(f"No labels generated from hypnogram: {hyp_path.name}")

        n_epochs = min([labels.shape[0], *[x.shape[0] for x in modality_data.values()]])
        labels = labels[:n_epochs]
        for modality in list(modality_data.keys()):
            modality_data[modality] = modality_data[modality][:n_epochs]

        keep_idx = _select_sleep_window(labels, epoch_seconds=epoch_seconds, wake_edge_mins=wake_edge_mins)
        labels = labels[keep_idx]
        for modality in list(modality_data.keys()):
            modality_data[modality] = modality_data[modality][keep_idx]

        if drop_move_unknown:
            valid_mask = (labels != MOVE) & (labels != UNK)
            labels = labels[valid_mask]
            for modality in list(modality_data.keys()):
                modality_data[modality] = modality_data[modality][valid_mask]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(output_path, "w") as h5:
            for modality, x_mod in modality_data.items():
                h5.create_dataset(modality, data=x_mod.astype(np.float32), compression="gzip")
                h5.create_dataset(
                    f"{modality}_channel_names",
                    data=np.asarray(modality_names[modality], dtype=str_dtype),
                )
                h5.create_dataset(f"{modality}_sample_rate", data=np.int32(modality_fs[modality]))

            y = labels.astype(np.int32)
            record_name = psg_path.stem.replace("-PSG", "")
            subject_id = _subject_id_from_record_name(record_name)
            h5.create_dataset("label", data=y, compression="gzip")
            h5.create_dataset("hypnogram", data=y, compression="gzip")
            h5.create_dataset("dataset_name", data=np.asarray(dataset_name, dtype=str_dtype))
            h5.create_dataset("record_name", data=np.asarray(record_name, dtype=str_dtype))
            h5.create_dataset("subject_id", data=np.asarray(subject_id, dtype=str_dtype))
            h5.create_dataset("source_psg", data=np.asarray(str(psg_path), dtype=str_dtype))
            h5.create_dataset("source_hypnogram", data=np.asarray(str(hyp_path), dtype=str_dtype))
            h5.create_dataset("epoch_seconds", data=np.int32(epoch_seconds))

        return {
            "record": psg_path.name,
            "output": str(output_path),
            "n_epochs": int(labels.shape[0]),
            "modalities": {k: list(v.shape) for k, v in modality_data.items()},
            "eeg_channels": modality_names.get("eeg", []),
        }
    finally:
        psg_reader.close()
        ann_reader.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert Sleep-EDF *.edf (PSG + Hypnogram) into multimodal *.h5 files."
    )
    parser.add_argument("--data_dir", type=str, default="./dset", help="Directory with EDF files.")
    parser.add_argument("--output_dir", type=str, default="./dset_h5", help="Output directory for H5 files.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SLEEPEDF",
        help="Dataset name stored in H5 (e.g. SLEEPEDF).",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=DEFAULT_MODALITIES,
        help="Modalities to export. Default: eeg eog emg ecg airflow thoracoabdominal spo2 ppg",
    )
    parser.add_argument(
        "--max_eeg_channels",
        type=int,
        default=6,
        help="Keep at most this many EEG channels per record. Default: 6",
    )
    parser.add_argument("--wake_edge_mins", type=int, default=30, help="Keep this many wake minutes around sleep.")
    parser.add_argument(
        "--keep_move_unknown",
        action="store_true",
        help="If set, keep MOVE/UNK labels. By default they are removed.",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Disable 0.3-35 Hz filtering for EEG/EOG/EMG.",
    )
    parser.add_argument("--low_hz", type=float, default=0.3, help="Band-pass low cut (Hz).")
    parser.add_argument("--high_hz", type=float, default=35.0, help="Band-pass high cut (Hz).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing H5 files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    enabled_modalities = {m.lower() for m in args.modalities}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("prepare_sleep_edf_to_h5")
    logger.info("Input dir: %s", data_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Modalities: %s", sorted(enabled_modalities))
    logger.info("Max EEG channels: %d", int(args.max_eeg_channels))

    pairs = pair_sleep_edf_files(data_dir)
    logger.info("Found %d PSG/Hypnogram pairs.", len(pairs))

    ok = 0
    failed = 0
    for psg_path, hyp_path in tqdm(pairs, desc="Converting EDF to H5"):
        out_name = psg_path.name.replace("-PSG.edf", ".h5")
        out_path = output_dir / out_name
        if out_path.exists() and not args.overwrite:
            logger.info("Skip existing: %s", out_path.name)
            continue

        try:
            result = convert_one_record(
                psg_path=psg_path,
                hyp_path=hyp_path,
                output_path=out_path,
                dataset_name=args.dataset_name,
                enabled_modalities=enabled_modalities,
                wake_edge_mins=int(args.wake_edge_mins),
                drop_move_unknown=not bool(args.keep_move_unknown),
                apply_filter=not bool(args.no_filter),
                low_hz=float(args.low_hz),
                high_hz=float(args.high_hz),
                max_eeg_channels=int(args.max_eeg_channels),
            )
            ok += 1
            logger.info(
                "OK %s -> %s | epochs=%d | modalities=%s | eeg_channels=%s",
                result["record"],
                Path(result["output"]).name,
                result["n_epochs"],
                result["modalities"],
                result["eeg_channels"],
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.exception("FAILED %s: %s", psg_path.name, exc)

    logger.info("Done. success=%d, failed=%d", ok, failed)


if __name__ == "__main__":
    main()
