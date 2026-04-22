# 睡眠模型基座（SleepGLM）

## 项目定位

`SleepGLM` 是一个可扩展的睡眠多模态训练框架。当前下游任务是 **睡眠分期（sleep staging）**，代码中也预留了下游任务扩展入口（`training.downstream_task`）。

## 当前能力

- 支持 `torchrun` + PyTorch DDP 训练。
- 支持多模态输入：`eeg/eog/emg/ecg/airflow/thoracoabdominal/spo2/ppg`。
- 支持三种 H5 布局读取：
  - 数组布局：`/eeg [N,C,T]`, `/label [N]`
  - sample-group 布局：`/samples/<id>/...`
  - 连续 PSG 布局：`/signals/...` + `/hypnogram`
- 支持通道标准化适配（name-based + alias + 缺失通道策略）。
- 支持 stage1/stage2 预训练与 stage3 下游微调。

## 数据划分（重点）

### 1) 使用 split 文件（推荐）

当前推荐使用固定的 `split_file`，直接指定 train/val/test 的文件列表。打开方式：

```yaml
data:
  split_file: configs/splits/sleepedf.json
  split_root_dir: dset/Sleep-EDF
```

当配置了 `split_file` 时：

- 训练直接读取 `files.train/val/test`
- 不再依赖 H5 内部 `split` 字段做二次切分
- 适合固定实验划分，复现更稳定

### 2) Sleep-EDF 按人划分规则

在本仓库中，Sleep-EDF 使用如下受试者提取规则：

- `SC4001E0` -> `SC400`
- `SC4002E0` -> `SC400`
- `ST7011J0` -> `ST701`

对应正则：

```yaml
data:
  subject_id_from_filename_regex: "^([A-Za-z]{2}\\d{3})\\d[A-Za-z]\\d$"
  subject_id_from_filename_regex_group: 1
```

这能保证同一人的不同夜晚不会被分到不同集合。

## Sleep-EDF 转 H5

脚本：`dset/prepare_sleep_edf_to_h5.py`

```bash
python dset/prepare_sleep_edf_to_h5.py \
  --data_dir ./dset/Sleep-EDF-edf \
  --output_dir ./dset/Sleep-EDF \
  --dataset_name SLEEPEDF
```

导出时会写入：

- `label`
- `hypnogram`
- `dataset_name`
- `record_name`
- `subject_id`（新增）

## 训练命令

```bash
# stage1
torchrun --nproc_per_node=4 -m mainmodel.train --config configs/stage1_eeg_jepa.yaml

# stage2
torchrun --nproc_per_node=4 -m mainmodel.train --config configs/stage2_multimodal_pretrain.yaml

# stage3（下游微调）
torchrun --nproc_per_node=4 -m mainmodel.train --config configs/stage3_downstream_finetune.yaml
```

## 关键配置说明

- `training.downstream_task`: 当前支持 `sleep_staging`。
- `data.split_file`: 固定划分清单（json）。
- `data.split_root_dir`: `split_file` 中相对文件名的根目录。
- `data.subject_id_key_candidates`: 受试者 ID 候选字段（用于统计/兼容场景）。

## GitHub README 乱码

本文件已改为 UTF-8 编码。若历史提交仍显示乱码，通常是旧版本文件编码导致；重新提交当前版本即可在 GitHub 正常显示。
