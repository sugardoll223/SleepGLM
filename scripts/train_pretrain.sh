#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
torchrun --nproc_per_node="${NPROC_PER_NODE:-4}" -m mainmodel.train --config configs/stage2_multimodal_pretrain.yaml "$@"

