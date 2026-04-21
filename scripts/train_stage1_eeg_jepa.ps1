$env:OMP_NUM_THREADS = "1"
torchrun --nproc_per_node=4 -m mainmodel.train --config configs/stage1_eeg_jepa.yaml

