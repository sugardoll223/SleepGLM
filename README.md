# 鐫＄湢鍩哄骇妯″瀷锛圫leepGLM锛?
## 椤圭洰鐩爣

鏈粨搴撴彁渚涗竴涓彲鎵╁睍鐨勭潯鐪犲垎鏈熻缁冩鏋讹紝婊¤冻浠ヤ笅鏍稿績瑕佹眰锛?
1. 鏁版嵁浣跨敤 `.h5` 鏍煎紡杩涜瀛樺偍鍜岃缁冦€?2. 鏀寔澶氭ā鎬佽緭鍏ワ細
   - EEG: 100Hz
   - EOG: 100Hz
   - EMG: 100Hz
   - ECG: 100Hz
   - airflow: 10Hz
   - thoracoabdominal belt: 10Hz
   - SPO2: 10Hz
   - PPG: 10Hz
3. 鏀寔棰勮缁冩暟鎹泦锛欻SP, SHHS, MrOS, MESA, PhysioNet2018銆?4. 鏀寔寰皟鏁版嵁闆嗭細WSC, CAP, Sleep-EDF, ISRUC銆?5. 璁粌妗嗘灦浣跨敤 PyTorch DDP锛屽苟棰勭暀寰皟涓庢秷铻嶅疄楠屽紑鍏炽€?6. 闈㈠悜璺ㄨ澶?璺ㄦ暟鎹泦閫氶亾宸紓锛屾敮鎸佲€滄寜閫氶亾鍚嶆槧灏勫埌缁熶竴閫氶亾妯℃澘鈥濄€?
## 宸叉惌寤烘鏋惰兘鍔?
1. `torchrun` 鍚姩鐨?PyTorch DDP 璁粌娴佺▼銆?2. 閫氱敤 H5 鏁版嵁璇诲彇鍣紝鍏煎甯歌涓ょ缁勭粐鏍煎紡锛堟暟缁勫紡 / sample-group 寮忥級銆?3. 澶氭ā鎬佸熀搴фā鍨嬮鏋讹紙妯℃€佺紪鐮佸櫒 + Transformer 铻嶅悎 + 鍒嗙被澶达級銆?4. 寰皟鎺ュ彛锛?   - 鍔犺浇棰勮缁冩潈閲?   - 鍐荤粨 backbone 鑻ュ共 epoch
   - 鏀寔鏂偣缁
5. 娑堣瀺瀹為獙鎺ュ彛锛堥厤缃嵆鐢熸晥锛夛細
   - 寮哄埗灞忚斀鎸囧畾妯℃€?   - 闅忔満妯℃€佷涪寮?   - 闅忔満閫氶亾涓㈠純
   - 鍏抽棴鍙傝€冪數鏋?embedding
6. 閫氶亾宸紓閫傞厤鎺ュ彛锛堥厤缃嵆鐢熸晥锛夛細
   - 璇诲彇 `channel_names` 骞跺仛 name-based channel mapping
   - 鏀寔鐢垫瀬鍛藉悕鍒悕锛堜緥濡?`F3-M2/F3-A2/F3`锛?   - 鏃犻€氶亾鍚嶆椂鍥為€€鍒?index-based pad/truncate

## 鐩綍缁撴瀯

```text
mainmodel/
鈹溾攢鈹€ configs/
鈹?  鈹溾攢鈹€ base.yaml
鈹?  鈹溾攢鈹€ pretrain.yaml
鈹?  鈹溾攢鈹€ finetune.yaml
鈹?  鈹溾攢鈹€ stage1_eeg_jepa.yaml
鈹?  鈹溾攢鈹€ stage2_multimodal_pretrain.yaml
鈹?  鈹斺攢鈹€ stage3_downstream_finetune.yaml
鈹溾攢鈹€ scripts/
鈹?  鈹溾攢鈹€ train_pretrain.sh
鈹?  鈹溾攢鈹€ train_finetune.sh
鈹?  鈹溾攢鈹€ train_pretrain.ps1
鈹?  鈹斺攢鈹€ train_finetune.ps1
鈹溾攢鈹€ mainmodel/
鈹?  鈹溾攢鈹€ train.py
鈹?  鈹溾攢鈹€ data/
鈹?  鈹溾攢鈹€ models/
鈹?  鈹溾攢鈹€ engine/
鈹?  鈹斺攢鈹€ utils/
鈹斺攢鈹€ requirements.txt
```

> 浣犳暣鐞嗙殑鏁版嵁闆嗛€氶亾宸紓鍙弬鑰冧粨搴撳唴 `鏁版嵁闆嗙粺璁?md`锛屽缓璁悓姝ョ淮鎶ら厤缃噷鐨?`data.channel_adapter.modality_channel_schema`銆?
## H5 鏁版嵁鏍煎紡璇存槑锛堟敮鎸佷袱绉嶏級

### 1) 鏁扮粍寮忥紙鎺ㄨ崘锛?
```text
/eeg            [N, C, T]
/eog            [N, C, T]
/emg            [N, C, T]
/ecg            [N, C, T]
/airflow        [N, C, T]
/thoracoabdominal [N, C, T]
/spo2           [N, C, T]
/ppg            [N, C, T]
/label          [N]
/split          [N]  # 鍙€? train/val/test
/reference_id   [N]  # 鍙€?/dataset_id     [N]  # 鍙€?```

### 2) sample-group 寮?
```text
/samples/<sample_id>/eeg            [C, T]
/samples/<sample_id>/eog            [C, T]
/samples/<sample_id>/...
/samples/<sample_id>/label          scalar
/samples/<sample_id>/split          scalar (鍙€?
/samples/<sample_id>/reference_id   scalar (鍙€?
/samples/<sample_id>/dataset_id     scalar (鍙€?
```

## 瀹夎渚濊禆

```bash
pip install -r requirements.txt
```

## 璁粌鍛戒护

### 棰勮缁?
```bash
bash scripts/train_pretrain.sh
```

### 寰皟

```bash
bash scripts/train_finetune.sh
```

### Stage1 纭綋绀轰緥

```bash
torchrun --nproc_per_node=4 -m mainmodel.train --config configs/stage1_eeg_jepa.yaml
```

### Linux 鏈嶅姟鍣ㄩ娆℃墽琛岋紙鍙€夛級

```bash
chmod +x scripts/train_pretrain.sh scripts/train_finetune.sh
```

## 甯哥敤鍙傛暟瑕嗙洊锛堝懡浠よ锛?
```bash
torchrun --nproc_per_node=4 -m mainmodel.train \
  --config configs/stage3_downstream_finetune.yaml \
  --override data.train_files="[\"/path/train_1.h5\",\"/path/train_2.h5\"]" \
  --override data.val_files="[\"/path/val.h5\"]" \
  --override finetune.pretrained_checkpoint="/path/pretrain_best.pt" \
  --override training.epochs=30 \
  --override training.view_dropout.random_modality_drop_prob=0.2
```

## 璇存槑

1. 褰撳墠妯″瀷鏄€滃彲鎵╁睍鍩哄骇妗嗘灦鈥濓紝閲嶇偣鍦ㄨ缁冩祦绋嬩笌鎺ュ彛瀹屾暣鎬с€?2. 鍚庣画鍙洿鎺ユ浛鎹?`mainmodel/models/model.py` 涓紪鐮佸櫒涓庤瀺鍚堢粨鏋勶紝涓嶅奖鍝?DDP 涓绘祦绋嬨€?3. 鑻ヤ綘鐨?H5 瀛楁鍚嶄笉鍚岋紝鍙湪 `configs/base.yaml` 鐨?`data.modality_keys`銆乣*_key_candidates`銆乣modality_channel_name_keys` 涓皟鏁淬€?4. 璺ㄦ暟鎹泦閫氶亾瀵归綈涓昏閫氳繃 `data.channel_adapter.modality_channel_schema` 瀹屾垚锛屽缓璁寜浣犵殑鏁版嵁瀹為檯鐢垫瀬鍛藉悕鎸佺画琛ュ叏 alias銆?

