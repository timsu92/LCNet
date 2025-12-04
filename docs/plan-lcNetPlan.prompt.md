# LCNet 實作與訓練系統規劃

## 目標
建立基於 CIFAR-10 的 LCNet 分類模型訓練系統，包含自動資源偵測、動態參數配置與完整日誌記錄。

## 討論與確認事項
1.  **模型架構細節**: 已參照 `model_arch.md` 提供的 LCNet 架構（4-Stage, DTConv, MDAM, Star Operation）。
2.  **評估指標**: 分類任務標準的 **Accuracy (準確率)** 與 **Loss**。
3.  **自動 Batch Size**: 直接使用現有 `src/utils/system_info.py` 中的 `auto_batch_size` 函式，不再重新實作。

## 執行步驟

### 1. 重構配置與參數系統 (`src/utils/config.py`)
- 清除舊有 DDPM 設定，建立新的 `Config` 類別。
- 整合 `argparse` 以支援 CLI 參數（如 `--data_path`, `--output_dir`, `--model_variant`）。
- 實作 `save_config` 功能，將最終參數（含自動偵測結果）存為 YAML。

### 2. 建置資料管線 (`src/data/cifar.py`)
- 實作 CIFAR-10 `Dataset` 與 `DataLoader`。
- 加入標準預處理（Normalize, RandomCrop, HorizontalFlip）。

### 3. 整合資源偵測 (`src/train.py` & `src/utils/system_info.py`)
- 在訓練腳本中引入 `src/utils/system_info.py` 的 `auto_batch_size`。
- 在模型建置前執行偵測，並將結果寫入 Config。

### 4. 搭建模型骨幹 (`src/models/lcnet.py`)
依據 `model_arch.md` 實作：
- **Stem Layer**: 3x3 Conv, BN.
- **LCNet Block**:
    - **DTConv (Dynamic Threshold Conv)**: Multi-scale Conv (3x3, 5x5, 7x7), Dynamic Weight Generation, Top-k selection.
    - **MDAM (Multipath Dynamic Attention)**: Multi-scale QKV, Attention Matrix, Token Selection (Top-k), Attention Correction.
    - **Star Operation**: Element-wise product of two branches.
- **Backbone**: 4-Stage 架構 (Tiny/Small/Base variants)。
- **Head**: GAP + FC。

### 5. 實作訓練與驗證迴圈 (`src/train.py`)
- 實作 Training Loop，包含 `tqdm` 進度條。
- 實作 Validation Loop，計算 Accuracy 與 Loss。
- 整合 CSV Logger，記錄每個 epoch 的效能數據 (Train/Val Loss, Train/Val Acc)。
- 實作 Optimizer (AdamW) 與 Scheduler (Cosine Annealing)。

### 6. 實作 Checkpoint 與輸出管理
- 實作模型權重儲存機制 (`out/models/<id>/epoch_X.pt`)。
- 實作推論結果輸出路徑管理 (`out/eval/proc/<model>-<run>`)。

### 7. 實作推論腳本 (`src/inference.py`)
- 載入訓練好的模型與參數 (`out/models/<id>/config.yaml` & `checkpoint`).
- 實作推論輸出路徑管理：`out/eval/proc/<model_id>-<run_id>` (自動遞增 run_id)。
- 執行推論：針對 CIFAR-10 測試集或指定圖片。
- 輸出結果：儲存預測結果 (CSV/JSON) 與混淆矩陣 (Confusion Matrix)。

### 8. 實作視覺化工具 (`src/visualize.py`)
- **訓練監控**: 讀取 `metrics.csv` 繪製 Loss 與 Accuracy 曲線圖。
- **預測結果**: 顯示圖片並標註 預測類別/真實類別 (Top-k)。
- (選用) **特徵視覺化**: 嘗試視覺化 MDAM 的 Attention Map 或 DTConv 的權重選擇。
