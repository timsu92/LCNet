這是一份針對 **LCNet (Lightweight Real-time Image Classification Network)** 的實作指南。根據論文內容，我將其架構、各層細節、模組公式與訓練參數整理如下，協助你重現該模型。

### **1\. 模型總體架構 (Overall Architecture)**

LCNet 採用四階段（4-Stage）的金字塔結構，整體流程如下：

1. **Stem Layer (前處理層):**  
   * 輸入圖片大小：$224 \\times 224$。

   * 操作：$3 \\times 3$ 卷積（stride=2）接 Batch Normalization (BN)。

   * 輸出：特徵圖 $X\_1$。  
2. **Four Stages (四個特徵提取階段):**  
   * 每個階段包含重複堆疊的 **LCNet Block**（雙路徑融合模組）。  
   * 隨著階段加深，特徵圖尺寸減半，通道數增加（具體數量見下表 Variants）。  
3. **Head (分類頭):**  
   * Global Average Pooling (全局平均池化)。  
   * Fully Connected Layer (全連接層)，輸出分類數（如 1000）。

### ---

**2\. 核心模組細節 (LCNet Block)**

每個 Block 的運作流程由公式 (5) 定義，採用雙分支結構處理特徵，最後透過殘差連接輸出。

**Block 內部的數據流：**

1. **輸入變換：** 輸入特徵 $X$ 先經過 $1 \\times 1$ 卷積與 Depthwise Conv (DWConv)。  
2. **雙路徑分流 (Dual-Path):** 特徵圖分為兩路，分別進入 **DTConv** 與 **MDAM**。  
3. **Star Operation 融合：** 將兩路輸出的特徵進行高維融合。  
4. **輸出變換：** 融合後的特徵經過 $1 \\times 1$ 卷積與 $3 \\times 3$ DWConv。  
5. **殘差連接：** $Y \= Output \+ Input$。

#### **A. 分支一：動態閾值卷積 (DTConv)**

負責提取**局部 (Local)** 特徵，特點是動態選擇卷積核。

* **輸入分割：** 將輸入通道 $C\_{in}$ 均分為 3 份：$X\_1, X\_2, X\_3$。

* **多尺度特徵提取 (MSMod1):**  
  * 分別對三份特徵進行 $3 \\times 3$、$5 \\times 5$、$7 \\times 7$ 的卷積操作。

  * 公式：$F\_i \= \\sigma(Conv\_{k\_i}(W\_i X\_i \+ b\_i))$，其中 $\\sigma$ 為激活函數。  
  * 將結果 Concat 合併並做 BN 處理，得到 $X\_{multi}$。

* **動態權重生成 (Dynamic Weight):**  
  * 對 $X\_{multi}$ 進行 Flatten 和 Softmax，生成權重向量 $W$。

  * **Top-k 選擇：** 將權重排序，僅保留前 $k$ 個最大的權重及其對應的卷積核。

* 動態聚合： 使用選出的 $k$ 個卷積核對輸入進行加權卷積：

  $$G \= X \* \\sum\_{i=1}^{k} (W\_{top\\\_k}^i \\times H\_i)$$  
  其中 $H\_i$ 代表第 $i$ 個卷積核。

#### **B. 分支二：多路徑動態注意力機制 (MDAM)**

負責提取**全局 (Global)** 特徵，特點是篩選關鍵 Token 以減少計算量。

* **多尺度 QKV 生成 (MSMod2):**  
  * 使用 $1 \\times 1$、$3 \\times 3$、$5 \\times 5$ 卷積提取特徵並 Concat。

  * 透過線性層 (Linear) 生成 Query ($Q$)、Key ($K$)、Value ($V$)。

* **動態建模 (Dynamic Modeling):**  
  * 計算 Attention Matrix：$a\_t \= \\text{softmax}(\\frac{Q\_t K\_t^T}{\\sqrt{d}})$。

  * **Token 篩選：** 計算每個 Key Token 的總注意力值 $A\_s \= \\sum A\[i,j\]$，並排序。

  * 保留注意力值最高的 $k$ 個 Token，其餘捨棄（視為背景或冗餘資訊）。

* **注意力修正 (Attention Correction):**  
  * 對保留的 Token 乘上修正係數 $\\alpha$ (範圍 0\~1)，未被選中的設為 0。

  * 輸出：$Output \= \\text{Concat}(A'\_t \\cdot V\_t) \\cdot W$。

  * 註：Embedding size 設為 64。

#### **C. 特徵融合：星型運算 (Star Operation)**

將 DTConv 和 MDAM 的輸出在隱式高維空間進行融合，不增加額外參數量。

* 運算邏輯類似於元素級乘法（Element-wise Multiplication），但在高維空間中表示為：

  $$w\_1^T(x) \* w\_2^T(x) \= \\sum \\sum w\_1^i w\_2^j x^i x^j$$

* 實作上通常是將兩個分支的輸出特徵圖進行**元素相乘**。

### ---

**3\. 模型配置與參數 (Model Variants)**

根據論文 Table 1，LCNet 有三種版本（Tiny, Small, Base），區別在於通道數 (C) 與堆疊層數 (L)。

| 階段 (Stage) | 解析度 | 模組 | LCNet-T (Tiny) | LCNet-S (Small) | LCNet-B (Base) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Stem** | $224^2$ | Conv $3\\times3$ | **24** filters | **32** filters | **32** filters |
| **Stage 1** | $56^2$ | Block | **32** ch (×2 layers) | **48** ch (×2 layers) | **64** ch (×4 layers) |
| **Stage 2** | $28^2$ | Block | **64** ch (×4 layers) | **96** ch (×4 layers) | **128** ch (×6 layers) |
| **Stage 3** | $14^2$ | Block | **128** ch (×8 layers) | **192** ch (×12 layers) | **256** ch (×16 layers) |
| **Stage 4** | $7^2$ | Block | **192** ch (×2 layers) | **256** ch (×2 layers) | **256** ch (×4 layers) |
| **FLOPs** |  |  | 1.28 G | 3.39 G | 4.92 G |
| **Params** |  |  | 19.48 M | 40.21 M | 75.42 M |

*(注意：表中 ch 代表通道數，×N 代表該 Block 重複堆疊次數)*

### ---

**4\. 訓練超參數 (Training Hyperparameters)**

為了復現論文結果（如 LCNet-T 在 ImageNet 上的表現），請使用以下設定：

* **框架:** PyTorch  
* **優化器 (Optimizer):** AdamW  
* **初始學習率 (Learning Rate):** 0.001  
* **權重衰減 (Weight Decay):** 0.001  
* **學習率策略 (Scheduler):** Cosine Annealing (餘弦退火)  
* **Embedding Size (MDAM):** 64

### **5\. 實作建議**

1. **Block 堆疊：** 每個 Stage 開始時通常會進行下採樣（Downsampling，通常透過 Stride=2 的卷積或 Patch Merging 實現），隨後才是上述的 LCNet Block 堆疊。  
2. **Top-k 實作：** 在 DTConv 和 MDAM 中，Top-k 選擇是不可導的操作，但在深度學習框架中，通常只對權重值進行梯度傳播，索引選擇過程本身不需要梯度。可以使用 torch.topk 來獲取索引。  
3. **Star Operation：** 若要達到論文所述「無額外計算開銷」且高效，建議實作為兩個分支輸出的 Element-wise Product ($Branch1 \\times Branch2$)。

這份整理涵蓋了從輸入到輸出、層數配置以及訓練所需的所有關鍵資訊。