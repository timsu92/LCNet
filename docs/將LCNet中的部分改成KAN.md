這是一份根據論文《Kolmogorov-Arnold Transformer (KAT)》內容整理的 Markdown 文件。重點在於如何將 Transformer 中的 MLP 替換為 KAN/GR-KAN，以及相關的實作細節。

# ---

**Kolmogorov-Arnold Transformer (KAT): 架構與實作細節**

## **1\. 核心概念：將 Transformer MLP 替換為 KAN**

傳統 Transformer 依賴多層感知機 (MLP) 進行通道間的資訊混合。KAT 提出將 MLP 層替換為 Kolmogorov-Arnold Network (KAN) 層，以提升模型的表達能力。

### **替換架構 (Architecture Replacement)**

在標準 Vision Transformer (ViT) 中，第 $l$ 層的操作如下：

$$x\_{l} \= MLP(LN(x'\_{l})) \+ x'\_{l}$$

其中 $x'\_{l}$ 是經過 Attention 層後的輸出。  
KAT 的修改方式：  
將上述公式中的 MLP 直接替換為 KAN (具體為 GR-KAN)：

$$x\_{l} \= KAN(LN(x'\_{l})) \+ x'\_{l}$$

* **層數對應：** 標準 Transformer 的 MLP 通常包含兩層 (Two-layer MLP)。KAT 同樣將其替換為 **兩層 KAN (Two-layer KANs)**。

* **權重轉移 (Weight Transfer)：** 這種設計允許 KAT 載入預訓練的 ViT 權重並繼續訓練。

## ---

**2\. 為什麼原始 KAN 無法擴展？ (Why Vanilla KAN Fails)**

直接將 MLP 換成原始 KAN (ViT+KAN) 在 ImageNet 等大規模任務上表現不佳且極慢，主要原因如下：

1. **B-Spline 對 GPU 不友善：** B-spline 需要遞迴計算且不易平行化，導致推論速度慢。

2. **參數與計算效率低：** KAN 需要為每一個「輸入-輸出對」建立一個獨立的函數。當網路寬度增加時，參數呈爆炸式增長。

3. **初始化問題：** 原始 KAN 的初始化無法保持變異數 (Variance-preserving)，導致深層網路難以收斂。

## ---

**3\. GR-KAN (Group-Rational KAN) 實作細節**

為了決上述問題，論文提出了 **GR-KAN**。以下是實作 GR-KAN 的三個關鍵技術細節。

### **A. 基底函數：有理函數 (Rational Base Functions)**

KAT 放棄了 B-spline，改用**有理函數 (Rational Functions)** 作為基底函數 $\\phi(x)$。

* 公式： 使用 Padé Activation Unit (PAU) 的形式：

  $$F(x) \= \\frac{P(x)}{Q(x)} \= \\frac{a\_{0} \+ a\_{1}x \+ \\cdots \+ a\_{m}x^{m}}{1 \+ |b\_{1}x \+ \\cdots \+ b\_{n}x^{n}|}$$  
  9

* **超參數 (Hyperparameters)：** 預設使用 $m=5$ (分子階數) 和 $n=4$ (分母階數)。

* **優勢：** 計算僅涉及簡單的加法與乘法，極度適合 GPU 平行運算。

### **B. 群組參數共享 (Group KAN Strategy)**

為了減少參數，KAT 提出了分組共享策略。

* **原理：** 將輸入通道 $d\_{in}$ 分為 $g$ 個群組 (Groups)。  
* **共享機制：**  
  * **函數形狀共享：** 同一個群組內的邊 (Edges) 共享有理函數的係數 ($a\_m, b\_n$)。

  * **權重獨立：** 每一條邊仍然保留一個獨立的純量縮放因子 $w$ (Scalar scaling factor)。

* 實作轉換 (Implementation as Linear Layer)：  
  數學上，GR-KAN 層可以被實作為「群組有理函數」後接一個「線性層」：

  $$GR-KAN(x) \= Linear(Group\\\_Rational(x))$$  
  這讓 GR-KAN 看起來像是一個帶有**可學習激活函數 (Learnable Activation)** 的特殊 MLP。

### **C. 變異數保持初始化 (Variance-Preserving Initialization)**

為了確保訓練穩定性，初始化過程分為三步：

1. **擬合目標函數：** 首先初始化 $a, b$ 係數，使有理函數 $F(x)$ 近似於常見的激活函數 (如 ReLU, GELU, Swish, 或 Identity)。  
   * *KAT 設定：* 第一個 KAN 層初始化為 Identity，第二個初始化為 Swish。

2. **計算增益 (Gain)：** 計算 $\\alpha \= \\frac{\\mathbb{E}\[F(x)^2\]}{Var\[x\]}$。  
3. **初始化權重 $w$：** 根據增益 $\\alpha$ 初始化縮放因子 $w$，採樣自 $\\mathcal{N}(0, \\frac{\\alpha}{d\_{in}})$。

## ---

**4\. 程式碼層面的實作與優化 (CUDA & Optimization)**

### **結構對應：ViT MLP vs. KAT GR-KAN**

若要將 ViT 的 MLP 區塊轉換為 KAT，其層級對應如下圖所示 18：

| ViT MLP Block | KAT GR-KAN Block | 說明 |
| :---- | :---- | :---- |
| (Input) | (Input) |  |
| **Linear 1 (FC1)** | **Group Rational 1** | 第一層 KAN 的激活部分 |
| **Activation (GELU)** | **Linear 1** | 第一層 KAN 的線性混合部分 |
| **Linear 2 (FC2)** | **Group Rational 2** | 第二層 KAN 的激活部分 |
| (Output) | **Linear 2** | 第二層 KAN 的線性混合部分 |

注意：雖然 GR-KAN 數學上是 $W \\cdot F(x)$，但在利用預訓練模型時，ViT 的 Linear 層權重可以直接載入到 GR-KAN 的 Linear 部分，而 Group Rational 層則負責模擬原本的激活函數或 Identity。

### **計算優化 (CUDA Implementation)**

為了進一步加速，論文使用了 CUDA 核心來實作有理函數：

* Horner's Method (霍納法)： 用於評估多項式，將運算簡化為巢狀形式：

  $$a\_0 \+ x(a\_1 \+ x(a\_2 \+ \\dots))$$  
  這將 $n$ 次多項式的計算減少到僅需 $n$ 次乘法和 $n$ 次加法 20。

* **效能差異：** 使用 Horner 法的有理函數比 B-spline 快約 9.3 倍 21。自定義的 CUDA 實作比 PyTorch 的 Loop 或 Vectorized 方式快得多且記憶體佔用更低 22。

## **5\. 總結比較表 (Comparison)**

| 特性 | MLP | Vanilla KAN | GR-KAN (KAT) |
| :---- | :---- | :---- | :---- |
| **基底函數** | 固定 (ReLU/GELU) | B-Spline | **Rational Function** (有理函數) |
| **參數數量** | $O(d\_{in} \\times d\_{out})$ | $O(d\_{in} \\times d\_{out} \\times G)$ |  **$O(d\_{in} \\times d\_{out})$** (與 MLP 相當) 23  |
| **計算方式** | Linear \-\> Activation | Activation for each edge \-\> Sum |  **Group Activation \-\> Linear** 24  |
| **GPU 友善度** | 高 | 低 (遞迴計算) | **高** (CUDA 優化) |
| **初始化** | 簡單 (Xavier/Kaiming) | 困難 (需特殊調整) | **Variance-Preserving** (分步初始化) |

