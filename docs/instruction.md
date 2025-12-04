## 概要
`LCNet - Lightweight real-time image classification network based on efficient multipath dynamic attention mechanism and dynamic threshold convolution.pdf`是一個論文的檔案，而我想要實作他的內容。

## 現有檔案

現在這個專案基本上可以說是空的，只有少數的檔案。
`src/utils/config.py`是一個之前對DDPM模型的設定檔，在現在的情況下記得要大改。

## 實作規定

- 我喜歡使用argparse輸入各項超參數，例如模型的超參數、訓練的圖片位置、checkpoint輸出資料夾
- 我想要用cifar10來訓練與驗證模型狀態

## 詳細初步想法

輸出檔位置
- checkpoint
  - 我想要放在`out/models/<流水號>`，例如`out/models/1`
  - 除了每訓練幾個epoch就有的model checkpoint，我希望有個yaml檔在處理完所有的訓練參數後，把他們存起來
  - 我想要放在`out/eval/proc/<模型id>-<使用該模型推論的流水號id>`，例如`out/eval/proc/1-1`

訓練流程相關
- 處理完參數、但還沒真的開始訓練模型前，我希望可以自動偵測當前系統，依照有個GPU數、有多少VRAM決定batch_size有多大，並且也將這個設定寫數參數的yaml檔
- 每隔幾個epoch算一次模型效能時，我希望可以輸出到csv檔，讓我得知訓練集、驗證集的分數
- 我希望可以有進度條得知訓練的進度