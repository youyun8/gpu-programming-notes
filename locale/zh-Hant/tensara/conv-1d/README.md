---
title: 一維卷積
platform: Tensara
upstream: conv-1d
url: https://tensara.org/problems/conv-1d
difficulty: easy
tags: [convolution, shared-memory, tiling]
status: solved
---

# 一維卷積

**平台：** Tensara · **難度：** easy · [題目敘述](https://tensara.org/problems/conv-1d)

## 問題

對長度為 $N$ 的 float32 訊號，以具有 $K$ 個取樣點的奇數核心執行「same」一維卷積，兩側以零填補 $r = (K-1)/2$。和 PyTorch 的 `conv1d` 一樣，這其實是互相關：核心**不會**翻轉。測試在 $N$ = 32 K … 512 K 上使用極大的核心（$K = 8191$），因此工作量為 $N K$ 次乘加，而非單純的記憶體串流。檢查誤差為
`rtol = 2e-4`、`atol = 5e-3`。

## 公式

$$
C[i] = \sum_{j=0}^{K-1} \tilde{A}[\,i + j - r\,]\; B[j], \qquad
\tilde{A}[t] = \begin{cases} A[t], & 0 \le t < N \\ 0, & \text{otherwise} \end{cases}, \qquad
r = \frac{K-1}{2}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 長度為 $N$ 的輸入訊號 |
| $\tilde{A}$ | 將 $A$ 的邊界外延伸為零（填補） |
| $B$ | 具有 $K$ 個取樣點的核心（濾波器），$K$ 為奇數 |
| $r$ | 核心半徑；核心以 $i$ 為中心 |
| $C$ | 長度為 $N$ 的輸出 |
| $i$ | 輸出索引；$j$ 為取樣點索引 |

輸出 $i$ 會存取輸入視窗 $[\,i - r,\ i + r\,]$。因此，產生輸出
$[b, b + T)$ 的執行緒區塊需要下列視窗：

$$
\bigl[\,b - r,\ b + T - 1 + r\,\bigr], \qquad \text{length } T + K - 1
$$

| 符號 | 意義 |
|---|---|
| $b$ | 區塊的第一個輸出 |
| $T$ | 每個區塊的輸出數（此處為 1024） |

## 方法

1. **將輸出圖塊化**：由 256 個執行緒組成的區塊負責 $T = 1024$ 個輸出，每個執行緒處理 4 個，並以 256 為跨距，因此在任一時刻，一個 warp 的 32 條通道會讀取 32 個連續的共享記憶體字組（無 bank 衝突）。
2. **將取樣點分段**：當 $K = 8191$ 時，完整視窗（$1024 + 8190$ 個浮點數）不容易放入共享記憶體，因此每次處理 2048 個取樣點。每一段中，區塊會載入 2048 個取樣點及對應的
   $1024 + 2047$ 個輸入（$[0, N)$ 以外補零），同步後再從共享記憶體執行 FMA 迴圈。共享記憶體用量為 8 KB + 12 KB，不受 $K$ 影響。
3. **內層迴圈**：每個 $j$ 只讀取一次 $B[j]$ 並廣播，接著以四次共享記憶體載入執行四次 `fmaf`。整體而言，每個輸入元素約從 DRAM 讀取 $\lceil K/2048 \rceil$ 次；相較於 $NK$ 次 FMA，這項成本可忽略。

## 成本分析

$$
W = 2NK\ \text{flops}, \qquad Q \approx 4N\left(2 + \left\lceil \tfrac{K}{2048} \right\rceil\right) + 4K\left\lceil \tfrac{N}{T} \right\rceil\ \text{bytes}, \qquad
I = \frac{W}{Q}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數（一次 FMA = 2 flops） |
| $Q$ | DRAM（或 L2）位元組數：輸入視窗、輸出、每個區塊重新讀取核心的資料量 |
| $I$ | 算術強度，此處為每位元組數千 flops |

當 $N = 524288$、$K = 8191$ 時：$W = 8.6$ GFLOP。核心函式受限於運算效能，但瓶頸是**共享記憶體載入**連接埠，而非 FMA 單元：每次 FMA 都需要一次 `LDS`，而一個 SM 每個週期能發出的共享記憶體載入數少於 FMA 數。下一步可採用暫存器分塊：讓每個執行緒負責數個*相鄰*輸出，並滑動暫存器視窗，使一次共享記憶體載入能供應多次 FMA。

## 常見陷阱

- **這是互相關，不是卷積**：$B[j]$ 乘上 $A[i + j - r]$；翻轉核心是錯誤的。
- **累加誤差**：每個輸出包含 8191 個 fp32 項；誤差容許範圍
  （`atol = 5e-3`）允許與 cuDNN 不同的加總順序。
- **邊界**：靠近起點時 $i + j - r$ 為負數；全域索引應使用有號
  64 位元算術。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 相關內容

- [二維卷積](../conv-2d/)、[三維方形卷積](../conv-square-3d/)、
  LeetGPU [一維卷積](../../leetgpu/009-1d-convolution/)。
