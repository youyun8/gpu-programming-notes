---
title: INT8 量化矩陣乘法
platform: LeetGPU
upstream: medium/32_int8_quantized_matmul
url: https://leetgpu.com/challenges/int8-quantized-matmul
difficulty: medium
tags: [gemm, int8, quantization, tensor-cores, wmma]
status: solved
---

# INT8 量化矩陣乘法

**平台：** LeetGPU · **難度：** medium · [題目敘述](https://leetgpu.com/challenges/int8-quantized-matmul)

## 題意

計算量化矩陣乘法。$A$（$M\times K$）與 $B$（$K \times N$）是 int8，
縮放係數為 $s_A, s_B$，零點為 $z_A, z_B$。int8 輸出 $C$
的縮放係數為 $s_C$，零點為 $z_C$
（$1 \le M, N, K \le 4096$；基準測試為 $M = 8192$、$N = 4096$、
$K = 2048$）。檢查要求**逐位元完全相同**。int8 推論就是這樣運作：
先執行整數張量核心矩陣乘法，再於 float「再量化」收尾階段完成轉換。

## 圖解

![INT8 GEMM：以整數精確累加，再以浮點 epilogue 重新量化](figure.svg)

分塊流程與一般 GEMM 相同，只是改用 int8 輸入、int32 累加。右側方框是 epilogue：依照參考實作的運算順序，把精確的整數和轉回 int8。

## 數學表述

$$
C_{ij} = \operatorname{clamp}\!\Bigl(\operatorname{rne}\bigl(\operatorname{fl}\bigl(\operatorname{fl}(\operatorname{fl}(S_{ij}\, s_A)\, s_B) / s_C\bigr)\bigr) + z_C,\ -128,\ 127\Bigr),
\qquad S_{ij} = \sum_{k=0}^{K-1} (A_{ik} - z_A)(B_{kj} - z_B)
$$

| 符號 | 意義 |
|---|---|
| $M,\ N,\ K$ | 輸出列數、輸出欄數、內部維度 |
| $A_{ik},\ B_{kj}$ | int8 輸入值 |
| $z_A,\ z_B,\ z_C$ | 零點（$[-128, 127]$ 內的整數） |
| $s_A,\ s_B,\ s_C$ | 正的 float32 縮放係數 |
| $S_{ij}$ | 對減去零點後的輸入執行精確整數點積（int32） |
| $\operatorname{fl}(\cdot)$ | 單次 float32 運算，捨入至最接近值；順序 $((S\,s_A)\,s_B)/s_C$ 與參考實作相同 |
| rne | 捨入至最接近值，平手時取偶數（`torch.round`、`rintf`） |
| Clamp | 飽和限制於 int8 範圍 |

### 展開零點

$A_{ik} - z_A$ 的範圍為 $[-255, 255]$，已無法放入 int8，
因此平移後的值不能直接交給 int8 張量核心。改為展開乘積：

$$
S_{ij} = \underbrace{\sum_k A_{ik}B_{kj}}_{P_{ij}\ (\text{int8 MMA})} - z_B \underbrace{\sum_k A_{ik}}_{r_i} - z_A \underbrace{\sum_k B_{kj}}_{c_j} + K z_A z_B
$$

| 符號 | 意義 |
|---|---|
| $P_{ij}$ | 原始 int8 × int8 乘積，在張量核心上以 int32 累加 |
| $r_i$ | $A$ 的列總和（每列一個 int） |
| $c_j$ | $B$ 的欄總和（每欄一個 int） |
| $K z_A z_B$ | 常數修正項 |

所有項目都以 int32 精確運算。由於
$\lvert P\rvert \le 4096 \cdot 128^2 \approx 6.7\times10^7 < 2^{31}$，
不會發生溢位。

## 解題思路

1. **`rowSums`**（每列一個 warp，使用 shuffle 歸約）與
   **`colSums`**（每欄一個執行緒，各欄之間合併存取）
   分別計算 $r$ 與 $c$。
2. **`imma`**，主要 GEMM：
   - 一個含 4 個 warp 的區塊計算 $C$ 的 64 × 64 分塊。每個 warp
     計算 32 × 32，也就是 2 × 2 個 WMMA `16x16x16` 片段，
     資料型別為 `signed char × signed char → int`。
   - 以 32 為切片大小走訪 K。int8 分塊會以**連續的 16 × 16 區塊**
     暫存在共享記憶體
     （`a_s[4][2][16][16]`、`b_s[2][4][16][16]`、`ldm = 16`）。
   - 累加器儲存至共享 int32 分塊。收尾階段計算
     $S_{ij} = P_{ij} - z_B r_i - z_A c_j + K z_A z_B$，
     接著依參考實作*完全相同*的 float32 運算順序與 `rintf` 再量化，
     加上 $z_C$、套用範圍限制，最後儲存為 int8。

### 為何使用連續的 16 × 16 區塊？

WMMA 要求片段指標以 32 位元組對齊。對每元素 1 位元組的資料，
列優先分塊 `a_s[64][32+pad]` 會讓某列的第二個片段從第 16 欄開始，
也就是偏移 16 位元組，並未對齊。將每個 16 × 16 片段各自存為
連續的 256 位元組區塊，可讓每個片段都從 256 位元組邊界開始。
[cuemu](../../tools/cuemu/README.md) 模擬器的對齊檢查抓出了原始
填補配置的問題。

## 成本分析

$$
W = 2MNK \ \text{int ops}, \qquad Q \approx MK\frac{N}{64} + KN\frac{M}{64} + MN \ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 整數乘加數 × 2 |
| $Q$ | 使用 64 × 64 分塊時的位元組數（每個 int8 元素 1 位元組） |

基準測試中，$W \approx 1.4\times10^{11}$ 次運算。以 A100 約
600 TOPS 的 int8 張量吞吐量計算，運算需 0.23 ms。相較於 fp16，
int8 將位元組數減半，並讓張量核心速率加倍，這正是量化推論快速的原因。
採用同步暫存的 64 × 64 分塊只能達到峰值的一部分。

## 常見陷阱

- **捨入模式。** `roundf` 會在平手時向遠離零的方向捨入，
  `torch.round` 則在平手時取偶數。`rintf` 才符合參考實作。
- **運算順序。** `S * (sA*sB/sC)` 在數學上相等，但 float32
  捨入結果不同，無法通過精確檢查。
- **前置處理的競爭。** 列總和與欄總和是同一串流上的獨立核心函式，
  因此會在 GEMM 讀取前完成。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md)
通過逐位元精確檢查。模擬器會模擬 int8 WMMA 及其對齊規則。
測試涵蓋非 16 倍數的形狀與極端零點。

## 延伸閱讀

- [GEMM（fp16）](../022-gemm/)、
  [INT4 矩陣乘法](../081-int4-matmul/)、
  [權重反量化](../064-weight-dequantization/)。
- Tensara [MXFP8 GEMM](../../tensara/mxfp8-gemm/)、
  [NVFP4 GEMM](../../tensara/nvfp4-gemm/)。
