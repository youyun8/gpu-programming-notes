---
title: 權重反量化
platform: LeetGPU
upstream: medium/64_weight_dequantization
url: https://leetgpu.com/challenges/weight-dequantization
difficulty: medium
tags: [elementwise, quantization, block-scaling]
status: solved
---

# 權重反量化

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/weight-dequantization)

## 題意

對一個 $M \times N$ 權重矩陣進行反量化，其縮放係數依每個
$T \times T$ **圖塊**儲存（$M, N \le 8192$、$T \in \{16, 32, 64, 128\}$；
基準測試為 $M = N = 8192$、$T = 128$；容許誤差 `1e-5`）。
分塊縮放是現代低精度格式（DeepSeek-V3 的 FP8 權重、MX 格式）
將量化誤差限制在局部範圍的方法。此核心是在 GEMM 前執行，
或融合至 GEMM 的「解包」步驟。

## 圖解

![逐分塊反量化：X 中每個 T × T 分塊共用 S 中的一個縮放係數](figure.svg)

顏色把左側 X 的每個分塊對應到右側縮放矩陣 S 中的一格。每個元素乘上其所在分塊的縮放係數；邊緣分塊可能不完整。

## 數學表述

$$
Y_{ij} = X_{ij}\cdot S_{\lfloor i/T\rfloor,\ \lfloor j/T\rfloor}, \qquad S \in \mathbb R^{\lceil M/T\rceil \times \lceil N/T\rceil}
$$

| 符號 | 意義 |
|---|---|
| $M,\ N$ | 矩陣的列數與欄數 |
| $T$ | 圖塊大小（`TILE_SIZE`） |
| $X_{ij}$ | 量化值（此處以 float32 提供） |
| $S_{rc}$ | 圖塊 $(r, c)$ 的縮放係數；採列優先，且有 $\lceil N/T\rceil$ 欄 |
| $Y_{ij}$ | 反量化值 |
| $\lfloor i/T\rfloor$ | 元素列 $i$ 所屬的圖塊列（邊緣圖塊可能不完整） |

## 解題思路

- 使用 $64 \times 4$ 執行緒區塊的二維網格。`threadIdx.x` 沿欄方向移動，
  因此 $X$ 的載入與 $Y$ 的寫入都是合併存取的 256 位元組列。
- 每個執行緒以兩次整數除法計算其圖塊座標，再乘上縮放係數。
- **縮放係數重複使用。** 當 $T = 128$ 時，同一區塊列的 64 個執行緒
  會共用一或兩個縮放係數，而整體而言每個縮放係數由
  $T^2 = 16\,384$ 個元素共用。縮放矩陣（$64 \times 64$ 個 float = 16 KB）
  會一直留在 L1/L2 中。

## 成本分析

$$
Q \approx 8MN + 4\left\lceil\frac{M}{T}\right\rceil\left\lceil\frac{N}{T}\right\rceil \ \text{bytes}, \qquad T_{\min} \approx \frac{8MN}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取 $X$ 並寫入 $Y$（各 4 位元組）；縮放係數可忽略 |
| $\beta$ | DRAM 頻寬 |

基準測試為 537 MB，在 2 TB/s 下約需 270 µs。在實際 int8/fp8 管線中，
量化後的 $X$ 每個元素只有 1 位元組，而且乘法會在 GEMM 的暫存器檔案中完成，
完全不需寫出 $Y$。

## 常見陷阱

- **不完整的邊緣圖塊。** 必須使用 $\lceil N/T\rceil$ 個縮放係數欄，
  而不是 $N/T$，作為 $S$ 的列步幅。
- **整數除法成本。** 會被記憶體延遲掩蓋。當 $T$ 是 2 的冪時可改用位移。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆通過，
包括全部四種圖塊大小與非整倍數維度。

## 延伸閱讀

- [INT8 量化矩陣乘法](../032-int8-quantized-matmul/)、[INT4 矩陣乘法](../081-int4-matmul/)。
- Tensara [MXFP8 反量化](../../tensara/mxfp8-dequantize/)、[NVFP4 反量化](../../tensara/nvfp4-dequantize/)。
