---
title: 多頭交叉注意力
platform: LeetGPU
upstream: hard/26_multi_head_cross_attention
url: https://leetgpu.com/challenges/multi-head-cross-attention
difficulty: hard
tags: [attention, cross-attention, flash-attention, multi-head]
status: solved
---

# 多頭交叉注意力

**平台：** LeetGPU · **難度：** hard · [題目敘述](https://leetgpu.com/challenges/multi-head-cross-attention)

## 題意

計算編碼器—解碼器 Transformer（T5、Whisper、Stable Diffusion 的
文字條件）所使用的多頭**交叉**注意力。解碼器查詢 $Q$ 的形狀為
$(M, H, D)$，編碼器鍵/值 $K, V$ 的形狀為 $(N, H, D)$。輸出形狀為
$(M, H, D)$（$M, N \le 4096$、$H \le 64$、$D \le 256$；
基準測試為 $M = 1024$、$N = 2048$、$H = 16$、$D = 128$；
容許誤差為 `1e-4`）。這裡沒有遮罩，而且通常 $M \ne N$。

## 圖解

![交叉注意力：M 個解碼器查詢注意全部 N 個編碼器鍵，沒有遮罩](figure.svg)

分數矩陣是長方形（M ≠ N）且全部可見。第 2 列（綠）會混合全部 N 個 value 向量；每個 head 各自獨立重複此計算。

## 數學表述

對每個頭 $h$：

$$
O_{i,h,:} = \sum_{j=0}^{N-1} \frac{e^{s_{ij}^{(h)} - m_i^{(h)}}}{\sum_{j'} e^{s_{ij'}^{(h)} - m_i^{(h)}}}\, V_{j,h,:}, \qquad
s_{ij}^{(h)} = \frac{1}{\sqrt D}\sum_{c=0}^{D-1} Q_{i,h,c}\, K_{j,h,c}
$$

| 符號 | 意義 |
|---|---|
| $M$ | 解碼器查詢數 |
| $N$ | 編碼器位置數（鍵/值數） |
| $H$ | 頭數 |
| $D$ | 每個頭的維度 |
| $Q_{i,h,c}$ | 查詢 $i$、頭 $h$、特徵 $c$；偏移量為 $(iH + h)D + c$ |
| $K_{j,h,c},\ V_{j,h,c}$ | 鍵/值 $j$、頭 $h$、特徵 $c$；偏移量為 $(jH + h)D + c$ |
| $s^{(h)}_{ij}$ | 頭 $h$ 中查詢 $i$ 與鍵 $j$ 的縮放分數 |
| $m^{(h)}_i$ | 列最大值 $\max_j s^{(h)}_{ij}$ |
| $O_{i,h,:}$ | 偏移量 $(iH + h)D$ 的輸出向量（長度為 $D$） |

### 轉置不需成本

參考實作會先將 $(M, H, D)$ 轉置為 $(H, M, D)$，再執行批次矩陣乘法。
在記憶體中，頭 $h$ 的第 $i$ 列起始位置為 $(iH + h)D$。因此頭 $h$
可視為一個**基底偏移量為 $hD$**、**列跨距為 $HD$** 的矩陣。
[多頭注意力](../012-multi-head-attention/)的跨距式 Flash 核心函式
正好接受這些參數，所以不必轉置或複製資料：

| `AttnGeom` 欄位 | 值 |
|---|---|
| `q_rows`, `kv_rows` | $M$, $N$ |
| `head_dim` | $D$ |
| `q_stride`, `kv_stride`, `o_stride` | $H \cdot D$ |
| `q_head`, `kv_head`, `o_head` | $D$ |
| `scale` | $1/\sqrt D$ |

## 解題思路

此核心函式是通用的 FlashAttention 風格前向運算：

- 網格為 $\lceil M/4\rceil \times H$，每個區塊 4 個 warp，
  每個 warp 處理一列查詢；
- 以 $D$ 的 128 寬切片，將 32 個鍵的 $K$/$V$ 分塊串流通過共享記憶體；
- lane $\ell$ 計算鍵 $\ell$ 的分數；使用修正項
  $\alpha = e^{m - m'}$ 的線上 softmax；透過 `__shfl_sync`
  廣播 $p_j$ 來累加 $PV$；使用暫存器累加器
  （當 $D = 256$ 時，每個 lane 最多 8 欄）。

每個分塊的代數推導請參閱
[Softmax 注意力](../006-softmax-attention/)，頭切片的細節請參閱
[多頭注意力](../012-multi-head-attention/)。

## 成本分析

$$
W = 4MNHD, \qquad Q_{\min} = 4\,(2MHD + 2NHD), \qquad I_{\max} = \frac{W}{Q_{\min}} = \frac{MN}{2(M+N)}
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP 數：每個頭的 $QK^{\mathsf T}$ 與 $PV$ 各為 $2MND$ |
| $Q_{\min}$ | 必要位元組數：$Q$、$K$、$V$ 各讀取一次，輸出寫入一次 |
| $I_{\max}$ | 可達到的最佳算術強度（FLOP/byte） |

基準測試中，$W \approx 17$ GFLOP、$Q_{\min} \approx 50$ MB，
且 $I_{\max} \approx 340$，因此明確**受限於計算**。每個區塊只有
4 列查詢，所以每個 K/V 分塊只重複使用 4 次。核心函式受限於 fp32 FMA
與共享記憶體載入吞吐量，而非 DRAM。下一步可改用張量核心版本，
並使用 64–128 列查詢的分塊。

## 常見陷阱

- **縮放。** 應使用 $1/\sqrt D$（每個頭的維度），不是
  $1/\sqrt{HD}$。
- **$M \ne N$。** `q_rows` 與 `kv_rows` 必須分開，列邊界檢查使用
  `q_rows`。
- **$D = 256$ 時的共享記憶體。**
  $4D + 32\cdot129 + 32\cdot128$ 個 float 約為 36 KB，
  低於預設的 48 KB。屬性設定呼叫可確保更大的 $D$ 仍安全。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md)
以 `1e-4` 容許誤差通過，包括 $M = 1$、$N = 1$ 與 $D = 256$。

## 延伸閱讀

- [多頭注意力](../012-multi-head-attention/)、
  [分組查詢注意力](../080-grouped-query-attention/)、
  [Softmax 注意力](../006-softmax-attention/)。
