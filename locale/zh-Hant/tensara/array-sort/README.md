---
title: 陣列排序
platform: Tensara
upstream: array-sort
url: https://tensara.org/problems/array-sort
difficulty: easy
tags: [sorting, radix-sort, bit-tricks]
status: solved
---

# 陣列排序

**平台：** Tensara · **難度：** easy · [題目敘述](https://tensara.org/problems/array-sort)

## 問題

將 $n$ 個**有號** int32 值依遞增順序排序（$n = 16\,384 \dots 262\,144$），並進行完全一致的檢查。對固定寬度鍵值而言，GPU 排序的首選是最低有效位優先（LSD）基數排序，但它處理的是*無號*位數。唯一要處理的細節，是用一個位元的轉換讓有號順序等同於無號順序。

## 公式

$$
f(x) = \operatorname{bits}(x) \oplus \texttt{0x80000000}, \qquad x < y \iff f(x) <_{\text{unsigned}} f(y)
$$

| 符號 | 意義 |
|---|---|
| $x, y$ | 有號 32 位元整數（二補數） |
| $\operatorname{bits}(x)$ | 原始 32 位元位元模式 |
| $\oplus$ | XOR；翻轉符號位元 |
| $f$ | 從 int32 到 uint32 且保留順序的映射 |

若以無號數解讀，二補數會將負數 $\texttt{0x80000000} \dots \texttt{0xFFFFFFFF}$ 排在非負數*之後*。翻轉最高位元相當於將每個值平移 $2^{31}$，也就是把 $[-2^{31}, 2^{31})$ 單調映射至 $[0, 2^{32})$。

映射後，執行四趟每次 8 位元的穩定計數排序即可排好鍵值（每趟的散佈公式請見 LeetGPU [基數排序](../../leetgpu/036-radix-sort/)）。

## 方法

1. 將 $a \to f(a)$ 映射至輸出緩衝區（型別視為 uint32）。
2. **穩定 LSD 基數排序**：執行 4 趟，並與暫存緩衝區交替使用：
   圖塊直方圖（以位數為主）、整個裝置範圍的互斥掃描，以及使用
   `__match_any_sync` 排名的穩定散佈。
3. 使用同一個 XOR 映射回原值（XOR 是自己的反函數）。

## 成本分析

$$
Q \approx 4\cdot 12n + 8n = 56n\ \text{bytes}, \qquad W = O(4n)
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：每趟為計數讀取一次，再為散佈讀寫一次；另加兩次映射 |
| $W$ | 工作量，與 $n$ 成線性關係 |

當 $n = 262\,144$ 時，資料量為 15 MB，可完全容納於 L2。執行時間主要來自約 20 次核心函式啟動。對這麼小的陣列，使用單一執行緒區塊排序（例如在共享記憶體中使用雙調排序，或使用單區塊基數排序）可減少啟動次數。

## 常見陷阱

- **將原始位元當成無號數排序**會把負數放在最後。
- **穩定性**是 LSD 能正確運作的必要條件。

## 驗證

所有測試案例在 [cuemu](../../tools/cuemu/README.md) 上皆完全相同，包括全為負數、所有值相同，以及極值（$\pm 2^{31}$ 邊界）。

## 相關內容

- LeetGPU [基數排序](../../leetgpu/036-radix-sort/)、[排序（浮點數）](../../leetgpu/015-sorting/)、
  [前 K 大](../../leetgpu/029-top-k-selection/)。
