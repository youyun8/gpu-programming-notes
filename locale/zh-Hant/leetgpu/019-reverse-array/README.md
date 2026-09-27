---
title: 反轉陣列
platform: LeetGPU
upstream: easy/19_reverse_array
url: https://leetgpu.com/challenges/reverse-array
difficulty: easy
tags: [in-place, memory-bound, race-conditions]
status: solved
---

# 反轉陣列

**平台：** LeetGPU · **難度：** easy · [題目敘述](https://leetgpu.com/challenges/reverse-array)

## 題意

**就地**反轉長度為 $N$ 的 float32 陣列（$1 \le N \le 10^8$；
基準測試為 $N = 2.5\times10^7$）。重點是不能使用第二個緩衝區，
也不能產生資料競爭。

## 圖解

![原地反轉：執行緒 i 交換 x[i] 與其鏡像位置 x[N−1−i]](figure.svg)

每一條弧線代表一個執行緒交換一對元素。各對互不重疊，因此不需要同步；N 為奇數時，中間的元素（紅色）保持不動。

## 數學表述

$$
x'_i = x_{N-1-i}, \qquad 0 \le i < N
$$

| 符號 | 意義 |
|---|---|
| $N$ | 陣列長度 |
| $x_i$ | 呼叫前索引 $i$ 的值 |
| $x'_i$ | 呼叫後索引 $i$ 的值 |

映射 $i \mapsto N-1-i$ 是一種**對合**：它將索引 $i$ 與鏡像索引
$j = N-1-i$ 配成一對，套用兩次後會回到原值。因此，更新可拆成
$\lfloor N/2\rfloor$ 個互相獨立的交換：

$$
(x_i,\ x_j) \leftarrow (x_j,\ x_i), \qquad j = N-1-i,\ \ 0 \le i < \lfloor N/2 \rfloor
$$

| 符號 | 意義 |
|---|---|
| $j$ | $i$ 的鏡像索引 |
| $\lfloor N/2 \rfloor$ | 交換次數；若 $N$ 為奇數，中間元素 $x_{(N-1)/2}$ 是自己的鏡像，因此保持不變 |

## 解題思路

啟動 $\lceil \lfloor N/2\rfloor / 256\rceil$ 個區塊。執行緒
$i < \lfloor N/2\rfloor$ 將 $x_i$ 與 $x_j$ 載入暫存器，再以相反位置
寫回。每個記憶體位置都只由一個執行緒讀取與寫入，因此**沒有資料競爭**。

### 合併存取

一個 warp 的 32 個執行緒會讀取 $x_i, \dots, x_{i+31}$
（遞增且連續）以及 $x_{j-31}, \dots, x_j$
（遞減，但仍位於同一個 128 位元組區段）。硬體依位址集合而非順序
進行合併存取，因此兩側都能完全合併存取。

## 成本分析

$$
Q = 2 \cdot 4N \ \text{bytes}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 傳輸量：每個元素各讀取與寫入一次 |
| $\beta$ | DRAM 頻寬 |

基準測試：$Q = 200$ MB，因此在 2 TB/s 下
$T_{\min} \approx 100\ \mu s$。

## 常見陷阱

- **競爭式複製。** 啟動 $N$ 個執行緒，讓每個執行緒執行
  `x[N-1-i] = x[i]`，會覆寫其他執行緒尚未讀取的值。結果會依排程而異。
- **大小為零的網格。** 當 $N = 1$ 時不需交換任何元素。此時要略過啟動，
  因為含 0 個區塊的網格是無效設定。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md) 通過，
包括 $N = 1$、$N = 2$ 與奇數長度。使用 `--reverse` 排程也確認結果
不受執行緒順序影響。

## 延伸閱讀

- [矩陣轉置](../003-matrix-transpose/)（另一種單純的資料搬移）、
  [交錯排列](../063-interleave/)。
