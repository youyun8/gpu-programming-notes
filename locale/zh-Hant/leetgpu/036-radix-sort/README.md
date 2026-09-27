---
title: 基數排序
platform: LeetGPU
upstream: hard/36_radix_sort
url: https://leetgpu.com/challenges/radix-sort
difficulty: hard
tags: [sorting, radix-sort, scan, warp-intrinsics, stable]
status: solved
---

# 基數排序

**平台：** LeetGPU · **難度：** 困難 · [題目敘述](https://leetgpu.com/challenges/radix-sort)

## 題意

使用**基數排序**，將 $N$ 個無號 32 位元整數依遞增順序排列
（$1 \le N \le 10^8$；效能評測使用 $N = 5\times10^7$）。結果寫入
`output`。對固定位數的鍵值而言，基數排序是最快的通用 GPU 排序演算法，
它結合了先前題目中的三項基本運算：**直方圖**、**掃描**與**穩定散佈**。

## 圖解

![LSD 基數排序的一輪：計數各位數、掃描計數、穩定地分散寫出](figure.svg)

鍵依本輪的位數上色。各位數的計數經過掃描後得到起始偏移量；每個鍵寫到「該位數的偏移量 + 同位數中的名次」，因此這一輪是穩定的。

## 數學表述

將每個鍵值以 $R = 2^8$ 為基底表示：

$$
u = \sum_{p=0}^{3} d_p(u)\, R^{p}, \qquad d_p(u) = \left\lfloor \frac{u}{R^p} \right\rfloor \bmod R
$$

| 符號 | 意義 |
|---|---|
| $u$ | 32 位元鍵值 |
| $R$ | 基數，$2^8 = 256$ |
| $p$ | 數位位置（輪次編號），0 為最低有效位 |
| $d_p(u)$ | $u$ 的第 $p$ 個數位（第 $8p \dots 8p+7$ 位元） |

**LSD 基數排序**從 $p = 0$ 到 3，對每個數位各執行一次*穩定*計數排序。
完成第 $p$ 輪後，鍵值會依最低的 $8(p+1)$ 個位元排序。
穩定性是指相同數位的元素仍維持原本的相對順序，正是這項特性讓歸納成立。

### 單輪操作：每個鍵值要放到哪裡？

將輸入分成每塊 $T = 2048$ 個鍵值的資料塊。令 $c_{d,t}$ 表示資料塊
$t$ 中數位為 $d$ 的鍵值數量。資料塊 $t$ 中位置為 $q$ 的鍵值 $u$，
其目的位置為

$$
\operatorname{pos}(u) = \underbrace{\sum_{d' < d}\sum_{t'} c_{d',t'} \;+\; \sum_{t' < t} c_{d,t'}}_{\operatorname{off}(d,\,t)\ =\ \text{exclusive scan of } c \text{ in digit-major order}} \;+\; \underbrace{\#\{\, q' < q : d_p(u_{q'}) = d \,\}}_{\text{rank within the tile}}, \qquad d = d_p(u)
$$

| 符號 | 意義 |
|---|---|
| $T$ | 資料塊大小：256 個執行緒 × 8 個分段 = 2048 個鍵值 |
| $t$ | 資料塊索引 |
| $c_{d,t}$ | 直方圖：資料塊 $t$ 中數位為 $d$ 的鍵值數量 |
| $\operatorname{off}(d, t)$ | （數位、資料塊）儲存區的全域起點 |
| $q$ | 鍵值在資料塊內的位置 |
| 排名 | 同一資料塊中，較早出現且數位相同的鍵值數量（這讓該輪保持穩定） |

以**數位優先**方式儲存 $c$，即索引為 $d \cdot \text{tiles} + t$，
就能以*一次*平坦的互斥掃描計算 $\operatorname{off}$：先放所有資料塊中
數位為 0 的項目，再放數位為 1 的項目，依此類推。

## 解題思路

每一輪的作法如下（共 4 輪，在 `output` 與暫存緩衝區之間交替）：

1. **`digitCounts`**：每個區塊負責一個資料塊，在共享記憶體中以原子操作
   建立 256 個分組的直方圖，再以數位優先方式寫出。
2. 對 $256 \times \text{tiles}$ 表格執行 **`exclusiveScan`**：
   如[前綴和](../016-prefix-sum/)一樣，以 3 個核心函式進行先歸約再掃描
   （分段總和 → 掃描總和 → 套用）。
3. **`scatterStable`**：區塊會**依序**處理資料塊中的 8 個分段，
   每段有 256 個鍵值。每個分段執行：
   - `peers = __match_any_sync(full, digit)`：位元遮罩，代表此 warp 中
     持有相同數位的執行緒。
   - `rank = __popc(peers & lanes_below_me)`：warp 內的穩定排名。
   - warp 領頭執行緒（`__ffs(peers)-1`）將群組大小記錄於
     `s_warp[w][digit]`。
   - 執行緒 $d$ 將數位 $d$ 在 8 個 warp 中的計數轉為起始偏移。
     持續更新的 `s_base[d]` 會跨分段保留，初值為
     $\operatorname{off}(d, t)$。
   - 每個鍵值會寫入 `s_warp[w][digit] + rank`。

超過 $N$ 的無效執行緒會使用虛擬數位 256，因此它們會自行形成配對群組，
不會與真正的數位衝突。

完成 4 輪後（輪數為偶數），結果會回到 `output`。

## 成本分析

$$
Q \approx P\,\bigl(\underbrace{4N}_{\text{count}} + \underbrace{4N + 4N}_{\text{scatter r/w}}\bigr) + 8N = 56N \ \text{bytes}, \qquad P = 4
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（直方圖表格與掃描約額外增加 0.5%） |
| $P$ | 輪數 |
| $8N$ | 將輸入初次複製至輸出 |

效能評測使用 $N = 5\times10^7$，此時 $Q \approx 2.8$ GB，
在 2 TB/s 下約需 1.4 ms。散佈寫入只能部分合併：同一 warp 中數位相同的
鍵值會連續寫入，但不同數位仍會分散。因此散佈是最慢的步驟。正式環境的排序
實作（如 CUB onesweep）會使用解耦回看機制，融合計數與散佈，並使用
11 位元數位（3 輪）。

## 常見陷阱

- **不穩定的散佈。** 對各數位的游標使用 `atomicAdd`，會以任意順序分配位置。
  如此在隨機資料上看似正確，但輸入包含許多最低位數相同的值時就會失敗。
- **資料塊順序。** 資料塊內的各分段必須依序處理，資料塊本身則透過數位優先
  掃描排序。兩者都不可少，才能維持穩定性。
- **暫存記憶體：**需要第二個可容納 $N$ 個鍵值的緩衝區，再加上
  $256\cdot\lceil N/2048\rceil$ 的直方圖及其掃描總和。

## 驗證

所有 LeetGPU 測試案例均在 [cuemu](../../tools/cuemu/README.md) 通過；
它使用真正的 warp 會合來實作 `__match_any_sync`。壓力測試涵蓋
$N = 1$、所有鍵值皆相同、只在最高位元組不同的鍵值，以及
$0$/$2^{32}-1$ 的極端值。

## 延伸閱讀

- [排序](../015-sorting/)（透過鍵值映射處理浮點數）、[Top-K](../029-top-k-selection/)、
  [直方圖](../013-histogramming/)、[前綴和](../016-prefix-sum/)。
