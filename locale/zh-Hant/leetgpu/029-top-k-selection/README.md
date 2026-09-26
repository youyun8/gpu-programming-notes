---
title: Top K 選取
platform: LeetGPU
upstream: medium/29_top_k_selection
url: https://leetgpu.com/challenges/top-k-selection
difficulty: medium
tags: [selection, radix-select, bitonic-sort, bit-tricks]
status: solved
---

# Top K 選取

**平台：** LeetGPU · **難度：** medium · [題目敘述](https://leetgpu.com/challenges/top-k-selection)

## 問題

從 $N$ 個 float32 值中找出最大的 $k$ 個，並以遞減順序傳回
（$1 \le k \le N \le 10^8$；基準測試為 $N = 5\times10^7$、
$k = 100$；除了 `atol = 1e-5` 外必須精確）。即使只需要 100 個值，
排序整個陣列仍需 $O(N\log N)$，或 4 次以上基數排序的散佈傳輸。
**基數選取**只需執行幾次唯讀直方圖處理，即可找出第 $k$ 大的值，
最後只排序留下的值。

## 公式

令 $x_{(0)} \ge x_{(1)} \ge \dots \ge x_{(N-1)}$ 為輸入遞減排序後的結果。
輸出為

$$
\text{out}_r = x_{(r)}, \qquad 0 \le r < k
$$

而**臨界值** $\tau = x_{(k-1)}$ 滿足

$$
\#\{\, i : x_i > \tau \,\} \;<\; k \;\le\; \#\{\, i : x_i \ge \tau \,\}
$$

| 符號 | 意義 |
|---|---|
| $N$ | 輸入值數量 |
| $k$ | 要傳回的值數量 |
| $x_{(r)}$ | 第 $r$ 大的值（順序統計量，從 0 起算） |
| $\tau$ | 第 $k$ 大的值（臨界值） |
| $\#\{\cdot\}$ | 滿足條件的索引數 |

答案包含每個 $> \tau$ 的元素，再加上正好
$k - \#\{x_i > \tau\}$ 個 $\tau$。

### 對保序鍵執行基數選取

使用 $f$ 將浮點數映射為無號鍵（請參閱[排序](../015-sorting/)），
使鍵的順序與浮點數順序相同。從最高有效位數開始，每次決定 8 個位元，
即可找出鍵 $T = f(\tau)$。第 $p$ 次處理
（$p = 3, 2, 1, 0$，位移量為 $8p$）只考慮高位數符合目前前綴的候選值：

$$
c_d = \#\{\, i : \operatorname{prefix}(f(x_i)) = \Pi,\ \operatorname{digit}_p(f(x_i)) = d \,\}, \qquad
d^\star = \max\Bigl\{ d : \sum_{d' \ge d} c_{d'} \ge \rho \Bigr\}, \qquad
\rho \leftarrow \rho - \sum_{d' > d^\star} c_{d'}
$$

| 符號 | 意義 |
|---|---|
| $\Pi$ | 先前處理已決定的 $T$ 位數（`prefix`，由 `mask` 標示有效位元） |
| $\operatorname{digit}_p(u)$ | 鍵 $u$ 的第 $8p \dots 8p+7$ 位元 |
| $c_d$ | 直方圖：目前位數等於 $d$ 的候選值數 |
| $\rho$ | $T$ 在目前候選值中的名次（初始為 $k$） |
| $d^\star$ | 本次處理的 $T$ 位數：從 $d = 255$ 向下走，累計數首次達到 $\rho$ 的 bucket |

4 次處理後即可完全得知 $T$，而 $\rho$ 是答案中應包含的 $T$ 複本數。

## 方法

所有狀態（`prefix`、`mask`、`remaining`、收集游標、直方圖）都存放在
`__device__` 變數中。各核心函式依序啟動，不需在主機與裝置之間複製：

1. **`digitHistogram` × 4。** 以網格跨步方式執行 `floatToKey`，
   符合前綴的候選值則遞增共享記憶體直方圖。每個區塊以全域原子操作
   寫回非零 bucket（與[直方圖統計](../013-histogramming/)相同的私有化）。
2. **`chooseDigit` × 4**（1 個執行緒）：執行上述 256 個 bucket 的走訪，
   並重設直方圖供下一次使用。
3. **`gatherGreater`。** 每個鍵 $> T$ 的元素都透過原子游標附加到小型
   緩衝區。這類元素少於 $k$ 個。
4. **`fillTail`。** 將 $[\text{count}, k)$ 的位置填入 $T$，
   $[k, \text{padded})$ 則填入鍵 0（最小鍵，排序後位於最後）。
5. 對填補至二次方大小的緩衝區執行**遞減雙調排序**。當 $k \le 2048$
   時（基準測試為 $k = 100 \to 128$），由一個區塊在共享記憶體中完成；
   否則每個（大小、跨距）步驟都使用一個全域核心函式。
6. **`writeOutput`**：對前 $k$ 個鍵套用 $f^{-1}$。

### 一句話說明雙調排序

對區塊大小 $s = 2, 4, \dots, P$ 與跨距
$t = s/2, \dots, 1$，元素 $i$ 會與 $j = i \oplus t$ 比較並交換。
當 $(i \mathbin{\&} s) = 0$ 時方向為遞減。總共有
$\frac{\log_2 P(\log_2 P + 1)}{2}$ 個與資料無關的階段，
非常適合 SIMT。

## 成本分析

$$
Q \approx \underbrace{4 \cdot 4N}_{\text{4 histogram passes}} + \underbrace{4N}_{\text{gather}} = 20N \ \text{bytes}, \qquad
W_{\text{sort}} = O\!\left(P \log^2 P\right),\ P = 2^{\lceil\log_2 k\rceil}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數；每次處理只讀取輸入（每元素 4 位元組） |
| $P$ | 填補後的排序大小（大於等於 $k$ 的下一個二次方） |
| $W_{\text{sort}}$ | 對留下元素執行雙調排序時的比較交換數 |

基準測試中，$Q = 1$ GB，在 2 TB/s 下約為 0.5 ms。這與 $k$ 無關，
傳輸量約比完整基數排序少 2–3 倍；完整排序每次都要讀取並**散佈**
$N$ 個鍵。128 個鍵的雙調排序成本可忽略。

## 常見問題

- **臨界值平手。** 收集時使用 `>=` $\tau$ 可能傳回超過 $k$ 個元素。
  準確計入 $\rho$ 個 $T$ 複本，才能正確處理重複值。
- **負浮點數。** 若不使用保序鍵映射，負數會排在正數前面。
- **填補值。** 遞減排序時，填補值必須排在所有真實鍵之後；
  鍵 0 可保證這點（除了測試不會出現的 NaN 位元模式外，
  沒有浮點數會映射到 0）。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md) 通過。
額外測試涵蓋 $k = N$、$k = 1$、輸入全部相等、大量 $\tau$ 重複值、
負值，以及 $k > 2048$（全域雙調路徑）。

## 相關內容

- [排序](../015-sorting/)、[基數排序](../036-radix-sort/)、
  [Top-p 取樣](../060-top-p-sampling/)、
  [MoE Top-k 閘控](../067-moe-topk-gating/)。
- Tensara [Argmax](../../tensara/argmax/)。
