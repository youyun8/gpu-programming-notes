---
title: 排序
platform: LeetGPU
upstream: hard/15_sorting
url: https://leetgpu.com/challenges/sorting
difficulty: hard
tags: [sorting, radix-sort, bit-tricks, scan]
status: solved
---

# 排序

**平台：** LeetGPU · **難度：** hard · [題目說明](https://leetgpu.com/challenges/sorting)

## 問題

將 $N$ 個 float32 值原地由小到大排序（$1 \le N \le 10^6$；
基準 $N = 10^6$）。參考實作為 `torch.sort`，可使用任何演算法。
對 32 位元鍵而言，GPU 上最快的通用選擇是完全不需比較的
**LSD 基數排序**。唯一障礙是 IEEE 浮點數的符號與指數編碼，
可用位元技巧解決。

## 公式

找出排列 $\pi$，使

$$
y_k = x_{\pi(k)}, \qquad y_0 \le y_1 \le \dots \le y_{N-1}
$$

| 符號 | 意義 |
|---|---|
| $N$ | 元素數量 |
| $x_i$ | 輸入值（float32） |
| $\pi$ | $\{0, \dots, N-1\}$ 的一個排列 |
| $y_k$ | 排序後的輸出，寫回 `data` |

### 保序的浮點數 → 整數映射

IEEE-754 浮點數比較方式類似符號－大小整數。映射

$$
f(u) =
\begin{cases}
u \oplus \texttt{0xFFFFFFFF}, & \text{sign bit of } u = 1 \ (\text{negative}) \\
u \mathbin{\vert} \texttt{0x80000000}, & \text{sign bit of } u = 0 \ (\text{non-negative})
\end{cases}
$$

是 32 位元模式上的雙射，且對所有非 NaN 浮點數都有
$a < b \iff f(\text{bits}(a)) < f(\text{bits}(b))$。
負數的所有位元都會翻轉，使絕對值較大的數成為較小的無號數。
非負數只設定最高位元，使其位於所有負數之後。

| 符號 | 意義 |
|---|---|
| $u$ | 浮點數的原始 32 位元模式（`__float_as_uint`） |
| $\oplus$ | 位元 XOR |
| $\vert$ | 位元 OR |
| $f(u)$ | 整數順序與浮點數順序相同的無號鍵 |

### LSD 基數排序

將每個鍵以 $2^8$ 為基底寫成數字 $(d_3, d_2, d_1, d_0)$。
依序對 $d_0$、$d_1$、$d_2$、$d_3$ 執行四次**穩定**計數排序，
即可完整排序。穩定性保證目前數字相同時，會保留下位數已建立的順序。
每一遍中，分塊 $t$ 內數字為 $d$ 的鍵 $x$ 會移到

$$
\text{pos}(x) = \underbrace{\sum_{d' < d}\ \sum_{t'} c_{d',t'} + \sum_{t' < t} c_{d,t'}}_{\text{exclusive scan of } c \text{ in digit-major order}} \;+\; \text{rank of } x \text{ among digit-}d\text{ keys of tile } t
$$

| 符號 | 意義 |
|---|---|
| $d$ | 鍵目前的 8 位元數字，$0..255$ |
| $t$ | 包含此鍵的分塊（每區塊 2048 個鍵） |
| $c_{d,t}$ | 分塊 $t$ 中數字為 $d$ 的鍵數 |
| 排名 | 同一分塊中，在此鍵之前且數字同為 $d$ 的鍵數（維持穩定性） |

## 方法

所有基數排序機制都與[基數排序](../036-radix-sort/)共用：

1. `floatToKey`：將 $f$ 套用後寫入暫存鍵緩衝區。
2. 對四個數字（位移 0、8、16、24）各執行：
   1. `digitCounts`：每個 2048 鍵分塊建立 256 分箱共享記憶體直方圖，
      並以數字優先方式儲存為 `hist[d * tiles + t]`。
   2. 對 $256 \times$ 分塊表執行 `exclusiveScan`（先歸約再掃描）。
   3. `scatterStable`：計算每個鍵在分塊內的排名。Warp 內的
      `__match_any_sync(digit)` 會傳回數字相同的 lane 遮罩，
      `__popc(peers & lanes_below)` 則是鍵在其中的排名。
      每個 warp 的數字計數在 8 個 warp 間做前綴和，並由各數字持續更新的
      基底跨越分塊的 8 個子區段。每個鍵都會取得唯一且保序的位置。
3. 經四次乒乓切換後，鍵回到原始緩衝區。`keyToFloat` 套用
   $f^{-1}$ 並寫入 `data`。

## 成本分析

$$
Q \approx \underbrace{8N}_{\text{map in/out}} + P\,(\underbrace{4N}_{\text{count}} + \underbrace{8N}_{\text{scatter}}) + 8N, \qquad P = \frac{32}{8} = 4
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 流量（位元組；直方圖表可忽略） |
| $P$ | 基數遍數（32 位元鍵、8 位元數字） |
| $4N$ | 讀取鍵以計算數字 |
| $8N$ | 讀取並散佈鍵 |

當 $N = 10^6$，$Q \approx 64$ MB，頻寬時間為數十微秒。
工作量是 $O(PN)$，而位元排序為 $O(N\log^2 N)$。
約 20 次核心啟動的延遲在此大小下占有明顯比例。

## 常見問題

- **負浮點數。** 將原始位元模式當無號整數排序，會把負數放在正數之後，
  而且順序相反；映射 $f$ 同時修正兩者。
- **穩定性。** 以 `atomicAdd` 更新各數字計數器的散佈雖快但不穩定，
  會破壞 LSD 基數排序。match-any 排名可維持穩定。
- **$-0.0$ 與 $+0.0$。** 兩者映射到相鄰鍵（$-0 < +0$）。
  浮點比較認為兩者相等，因此任一順序都可接受。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上通過；
額外測試涵蓋全相等、反向、$\pm 0$、非正規數及 $\pm\infty$ 輸入。

## 相關內容

- [基數排序](../036-radix-sort/)（無號整數）、[Top-K](../029-top-k-selection/)、
  [前綴和](../016-prefix-sum/)（此處使用的掃描）。
- Tensara [陣列排序](../../tensara/array-sort/)。
