---
title: 多代理人模擬
platform: LeetGPU
upstream: hard/14_multi_agent_sim
url: https://leetgpu.com/challenges/multi-agent-simulation
difficulty: hard
tags: [n-body, simulation, shared-memory, floating-point]
status: solved
---

# 多代理人模擬

**平台：** LeetGPU · **難度：** hard · [題目說明](https://leetgpu.com/challenges/multi-agent-simulation)

## 問題

為 $N$ 個代理人執行一步群聚（「boids」）**對齊**規則
（$1 \le N \le 10^5$；基準 $N = 10\,000$）。每個代理人由 4 個浮點數
$[x, y, v_x, v_y]$ 表示。每個代理人會將速度朝半徑 $r = 5$ 內其他
代理人的平均速度調整 5%，再依新速度移動。結果寫入 `agents_next`，
容許誤差 `1e-5`。這是全配對（$N$ 體風格）互動，也是用共享記憶體
平鋪二次迴圈的經典示範。

## 公式

$$
\mathcal N_i = \bigl\{\, j \ne i \ :\ (x_i - x_j)^2 + (y_i - y_j)^2 < r^2 \,\bigr\}
$$

$$
\bar{\mathbf v}_i =
\begin{cases}
\dfrac{1}{\lvert\mathcal N_i\rvert}\displaystyle\sum_{j\in\mathcal N_i} \mathbf v_j, & \lvert\mathcal N_i\rvert > 0\\[2mm]
\mathbf v_i, & \lvert\mathcal N_i\rvert = 0
\end{cases}
\qquad
\mathbf v_i' = \mathbf v_i + \alpha\,(\bar{\mathbf v}_i - \mathbf v_i), \qquad
\mathbf p_i' = \mathbf p_i + \mathbf v_i'
$$

| 符號 | 意義 |
|---|---|
| $N$ | 代理人數量 |
| $\mathbf p_i = (x_i, y_i)$ | 代理人 $i$ 的位置 |
| $\mathbf v_i = (v_{x,i}, v_{y,i})$ | 代理人 $i$ 的速度 |
| $r$ | 鄰域半徑，$r = 5$（所以 $r^2 = 25$） |
| $\mathcal N_i$ | 代理人 $i$ 的鄰居集合（不含自己） |
| $\lvert\mathcal N_i\rvert$ | 鄰居數量 |
| $\bar{\mathbf v}_i$ | 鄰居平均速度（沒有鄰居時為自己的速度） |
| $\alpha$ | 轉向率，$0.05$ |
| $\mathbf v_i',\ \mathbf p_i'$ | 更新後的速度與位置，寫入 `agents_next` |

所有更新都只讀取**舊**狀態，因此這一步是純函式
`agents → agents_next`，沒有讀寫競爭。

## 方法

### 平鋪全配對迴圈

- 每個代理人 $i$ 由一個執行緒處理，每區塊 256 個執行緒。
  執行緒將自己的 `float4` 保存在暫存器中。
- 其他代理人以每塊 256 個處理。每個區塊合作將一塊以 `float4`
  載入共享記憶體（每執行緒一次 16 位元組載入），執行
  `__syncthreads()`，再由每個執行緒走訪 256 個已暫存的代理人。
  載入是**廣播**：所有執行緒同時讀取 `tile[t]`。
  經第二次屏障後載入下一塊。
- 執行緒在暫存器中累加 $\sum v_x$、$\sum v_y$ 與鄰居數量，
  再套用更新公式。

每個代理人的資料由每個區塊從 DRAM 取一次，而不是每對取一次，
使全域流量減少 256 倍。

### 位元完全一致的鄰居測試

成員條件 $d^2 < 25$ 是硬性臨界值。剛好位於半徑上或相差一個 ulp 的
代理人，必須與 PyTorch 完全相同地分類。PyTorch 計算
`(diff**2).sum()` 時會分別取整兩個平方，再對加法取整。
因此核心使用 `__fsub_rn`、`__fmul_rn` 與 `__fadd_rn`，
編譯器不得將它們**融合為 FMA**。FMA 只取整一次
`dx*dx + dy*dy`，可能翻轉邊界案例，使 $\lvert\mathcal N_i\rvert$
相差一，足以無法通過 `1e-5`。

## 成本分析

$$
W \approx c\,N^2, \qquad Q_{\text{DRAM}} \approx 16N\left\lceil\frac{N}{256}\right\rceil + 32N
$$

| 符號 | 意義 |
|---|---|
| $W$ | 運算量：每個有序配對約 $c \approx 8$ FLOP（距離、比較、累加） |
| $Q_{\text{DRAM}}$ | 位元組數：每個區塊串流全部 $N$ 個代理人（各 16 位元組），加上讀寫自身狀態 |

當 $N = 10^4$，$W \approx 8\times10^8$ 次運算，最快也需數百微秒。
對更大的 $N$，問題在 $O(N^2)$ 演算法，而非常數因子。
以單元大小 $r$ 的**均勻網格**（依單元排序代理人，只掃描鄰近 3 × 3
單元）可讓有界密度下的步驟成為 $O(N)$。

## 常見問題

- **FMA 融合**會改變邊界分類（見上文）。
- **排除自身。** 必須以索引測試 $j \ne i$，不能用距離 0；
  兩個代理人可能位於相同位置。
- **分塊迴圈中的屏障。** $i \ge N$ 的執行緒仍須執行有防護的載入，
  並到達每個分塊的兩次 `__syncthreads()`。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上以
`1e-5` 通過，包括代理人距離恰為 $r$ 的配置。

## 相關內容

- [最近鄰居](../038-nearest-neighbor/)（相同的全配對平鋪）、[K-Means](../020-kmeans-clustering/)。
