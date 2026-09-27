---
title: 平行反向掃描（GAE）
platform: LeetGPU
upstream: medium/110_gae_reverse_scan
url: https://leetgpu.com/challenges/parallel-reverse-scan-gae
difficulty: medium
tags: [scan, rl, reverse-scan, affine-maps]
status: solved
---

# 平行反向掃描（GAE）

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/parallel-reverse-scan-gae)

## 問題

對 $B$ 條長度為 $S$ 的軌跡執行**廣義優勢估計**：先計算一步 TD 誤差，再從每個序列末端進行*反向*折扣累加（容許誤差 `1e-3`）。GAE 會將優勢提供給 [PPO](../107-ppo-clipped-surrogate-loss/)。它由右至左的相依關係，是一種向後執行的掃描。

## 公式

$$
\delta_t = r_t + \gamma\,V_{t+1} - V_t\quad (V_S = 0), \qquad
A_t = \delta_t + c\,A_{t+1}\quad (A_S = 0), \qquad c = \gamma\lambda
$$

展開後：$A_t = \sum_{k=0}^{S-1-t} c^{k}\,\delta_{t+k}$。

| 符號 | 意義 |
|---|---|
| $B,\ S$ | 批次（軌跡）與序列長度 |
| $r_t$ | 步驟 $t$ 的獎勵 |
| $V_t$ | 步驟 $t$ 的價值估計；最後一步之後的價值為 0 |
| $\gamma$ | 折扣因子 |
| $\lambda$ | GAE 參數，用來權衡偏差與變異數 |
| $c$ | 合併衰減 $\gamma\lambda$ |
| $\delta_t$ | 時序差分誤差 |
| $A_t$ | 優勢（輸出） |

### 以仿射映射進行掃描

每個步驟都是映射 $f_t(a) = c\,a + \delta_t$，從 $t = S-1$ 往 0 套用，並從 $a = 0$ 開始。合成具有結合律（請參閱[線性遞迴](../082-linear-recurrence/)）。分段 $[lo, hi)$ 會合成為

$$
A_{lo} = M\cdot A_{hi} + D, \qquad M = c^{\,hi - lo}, \qquad D = \sum_{t=lo}^{hi-1} c^{\,t - lo}\,\delta_t
$$

| 符號 | 意義 |
|---|---|
| $[lo, hi)$ | 一段連續的時間步 |
| $M,\ D$ | 分段合成映射的乘數與位移 |

## 方法

**每條軌跡使用一個含 1024 個執行緒的區塊：**

1. **反向分段分配。** 執行緒 $k$ 負責從末端數來第 $k$ 個分段：$[\,S - (k+1)p,\ S - kp\,)$，其中 $p = \lceil S/1024\rceil$。依序掃描執行緒 0、1、2、……，正好會沿著遞迴方向（時間上由右至左），因此一般的*正向*區塊掃描即可運作。
2. 每個執行緒向後走訪自己的分段，以 float64 摺疊 $(M, D)$。$\delta_t$ 會即時從 $r_t, V_t, V_{t+1}$ 計算。
3. 使用合成方式 $(M_1, D_1)$ 再接 $(M_2, D_2) = (M_1M_2,\ D_2 + M_2 D_1)$，對映射執行**區塊掃描**：先做 warp shuffle，再處理各 warp 的總結果。
4. 將互斥前綴套用到 $A_S = 0$，即可得到從右側**進入**分段的優勢。執行緒會由右至左再次走訪其分段，寫入 $A_t$。

## 成本分析

$$
Q = 12BS\ \text{bytes}, \qquad W \approx 8BS + O(B\cdot1024\log1024)
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 位元組數：讀取獎勵與價值（價值讀取兩次但會被快取），並寫入優勢 |
| $W$ | 摺疊與重播的浮點運算次數（線性），另加區塊掃描 |

在一般的 RL 大小下（數千步 × 數百條軌跡），核心受記憶體頻寬限制，且每條軌跡使用一個區塊。

## 常見問題

- **合成方向。** 位於*右側*的分段會先作用。若顛倒運算元順序，只有長序列會出錯。
- **啟動值。** $V_S = 0$（末端之後沒有啟動值），與參考實作相同。
- 當 $S < 1024$ 時，**空分段**必須是恆等映射 $(1, 0)$。

## 驗證

所有 LeetGPU 測試案例都以 `1e-3` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $S = 1$，以及題目說明中的範例（$c = 0.45$ 時會得到 $A = [3.308, 4.24, 4.2, 2.0]$）。

## 相關內容

- [線性遞迴](../082-linear-recurrence/)、[PPO 截斷損失](../107-ppo-clipped-surrogate-loss/)、[前綴和](../016-prefix-sum/)。
