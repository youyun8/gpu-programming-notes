---
title: SSM 選擇性掃描
platform: LeetGPU
upstream: medium/94_ssm_selective_scan
url: https://leetgpu.com/challenges/ssm-selective-scan
difficulty: medium
tags: [ssm, mamba, scan, registers]
status: solved
---

# SSM 選擇性掃描

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/ssm-selective-scan)

## 問題

Mamba 的**選擇性掃描**：一種線性狀態空間遞迴，其離散化方式會透過時間步長 $\Delta$ 隨輸入改變。輸入為 $u, \Delta \in \mathbb R^{B\times L\times D}$、$A \in \mathbb R^{D\times N}$（負值）、$B_{\text{proj}}, C \in \mathbb R^{B\times L\times N}$，以及略過向量 $\mathbf s \in \mathbb R^{D}$；輸出為 $y$（$B \le 16$、$L \le 8192$、$D \le 2048$、$N \le 64$；效能測試為 $B = 4$、$L = 4096$、$D = 512$、$N = 16$；容許誤差 `1e-3`）。

## 公式

對批次 $b$、通道 $d$、狀態索引 $n$ 與時間 $t$（其中 $h_{-1} = 0$）：

$$
\bar A_{t,n} = e^{\Delta_{t,d}A_{d,n}}, \qquad \bar B_{t,n} = \Delta_{t,d}\,B_{t,n}, \qquad
h_{t,n} = \bar A_{t,n}\, h_{t-1,n} + \bar B_{t,n}\, u_{t,d}, \qquad
y_{t,d} = \sum_{n=0}^{N-1} C_{t,n}\,h_{t,n} + s_d\,u_{t,d}
$$

| 符號 | 意義 |
|---|---|
| $B,\ L,\ D,\ N$ | 批次、序列長度、通道數（`d_model`）、狀態大小（`d_state`） |
| $u_{t,d}$ | 時間 $t$、通道 $d$ 的輸入（省略批次索引 $b$） |
| $\Delta_{t,d}$ | 正值且隨輸入改變的步長（這正是掃描具「選擇性」的原因） |
| $A_{d,n}$ | 連續時間狀態矩陣（每個通道皆為對角矩陣），值為負 |
| $\bar A_{t,n}$ | 離散化衰減 $\in (0,1)$（零階保持） |
| $B_{t,n},\ C_{t,n}$ | 輸入與輸出投影，由該批次的所有通道共用 |
| $\bar B_{t,n}$ | 離散化輸入增益（Euler） |
| $h_{t,n}$ | 通道 $d$ 的隱藏狀態（長度 $N$） |
| $s_d$ | 略過項（Mamba 記號中的「$D$」項） |
| $y_{t,d}$ | 輸出 |

每個 $(b, d)$ 配對都會執行 $N$ 個互相獨立的一階線性遞迴，其形式與[線性遞迴](../082-linear-recurrence/)相同，但係數會隨時間變化。

## 方法

### 跨通道平行、時間軸循序

在效能測試大小下，共有 $B\cdot D = 2048$ 個獨立的 $(b, d)$ 配對，因此每個執行緒可負責一個配對，並沿時間軸依序處理。這樣便不必使用複雜的平行掃描（Mamba 的 CUDA 核心在通道少、序列長時會使用該方法）。

- **暫存器**：執行緒將狀態向量 `h[64]` 與其資料列 `A[d, :]` 保存在暫存器中。對 $n$ 的迴圈具有編譯期上限（64），會完全展開並以 `n < d_state` 防護，因此每個索引都是常數，不會溢出到區域記憶體。
- **共享廣播**：同一區塊中的 128 個執行緒具有相同批次 $b$（網格為 $\lceil D/128\rceil \times B$）。它們的 $B_{t,:}$ 與 $C_{t,:}$ 相同，因此區塊每次將 32 個時間步的資料暫存在共享記憶體中，再以廣播方式讀取。
- **合併串流**：$u_{t,d}$ 與 $\Delta_{t,d}$ 採通道置後配置，因此每個 warp 在各時間步都會讀取 32 個連續通道。
- 每一步會進行 $N$ 次指數運算、$N$ 次狀態更新，以及與 $C$ 的 $N$ 項內積。

## 成本分析

$$
W \approx BLDN\,(c_{\exp} + 5), \qquad Q = 4BLD\cdot 3 + 8BLN\cdot\left\lceil\frac{D}{128}\right\rceil\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 運算量：每個 (b, t, d, n) 執行一次 `expf`，另加約 5 次浮點運算 |
| $c_{\exp}$ | `expf` 的成本 |
| $Q$ | 位元組數：各讀取 $u$、$\Delta$ 並寫入 $y$ 一次；每個通道區塊都會重新讀取 $B$ 與 $C$ |

效能測試：共有 $BLDN = 1.3\times10^8$ 次狀態更新，主要成本來自 `expf`（SFU 執行 `ex2` 的速率為 1/4），因此最多只需數毫秒。記憶體流量約為 100 MB。Mamba 的正式環境核心也會融合離散化，並以平行分段的方式沿時間軸掃描。

## 常見問題

- **使用動態索引的暫存器陣列**會降級到區域記憶體（較慢）。固定且展開的上限可避免此問題。
- **屏障位置。** 非作用中的執行緒（通道 ≥ $D$）仍須協助暫存資料並抵達屏障；它們只能在暫存完成*之後*才 `continue`。
- **離散化順序。** 參考實作會先計算 $\Delta\cdot B$，再乘上 $u$。核心維持 `(dt * B) * u` 的順序。

## 驗證

所有 LeetGPU 測試案例都以 `1e-3` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $N = 1$ 與 $N = 64$。

## 相關內容

- [線性遞迴](../082-linear-recurrence/)、[因果深度可分離一維卷積](../090-causal-depthwise-conv1d/)、[線性注意力](../056-linear-attention/)。
