---
title: 最小生成樹
platform: Tensara
upstream: min-spanning-tree
url: https://tensara.org/problems/min-spanning-tree
difficulty: medium
tags: [graph, prim, single-block, reduction]
status: solved
---

# 最小生成樹

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/min-spanning-tree)

## 問題

給定一個有 $n$ 個頂點（$n$ = 1024 … 6144）的無向圖，求其最小生成樹的總權重。輸入是對稱的稠密鄰接矩陣，邊權重為正整數，0 表示「沒有邊」。參考實作執行 Prim 演算法，若圖不連通則回傳 $+\infty$（題目敘述寫的是 $-1$，但以參考實作為準）。檢查條件為 `rtol = 1e-4`、`atol = 1e-3`。

## 公式

$$
\text{MST} = \operatorname*{arg\,min}_{T \in \mathcal{T}(G)} \sum_{(u,v)\in T} a_{uv}, \qquad \text{output} = \sum_{(u,v)\in\text{MST}} a_{uv}
$$

| 符號 | 意義 |
|---|---|
| $G$ | 圖；若且唯若 $a_{uv} > 0$，邊 $(u, v)$ 存在 |
| $a_{uv}$ | 邊的權重（對稱，$a_{uv} = a_{vu}$） |
| $\mathcal{T}(G)$ | $G$ 的生成樹集合（以 $n - 1$ 條邊連接所有頂點） |
| MST | 總權重最小的樹 |

Prim 演算法從頂點 0 開始擴展樹 $S$，並為 $S$ 外的每個頂點保留一條連入 $S$ 的最輕邊：

$$
\text{best}[v] = \min_{u \in S} a_{uv}, \qquad
u^\star = \operatorname*{arg\,min}_{v \notin S} \text{best}[v], \qquad
S \leftarrow S \cup \{u^\star\}, \qquad \text{best}[v] \leftarrow \min(\text{best}[v],\ a_{u^\star v})
$$

| 符號 | 意義 |
|---|---|
| $S$ | 已加入樹的頂點（`in_tree`） |
| $\text{best}[v]$ | 從 $v$ 連到 $S$ 的最輕邊權重（若不存在則為 $+\infty$） |
| $u^\star$ | 此步驟加入的頂點（同值時取最小索引） |

每一步都將 $\text{best}[u^\star]$ 加到總和。根據割集性質，跨越 $(S, V\setminus S)$ 的最輕邊必定屬於某棵 MST。

## 方法

**由一個含 1024 個執行緒的區塊執行全部 $n - 1$ 個步驟**，並以 `__syncthreads()` 同步，不需反覆啟動核心：

1. **Arg-min**：每個執行緒掃描其跨步分配且位於 $S$ 外的頂點；先以 warp shuffle，再以含 32 個項目的共享記憶體掃描，選出 $(\text{best}[u^\star], u^\star)$，同值時取索引較小者。
2. 若最小值為無限大（圖不連通），就**停止**並回傳 $+\infty$。
3. **鬆弛**：執行緒讀取矩陣的第 $u^\star$ 列（合併存取，共 $n$ 個 float），並降低 $\text{best}[v]$。
4. 每個執行緒都以 `double` 累加總和（它們看到的值都相同），最後由執行緒 0 寫入。

## 成本分析

$$
W = O(n^2), \qquad Q = 4n^2\ \text{bytes (each row read once)}, \qquad
T \approx (n - 1)\bigl(t_{\text{sync}} + t_{\text{scan}}(n)\bigr)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 總工作量：$n$ 個步驟，每步 $O(n)$ |
| $Q$ | DRAM 位元組數：每一步讀取一次第 $u^\star$ 列 |
| $t_{\text{sync}}$ | 兩次區塊屏障與 arg-min 的成本（約數 µs） |
| $t_{\text{scan}}(n)$ | 使用 1024 個執行緒掃描 $n$ 個值所需的時間 |

當 $n = 6144$ 時：需讀取 151 MB 的列資料，並執行約 6 K 個各需數 µs 的步驟，因此執行時間（約 10–20 ms）主要受串列步驟延遲影響，而非頻寬。單一區塊只會使用一個 SM；另一種做法是每一步都在整張 GPU 上啟動一次核心，但每步會付出約 3–5 µs 的啟動開銷。若要使用整張 GPU，應採用只有 $O(\log n)$ 個平行回合的 Borůvka 演算法。

## 注意事項

- **0 表示「沒有邊」**，因此在 `best` 中必須轉成 $+\infty$，不能視為權重為 0 的免費邊。
- **圖不連通**：依參考實作回傳 $+\infty$。
- **精度**：以 fp32 累加整數權重時，超過 $2^{24}$ 便會失去精確度；fp64 總和可保持精確。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 相關內容

- [最短路徑](../shortest-path/)、[全點對最短路徑](../all-pairs-shortest-path/)、[Argmin](../argmin/)。
