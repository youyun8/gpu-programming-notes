---
title: 單一起點最短路徑
platform: Tensara
upstream: shortest-path
url: https://tensara.org/problems/shortest-path
difficulty: medium
tags: [graph, bellman-ford, early-exit]
status: solved
---

# 單一起點最短路徑

**平台：** Tensara · **難度：** 中等 · [題目敘述](https://tensara.org/problems/shortest-path)

## 問題

在含 $N$ 個頂點（$N$ = 512 … 8192）的有向圖中，求單一起點最短路徑。
圖以含正整數權重的稠密鄰接矩陣表示（0 代表沒有邊），並指定起點 $s$。
無法到達的頂點輸出 $-1$。參考實作會執行 $N - 1$ 輪 Bellman–Ford。
檢查條件為 `rtol = 1e-4`、`atol = 1e-3`。

## 公式

$$
d^{(0)}[v] = \begin{cases} 0, & v = s \\ +\infty, & v \ne s \end{cases}, \qquad
d^{(t+1)}[v] = \min\Bigl(d^{(t)}[v],\ \min_{u\,:\,a_{uv} > 0} \bigl(d^{(t)}[u] + a_{uv}\bigr)\Bigr)
$$

$$
\text{out}[v] = \begin{cases} d^{(T)}[v], & d^{(T)}[v] < \infty \\ -1, & \text{otherwise} \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $s$ | 起點 |
| $a_{uv}$ | 邊 $u \to v$ 的權重（0 = 不存在） |
| $d^{(t)}[v]$ | 到 $v$ 的最短距離，最多使用 $t$ 條邊 |
| $T$ | 已執行的輪數 |
| out | 最終距離；無法到達時為 $-1$ |

執行 $t$ 輪後，$d^{(t)}$ 對每個最短路徑至多含 $t$ 條邊的頂點
都是精確值。因此，第一輪沒有任何變化時即可停止：

$$
d^{(t+1)} = d^{(t)} \ \Longrightarrow\ d^{(t)} = d^{(\infty)}, \qquad T \le \text{(max edges on a shortest path)} + 1
$$

| 符號 | 意義 |
|---|---|
| $d^{(\infty)}$ | 真正的最短距離（最多經過 $N - 1$ 輪即可得到） |

## 方法

1. **`initDist`**：起點設為 $0$，其餘設為 $+\infty$。
2. **`relax`**：每個目的地 $v$ 由一個執行緒負責；走訪所有 $u$ 並讀取
   $a_{uv}$，也就是第 $v$ 欄。對固定的 $u$，warp 的 32 個 lane 會讀取
   $a_{u,v..v+31}$，形成 128 個連續位元組的合併存取。新距離會寫入第二個
   緩衝區（Jacobi 方式），任何變化都會設定裝置旗標。
3. **主機端迴圈**：重設旗標、啟動核心、讀回旗標並交換緩衝區；
   沒有變化時停止（最多 $N - 1$ 輪）。
4. **`finish`**：$+\infty \to -1$。

## 成本分析

$$
Q = 4N^2 \cdot T\ \text{bytes}, \qquad W = N^2 T\ \text{relaxations}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：每輪讀取完整矩陣 |
| $W$ | 加法與比較操作次數 |
| $T$ | 收斂前的輪數 |

$N = 8192$ 時，矩陣為 268 MB；在 2 TB/s 下，每輪約 0.13 ms。
隨機稠密圖的最短路徑通常只含少量邊，因此 $T$ 很小；參考實作固定執行
$N - 1$ 輪則會超過一秒。若有 $N$ 個執行緒 = 8192，則只有 32 個區塊，
所以每輪受延遲限制；若將 $u$ 迴圈分給每個 $v$ 的多個執行緒
（並執行最小值歸約），可提高 GPU 使用率。

## 注意事項

- **每輪都讀取旗標**：裝置到主機的複製會同步，每輪花費數 µs，
  但可提早結束。
- **Jacobi 與 Gauss–Seidel**：原地更新對 Bellman–Ford 也正確
  （只會更快收斂），但雙緩衝可避免核心發生讀寫競爭。
- **0 = 沒有邊**；**無法到達 = $-1$**。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [全點對最短路徑](../all-pairs-shortest-path/)、[最小生成樹](../min-spanning-tree/)、
  LeetGPU [BFS 最短路徑](../../leetgpu/046-bfs-shortest-path/)。
