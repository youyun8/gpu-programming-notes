---
title: 彩虹表
platform: LeetGPU
upstream: easy/24_rainbow_table
url: https://leetgpu.com/challenges/rainbow-table
difficulty: easy
tags: [hashing, compute-bound, integer]
status: solved
---

# 彩虹表

**平台：** LeetGPU · **難度：** easy · [題目敘述](https://leetgpu.com/challenges/rainbow-table)

## 題意

對 $N$ 個整數分別套用 $R$ 次 32 位元 **FNV-1a** 雜湊
（$1 \le N \le 10^7$、$1 \le R \le 100$，輸入位於 $[0, 2^{31})$；
基準測試為 $N = 5\times10^6$）。輸出為 uint32，且必須完全相符。
這是彩虹表鏈的基本元件，也是清單中第一個**受限於計算**的問題：
每傳輸 8 位元組，最多可執行 800 次整數運算。

## 圖解

![彩虹表鏈：對每個輸入字組做 R 輪 32 位元 FNV-1a，每個執行緒處理一個](figure.svg)

上排是單一執行緒的 R 輪雜湊鏈；下排展開其中一輪：字組的四個位元組逐一以 XOR 混入，並乘上 FNV 質數。

## 數學表述

對 32 位元字組 $x$ 的 4 個小端序位元組執行 FNV-1a：

$$
h_0 = \texttt{0x811C9DC5}, \qquad
h_{b+1} = \bigl(h_b \oplus \operatorname{byte}_b(x)\bigr)\cdot P \bmod 2^{32}, \quad b = 0,1,2,3, \qquad
H(x) = h_4
$$

$$
\operatorname{byte}_b(x) = \left\lfloor \frac{x}{2^{8b}} \right\rfloor \bmod 256, \qquad P = 16777619 = \texttt{0x01000193}
$$

輸出是 $R$ 次函數合成：

$$
y_i = \underbrace{H(H(\cdots H}_{R}(x_i)\cdots))
$$

| 符號 | 意義 |
|---|---|
| $x$, $x_i$ | 輸入字組（將 int32 輸入重新解讀為 uint32） |
| $h_b$ | 取用 $b$ 個位元組後的雜湊狀態 |
| $\operatorname{byte}_b(x)$ | $x$ 中第 $b$ 個最低有效位元組 |
| $\oplus$ | 位元 XOR |
| $P$ | FNV 32 位元質數 |
| $\bmod 2^{32}$ | 無號 32 位元算術的環繞 |
| $H$ | 對一個 32 位元字組完成一次雜湊 |
| $R$ | 回合數 |
| $y_i$ | 輸出（uint32） |

## 解題思路

- 每個元素使用一個執行緒。只載入一次 $x_i$，在暫存器內完成全部
  $R$ 個回合，再儲存一次。
- `fnv1a` 展開其 4 位元組迴圈：4 ×（位移、AND、XOR、乘法）。
  採用**無號**算術時，溢位依定義會以模 $2^{32}$ 環繞，
  因而完全重現參考實作的 `& 0xFFFFFFFF`。
- 各回合本質上必須循序執行（每回合都依賴前一次雜湊），但不同元素
  互相獨立。所有平行處理能力都來自 $N$。

## 成本分析

$$
W \approx 16RN \ \text{integer ops}, \qquad Q = 8N \ \text{bytes}, \qquad I = 2R\ \text{ops/byte}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 每次雜湊約 16 條整數指令（4 位元組 × {位移、AND、XOR、乘法}），再乘以 $R$ 回合 |
| $Q$ | 每個元素讀取 4 位元組並寫入 4 位元組 |
| $I$ | 每位元組 DRAM 傳輸量的運算數 |

當 $R = 100$ 時，$I = 200$ ops/byte，遠高於任何 GPU 的平衡點。
吞吐量受限於 **32 位元整數乘法**速率。在多數 NVIDIA 架構上，
IMAD 的發出速率是 FP32 的一半或相同。占用率可隱藏延遲：
許多互相獨立的執行緒各自執行一條很長的相依鏈。

## 常見陷阱

- **帶號溢位**在 C++ 中是未定義行為。編譯器可能假設它永遠不會發生。
  雜湊一律要使用 `unsigned int`。
- **位元組順序。** FNV-1a 從最低有效位元組開始取用，
  與參考實作的 `(x >> (8*i)) & 0xFF` 迴圈相符。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md)
通過完全相等檢查，包括 $R = 1$ 與 $R = 100$。

## 延伸閱讀

- [蒙地卡羅積分](../035-monte-carlo-integration/)（另一個計算密集的逐元素核心函式）。
