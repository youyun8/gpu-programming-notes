---
title: 快速傅立葉轉換
platform: LeetGPU
upstream: hard/39_Fast_Fourier_transform
url: https://leetgpu.com/challenges/fast-fourier-transform
difficulty: hard
tags: [fft, bluestein, stockham, complex]
status: solved
---

# 快速傅立葉轉換

**平台：** LeetGPU · **難度：** 困難 · [題目敘述](https://leetgpu.com/challenges/fast-fourier-transform)

## 問題

計算 $N$ 個複數 float32 樣本的離散傅立葉轉換。樣本以
`[re0, im0, re1, im1, …]` 交錯儲存（$1 \le N \le 262\,144$；
可為**任意** $N$，不只限於 2 的冪次；效能評測使用 $N = 2^{18}$；
容許誤差為 `1e-3`）。不得使用 cuFFT。本頁會推導採用自行排序
**Stockham** 形式的 radix-2 FFT，以及將任意長度化為 2 的冪次 FFT 的
**Bluestein 演算法**。

## 公式

$$
X_k = \sum_{n=0}^{N-1} x_n\, \omega_N^{kn}, \qquad \omega_N = e^{-2\pi i/N}, \qquad 0 \le k < N
$$

| 符號 | 意義 |
|---|---|
| $N$ | 轉換長度 |
| $x_n$ | 輸入樣本（複數：`signal[2n] + i·signal[2n+1]`） |
| $X_k$ | 輸出係數（交錯寫入 `spectrum`） |
| $\omega_N$ | $N$ 次單位根的主值（旋轉因子的基底） |
| $i$ | 虛數單位 |

直接計算的複雜度為 $O(N^2)$；當 $N = 2^{18}$ 時，需要
$6.9\times10^{10}$ 次複數乘加。

### Radix-2 拆分（Cooley–Tukey）

當 $N$ 為偶數時，將樣本分成偶數索引與奇數索引：

$$
X_k = E_k + \omega_N^{k} O_k, \qquad X_{k + N/2} = E_k - \omega_N^{k} O_k, \qquad 0 \le k < N/2
$$

| 符號 | 意義 |
|---|---|
| $E_k$ | 偶數索引樣本 $x_0, x_2, \dots$ 的長度 $N/2$ DFT |
| $O_k$ | 奇數索引樣本 $x_1, x_3, \dots$ 的長度 $N/2$ DFT |
| $\omega_N^k$ | 旋轉因子 |

遞迴後會得到 $\log_2 N$ 層，每層有 $N/2$ 個「蝶形」運算。
總複雜度為 $O(N\log N)$。

### Stockham 公式（不需位元反轉）

就地 Cooley–Tukey FFT 需要位元反轉排列。Stockham 則在每一輪從一個
緩衝區讀取並寫入另一個緩衝區，直接將結果放入自然順序。在子轉換大小為
$s$ 的輪次中（$s = 1, 2, 4, \dots, N/2$），對
$j = 0 \dots N/2 - 1$，令 $k = j \bmod s$ 且
$q = \lfloor j/s\rfloor$：

$$
v_0 = \text{in}[j], \quad v_1 = \text{in}[j + N/2]\cdot e^{\mp i\pi k/s}, \qquad
\text{out}[2qs + k] = v_0 + v_1, \quad \text{out}[2qs + k + s] = v_0 - v_1
$$

| 符號 | 意義 |
|---|---|
| $s$ | 先前輪次已完成的子轉換大小 |
| $j$ | 蝶形運算索引（每個蝶形使用一個執行緒） |
| $k$ | 目前子轉換內的位置 |
| $q$ | 子轉換編號 |
| $e^{\mp i\pi k/s}$ | 旋轉因子（正向用 $-$、反向用 $+$），以 `sincospif` 計算 |

### 適用任意 $N$ 的 Bluestein（Chirp-Z）

利用 $kn = \tfrac12\bigl(k^2 + n^2 - (k-n)^2\bigr)$：

$$
X_k = \overline{w_k}^{\,*}\ \sum_{n=0}^{N-1} \underbrace{\bigl(x_n w_n\bigr)}_{a_n}\ \underbrace{\overline{w_{k-n}}}_{b_{k-n}}, \qquad w_m = e^{-i\pi m^2/N}
$$

（因此 $X_k = w_k \cdot (a * b)_k$，其中
$b_m = \overline{w_m}$），這是一個線性摺積。以零填補至滿足
$L \ge 2N - 1$ 的 2 的冪次，再使用 FFT 計算：

$$
(a * b) = \operatorname{IFFT}_L\bigl(\operatorname{FFT}_L(a) \odot \operatorname{FFT}_L(b)\bigr)
$$

| 符號 | 意義 |
|---|---|
| $w_m$ | Chirp $e^{-i\pi m^2/N}$ |
| $\overline{\,\cdot\,}$ | 複數共軛 |
| $a_n$ | 經 chirp 調變的輸入，當 $n \ge N$ 時為零 |
| $b_m$ | 共軛 chirp，在 $m = 0..N-1$ 儲存，負偏移則環繞至 $L - m$ |
| $*$ | 線性摺積 |
| $L$ | 填補後的 2 的冪次長度，$L \ge 2N-1$ |
| $\odot$ | 逐元素（逐點）乘積 |

## 方法

- **$N$ 為 2 的冪次**：將輸入複製到 `spectrum`，接著執行
  $\log_2 N$ 個 `stockhamPass` 核心函式，並與暫存緩衝區交替使用。
  每個執行緒以網格跨步方式執行一個蝶形運算。
- **其他 $N$：**
  1. `bluesteinPrep` 建立長度為 $L$ 的 $a$ 與 $b$。
  2. 執行兩次大小為 $L$ 的正向 FFT。
  3. 執行 `pointwiseMul`。
  4. 執行一次反向 FFT（符號為 $+1$，不正規化）。
  5. `bluesteinFinish` 乘上 $w_k / L$。

### 精確的 Chirp 相位

當 $n$ 接近 $2.6\times10^5$ 時，$n^2 \approx 7\times10^{10}$，
遠超過 float32 的精度，因此若以 float32 計算 $\pi n^2/N$，
得到的相位將無法使用。由於 $w_m$ 對 $m^2$ 的週期為 $2N$，
核心函式會先以 **64 位元整數**計算 $m^2 \bmod 2N$。接著呼叫
`sincospif(-(m² mod 2N)/N)`，直接計算 $\sin(\pi t)$ 與
$\cos(\pi t)$，避免乘上一個經捨入的 $\pi$ 所造成的誤差。

## 成本分析

$$
W_{\text{pow2}} \approx 5N\log_2 N, \qquad
W_{\text{Bluestein}} \approx 3 \cdot 5L\log_2 L + O(L), \qquad
Q \approx \log_2 N \cdot 16N \ \text{bytes (pow2)}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 實數 FLOP 數（標準 radix-2 估算：每點每層 5 次） |
| $Q$ | DRAM 流量：每一輪讀寫 $N$ 個複數值（每個 8 位元組） |
| $L$ | Bluestein 填補後的長度（當 $N < 2^{18}$ 時最大為 $2^{19}$） |

當 $N = 2^{18}$ 時，18 輪 × 4 MB = 75 MB 流量，約需 40 µs。
由於每輪的運算量不多，因此受頻寬限制。使用 radix-4/8 輪次，或在共享記憶體
中完成前 10 層，可將全域輪次減少 3–5 倍。

## 常見陷阱

- 大型 $n$ 的**相位精度**（見上文）。若不先取模，當
  $N \approx 10^4$ 以上時，誤差會超過 `1e-3`。
- **正規化。** 反向 FFT 未正規化；$1/L$ 會併入最後一次 chirp 乘法。
- **Bluestein 的 $b$ 必須循環對稱**：$m = 1..N-1$ 時，$L-m$ 的項目
  儲存負偏移。
- **$N = 1$** 是零輪的 2 的冪次，也就是直接複製。

## 驗證

已與 numpy 比對從 1 到 4096 的**每一個** $N$、各種質數，以及
$N = 2^{18} - 1$ 與 $2^{18}$。所有 LeetGPU 測試案例均以 `1e-3`
的容許誤差在 [cuemu](../../tools/cuemu/README.md) 通過。

## 相關內容

- [2D FFT](../078-2d-fft/)、Tensara [多項式乘法（有限體）](../../tensara/poly-multiply-ff/)
  （FFT 摺積在數論中的近親）。
