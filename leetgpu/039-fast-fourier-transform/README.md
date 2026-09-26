---
title: Fast Fourier Transform
platform: LeetGPU
upstream: hard/39_Fast_Fourier_transform
url: https://leetgpu.com/challenges/fast-fourier-transform
difficulty: hard
tags: [fft, bluestein, stockham, complex]
status: solved
---

# Fast Fourier Transform

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/fast-fourier-transform)

## Problem

Compute the discrete Fourier transform of $N$ complex float32 samples,
stored interleaved as `[re0, im0, re1, im1, …]` ($1 \le N \le 262\,144$,
**any** $N$, not only powers of two; benchmark $N = 2^{18}$; tolerance
`1e-3`). cuFFT is not allowed. This page derives the radix-2 FFT in its
self-sorting **Stockham** form, and **Bluestein's algorithm**, which reduces
arbitrary lengths to power-of-two FFTs.

## Formulation

$$
X_k = \sum_{n=0}^{N-1} x_n\, \omega_N^{kn}, \qquad \omega_N = e^{-2\pi i/N}, \qquad 0 \le k < N
$$

| Symbol | Meaning |
|---|---|
| $N$ | transform length |
| $x_n$ | input sample (complex: `signal[2n] + i·signal[2n+1]`) |
| $X_k$ | output coefficient (written interleaved to `spectrum`) |
| $\omega_N$ | principal $N$-th root of unity (twiddle factor base) |
| $i$ | imaginary unit |

Direct evaluation is $O(N^2)$, i.e. $6.9\times10^{10}$ complex multiply-adds
for $N = 2^{18}$.

### Radix-2 split (Cooley–Tukey)

For even $N$, split into even and odd samples:

$$
X_k = E_k + \omega_N^{k} O_k, \qquad X_{k + N/2} = E_k - \omega_N^{k} O_k, \qquad 0 \le k < N/2
$$

| Symbol | Meaning |
|---|---|
| $E_k$ | length-$N/2$ DFT of the even-indexed samples $x_0, x_2, \dots$ |
| $O_k$ | length-$N/2$ DFT of the odd-indexed samples $x_1, x_3, \dots$ |
| $\omega_N^k$ | twiddle factor |

Recursing gives $\log_2 N$ levels of $N/2$ "butterflies". The total is
$O(N\log N)$.

### Stockham formulation (no bit reversal)

The in-place Cooley–Tukey FFT needs a bit-reversal permutation. Stockham
instead reads from one buffer and writes to another in each pass, placing
results directly in natural order. In the pass with sub-transform size $s$
($s = 1, 2, 4, \dots, N/2$), for $j = 0 \dots N/2 - 1$ with
$k = j \bmod s$ and $q = \lfloor j/s\rfloor$:

$$
v_0 = \text{in}[j], \quad v_1 = \text{in}[j + N/2]\cdot e^{\mp i\pi k/s}, \qquad
\text{out}[2qs + k] = v_0 + v_1, \quad \text{out}[2qs + k + s] = v_0 - v_1
$$

| Symbol | Meaning |
|---|---|
| $s$ | size of the sub-transforms completed in earlier passes |
| $j$ | butterfly index (one thread per butterfly) |
| $k$ | position inside the current sub-transform |
| $q$ | which sub-transform |
| $e^{\mp i\pi k/s}$ | twiddle ($-$ forward, $+$ inverse), computed with `sincospif` |

### Bluestein (chirp-z) for arbitrary $N$

Using $kn = \tfrac12\bigl(k^2 + n^2 - (k-n)^2\bigr)$:

$$
X_k = \overline{w_k}^{\,*}\ \sum_{n=0}^{N-1} \underbrace{\bigl(x_n w_n\bigr)}_{a_n}\ \underbrace{\overline{w_{k-n}}}_{b_{k-n}}, \qquad w_m = e^{-i\pi m^2/N}
$$

(so $X_k = w_k \cdot (a * b)_k$ with $b_m = \overline{w_m}$), which is a linear
convolution. Zero-pad to a power of two $L \ge 2N - 1$ and evaluate it with
FFTs:

$$
(a * b) = \operatorname{IFFT}_L\bigl(\operatorname{FFT}_L(a) \odot \operatorname{FFT}_L(b)\bigr)
$$

| Symbol | Meaning |
|---|---|
| $w_m$ | chirp $e^{-i\pi m^2/N}$ |
| $\overline{\,\cdot\,}$ | complex conjugate |
| $a_n$ | chirp-modulated input, zero for $n \ge N$ |
| $b_m$ | conjugate chirp, stored at $m = 0..N-1$ and wrapped to $L - m$ for negative offsets |
| $*$ | linear convolution |
| $L$ | padded power-of-two length, $L \ge 2N-1$ |
| $\odot$ | elementwise (pointwise) product |

## Approach

- **Power-of-two $N$**: copy the input to `spectrum`, then run $\log_2 N$
  `stockhamPass` kernels, ping-ponging with a scratch buffer. Each thread
  (grid-stride) performs one butterfly.
- **Other $N$**:
  1. `bluesteinPrep` builds $a$ and $b$ of length $L$.
  2. Two forward FFTs of size $L$.
  3. `pointwiseMul`.
  4. An inverse FFT (sign $+1$, unnormalised).
  5. `bluesteinFinish` multiplies by $w_k / L$.

### Exact chirp phases

For $n$ near $2.6\times10^5$, $n^2 \approx 7\times10^{10}$ exceeds float32
precision by far, so $\pi n^2/N$ computed in float32 would have a useless
phase. Because $w_m$ has period $2N$ in $m^2$, the kernel reduces
$m^2 \bmod 2N$ **in 64-bit integers** first. It then calls
`sincospif(-(m² mod 2N)/N)`, which evaluates $\sin(\pi t)$ and $\cos(\pi t)$
without the error of multiplying by a rounded $\pi$.

## Cost analysis

$$
W_{\text{pow2}} \approx 5N\log_2 N, \qquad
W_{\text{Bluestein}} \approx 3 \cdot 5L\log_2 L + O(L), \qquad
Q \approx \log_2 N \cdot 16N \ \text{bytes (pow2)}
$$

| Symbol | Meaning |
|---|---|
| $W$ | real FLOPs (standard radix-2 estimate: 5 per point per level) |
| $Q$ | DRAM traffic: each pass reads and writes $N$ complex values (8 bytes each) |
| $L$ | Bluestein padded length (up to $2^{19}$ for $N < 2^{18}$) |

For $N = 2^{18}$: 18 passes × 4 MB = 75 MB of traffic, about 40 µs. It is
bandwidth-bound, because each pass does little math. Radix-4/8 passes, or
doing the first 10 levels inside shared memory, would cut the number of
global passes by 3–5×.

## Pitfalls

- **Phase precision** for large $n$ (see above). Without the modular
  reduction, errors exceed `1e-3` beyond $N \approx 10^4$.
- **Normalisation.** The inverse FFT is unnormalised. The $1/L$ is folded
  into the final chirp multiply.
- **Bluestein's $b$ must be circularly symmetric**: entries $L-m$ for
  $m = 1..N-1$ hold the negative offsets.
- **$N = 1$** is a power of two with zero passes, i.e. a plain copy.

## Verification

Checked against numpy for **every** $N$ from 1 to 4096, for primes, and for
$N = 2^{18} - 1$ and $2^{18}$. All LeetGPU cases pass on
[cuemu](../../tools/cuemu/README.md) at `1e-3`.

## Related

- [2D FFT](../078-2d-fft/), Tensara [Polynomial Multiply (finite field)](../../tensara/poly-multiply-ff/)
  (the number-theoretic cousin of FFT convolution).
