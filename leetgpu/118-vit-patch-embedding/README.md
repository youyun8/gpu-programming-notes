---
title: Vision Transformer Patch Embedding
platform: LeetGPU
upstream: medium/118_vit_patch_embedding
url: https://leetgpu.com/challenges/vision-transformer-patch-embedding
difficulty: medium
tags: [gemm, im2col, vision, vit]
status: solved
---

# Vision Transformer Patch Embedding

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/vision-transformer-patch-embedding)

## Problem

The patch-embedding stem of a **Vision Transformer** (ViT, CLIP, SigLIP,
DiT). Split each of $B$ images ($C\times H\times W$, NCHW) into
non-overlapping $P\times P$ patches, project each flattened patch to $D$
dimensions, prepend a learned CLS token, and add learned positional
embeddings (tolerance `1e-4`). This is a convolution with kernel = stride =
$P$, and it is implemented as a GEMM.

## Formulation

$$
g_h = \frac HP,\quad g_w = \frac WP,\quad N = g_hg_w, \qquad n = p_y g_w + p_x
$$

$$
t_{b,n,d} = \beta_d + \sum_{c=0}^{C-1}\sum_{i=0}^{P-1}\sum_{j=0}^{P-1} I_{b,c,\ p_yP + i,\ p_xP + j}\ \ W_{d,c,i,j}
$$

$$
Y_{b,0,:} = \mathbf{cls} + E_{0,:}, \qquad Y_{b,n+1,:} = \mathbf t_{b,n,:} + E_{n+1,:}
$$

| Symbol | Meaning |
|---|---|
| $B,\ C,\ H,\ W$ | batch, channels, image height and width |
| $P$ | patch size ($H$, $W$ divisible by $P$) |
| $g_h,\ g_w$ | patch-grid height and width |
| $N$ | patches per image |
| $n,\ (p_y, p_x)$ | patch index and its grid position (row-major) |
| $I_{b,c,y,x}$ | input pixel |
| $W_{d,c,i,j}$ | projection weight ($D\times C\times P\times P$), i.e. a conv kernel |
| $\beta_d$ | projection bias |
| $t_{b,n,d}$ | patch token |
| $\mathbf{cls}$ | learned CLS vector (not projected) |
| $E$ | positional embeddings, $(N+1)\times D$ |
| $Y$ | output, $B\times(N+1)\times D$ |

### As a GEMM (Implicit im2col)

Flatten each patch in $(c, i, j)$ order into a row of length $K = CP^2$. The
patch matrix $\mathcal P \in \mathbb R^{BN\times K}$ then gives

$$
T = \mathcal P\,W_{\text{flat}}^{\mathsf T} + \boldsymbol\beta, \qquad W_{\text{flat}} \in \mathbb R^{D\times CP^2}
$$

| Symbol | Meaning |
|---|---|
| $\mathcal P$ | im2col matrix: row $bN + n$ is patch $n$ of image $b$ |
| $W_{\text{flat}}$ | the weight reshaped to $D\times CP^2$ (row-major, so this is an "NT" GEMM) |

Since stride = kernel, patches do not overlap, and im2col duplicates no
pixel: $\mathcal P$ is a pure permutation of the image.

## Approach

- **`patchGemm`**: the 64 × 64 register-blocked NT-GEMM. Its **A-tile
  loader** computes, for (row, $k$), the source pixel
  $(b, c, p_yP + i, p_xP + j)$ from the index arithmetic above (`patchPixel`)
  and reads it straight from the image. The patch matrix is **never
  materialised**.
- The **epilogue** adds $\beta_d$ and $E_{n+1,d}$, and writes to output row
  $b(N+1) + n + 1$, skipping the CLS slot.
- A tiny kernel writes the $B$ CLS rows $\mathbf{cls} + E_0$.

## Cost Analysis

$$
W = 2BN\cdot CP^2\cdot D = 2BCHW\cdot D, \qquad Q_{\min} = 4\bigl(BCHW + DCP^2 + B(N+1)D\bigr)
$$

| Symbol | Meaning |
|---|---|
| $W$ | FLOPs; note $NP^2 = HW$ |
| $Q_{\min}$ | compulsory bytes: image, weight, output |

For ViT-B/16 on $224^2$ images ($C = 3$, $P = 16$, $D = 768$, $N = 196$), one
image costs $2\cdot3\cdot224^2\cdot768 = 231$ MFLOP. With $K = 768$, the
GEMM is modestly compute-bound. The gather loads in the A-tile are coalesced
along $j$, i.e. $P$ consecutive pixels of an image row.

## Pitfalls

- **Flattening order** must match the weight layout $(c, i, j)$,
  channel-major, as in the reference's `permute(0,2,4,1,3,5)`.
- **Row offset** of $+1$ for the CLS token in every image's output block.
- **CLS is not projected.** It is copied and gets $E_0$ added.

## Verification

All LeetGPU cases pass on [cuemu](../../tools/cuemu/README.md) at `1e-4`,
including $P = 1$ (every pixel is a patch) and $P = H = W$ (one patch).

## Related

- [DiT Block](../116-dit-block/), [Token Embedding](../106-token-embedding-layer/),
  [2D Convolution](../010-2d-convolution/). Tensara [2D Convolution](../../tensara/conv-2d/).
