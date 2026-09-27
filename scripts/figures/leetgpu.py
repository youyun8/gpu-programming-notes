"""One figure per LeetGPU problem, written to leetgpu/<dir>/figure.svg (see build_figures.py).

The figures use small, concrete numbers (computed here, so they are always
consistent) to show what the kernel computes and how the work is divided.
"""
import math

from . import kit
from .kit import W, fmt, cells, row_label, note, matrix, finish, Plot
from .kit import Svg


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def silu(x):
    return x * sigmoid(x)


def gelu(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


# ----- 001-020 -----------------------------------------------------------------------------------------------

def fig_001_vector_add(name):
    a = [1.5, 2, -1, 0.5, 3, 4, -2, 1]
    b = [0.5, 1, 2, 2.5, -1, 0, 3, 1]
    return kit.elementwise(name, "Vector addition: thread i reads A[i] and B[i] and writes C[i]",
                           [("A", [fmt(v) for v in a], "a"), ("B", [fmt(v) for v in b], "b")],
                           ("C", [fmt(x + y) for x, y in zip(a, b)], "c"), "+",
                           note_lines=["i = blockIdx.x · blockDim.x + threadIdx.x; threads with i ≥ N return.",
                                       "12 bytes of traffic per FLOP: the kernel is memory-bound."])


def fig_002_matrix_multiplication(name):
    return kit.gemm(name, "Tiled matrix multiplication: one block owns a C tile and walks the inner dimension",
                    dims=("M", "K", "N"),
                    note_lines=["One block computes one C tile (dark green).",
                                "Per step it stages a slice of A and a slice",
                                "of B (dark blue / orange) in shared memory,",
                                "then moves on along N, the inner dimension",
                                "in this problem. Each staged value is reused",
                                "by a whole row or column of the tile."])


def fig_003_matrix_transpose(name):
    s = Svg(name, W, 100, "Transpose through a shared-memory tile: coalesced reads and coalesced writes")
    R, C, cw = 4, 4, 30
    lab = lambda r, c: str(r * C + c)
    x1, y0 = 40, 50
    matrix(s, x1, y0, R, C, cw, fill_fn=lambda r, c: "f-a2" if r == 1 else None, role="a")
    for r in range(R):
        for c in range(C):
            s.text(x1 + c * cw + cw / 2, y0 + r * cw + cw / 2, lab(r, c), size="tiny")
    s.text(x1 + C * cw / 2, y0 - 14, "input tile (row-major)", size="small", role="a", bold=True)
    x2 = x1 + C * cw + 90
    matrix(s, x2, y0, R, C + 1, cw, fill_fn=lambda r, c: "fig-panel" if c == C else ("f-a2" if r == 1 else None),
           role="d")
    for r in range(R):
        for c in range(C):
            s.text(x2 + c * cw + cw / 2, y0 + r * cw + cw / 2, lab(r, c), size="tiny")
    s.text(x2 + (C + 1) * cw / 2, y0 - 14, "shared tile[32][33]", size="small", role="d", bold=True)
    x3 = x2 + (C + 1) * cw + 90
    matrix(s, x3, y0, C, R, cw, fill_fn=lambda r, c: "f-c2" if r == 1 else None, role="c")
    for r in range(C):
        for c in range(R):
            s.text(x3 + c * cw + cw / 2, y0 + r * cw + cw / 2, lab(c, r), size="tiny")
    s.text(x3 + R * cw / 2, y0 - 14, "output tile", size="small", role="c", bold=True)
    s.arrow(x1 + C * cw + 8, y0 + 1.5 * cw, x2 - 8, y0 + 1.5 * cw, role="a", sw=1.3)
    s.arrow(x2 + (C + 1) * cw + 8, y0 + 1.5 * cw, x3 - 8, y0 + 1.5 * cw, role="c", sw=1.3)
    yb = y0 + R * cw
    s.text((x1 + C * cw + x2) / 2, yb + 22, "warp reads a row", size="small", role="a")
    s.text((x2 + (C + 1) * cw + x3) / 2, yb + 22, "warp reads a column,", size="small", role="c")
    s.text((x2 + (C + 1) * cw + x3) / 2, yb + 39, "writes a row", size="small", role="c")
    return finish(s, note(s, 40, yb + 72, ["Row 1 of the input (dark blue) becomes row 1 of the output (dark green).",
                                          "The extra padding column (grey) shifts each row by one bank, so reading a",
                                          "column of the shared tile hits 32 different banks: no bank conflicts."]))


def fig_004_reduction(name):
    vals = [3, 1, 4, 1, 5, 9, 2, 6]
    return kit.reduce_tree(name, "Parallel reduction: partial sums combined as a balanced tree",
                           vals, lambda p, q: p + q, "thread sums", result_label="S = Σ x_i",
                           final_lines=["Each thread first sums many elements in a grid-stride loop (top row);",
                                        "lanes, warps and blocks then combine their partial sums. The tree has",
                                        "height ≈ log₂ N, so rounding error grows much more slowly than in a loop."])


def fig_005_softmax(name):
    s = Svg(name, W, 100, "Softmax with the max subtracted: every exponent is ≤ 0, so nothing overflows")
    xs = [1.0, 3.0, 2.0, 5.0, 0.0, 4.0]
    m = max(xs)
    e = [math.exp(v - m) for v in xs]
    tot = sum(e)
    p = [v / tot for v in e]
    x0, cw = 160, 64
    rows = [("x", [fmt(v) for v in xs], "a"), ("x − m", [fmt(v - m) for v in xs], "b"),
            ("e^{x − m}", [fmt(v, 3) for v in e], "d"), ("p = e / s", [fmt(v, 3) for v in p], "c")]
    y = 30
    for i, (nm, labs, role) in enumerate(rows):
        row_label(s, x0 - 14, y + 14, nm, role=role, bold=True)
        cells(s, x0, y, labs, cw=cw - 8, gap=8, role=role,
              fills=["f-c2" if (role == "c" and k == 3) else None for k in range(len(labs))])
        if i < len(rows) - 1:
            s.arrow(x0 + len(xs) * cw + 12, y + 14, x0 + len(xs) * cw + 12, y + 50, role="muted", sw=1)
        y += 50
    s.text(x0 + len(xs) * cw + 20, 62, f"m = max = {fmt(m)}", anchor="start", size="small", role="b")
    s.text(x0 + len(xs) * cw + 20, 112, "", anchor="start", size="small")
    s.text(x0 + len(xs) * cw + 20, 162, f"s = Σ = {fmt(tot, 3)}", anchor="start", size="small", role="d")
    return finish(s, note(s, 40, y + 16, ["The max and the sum are found in one pass by merging (m, s) pairs:",
                                          "(m₁, s₁) ⊕ (m₂, s₂) = (m, s₁·e^{m₁−m} + s₂·e^{m₂−m}),  m = max(m₁, m₂)."]))


def _tiles(i, j, size=4):
    return "f-a2" if (j // size) % 2 == 0 else "f-a"


def fig_006_softmax_attention(name):
    return kit.attention(name, "FlashAttention: a block of queries streams over key tiles, never storing S", 8, 12,
                         lambda i, j: True, weight=lambda i, j: "f-c2" if i in (2, 3) else _tiles(i, j),
                         legend=[("f-a2", "a", "key tile 0 and 2"), ("f-a", "a", "key tile 1"),
                                 ("f-c2", "c", "one query block")],
                         pipeline_steps=["S_{tile} = Q Kᵀ / √d", "m′, α = e^{m − m′}, ℓ′", "a ← α·a + P·V_{tile}"],
                         note_lines=["Per query row only (m, ℓ, a) live in registers; after the last tile O = a / ℓ.",
                                     "The M × N score matrix is never written to memory."])


def fig_007_color_inversion(name):
    before = [200, 30, 90, 255, 0, 128, 255, 255]
    after = [255 - v if i % 4 != 3 else v for i, v in enumerate(before)]
    s = kit.elementwise(name, "Colour inversion: R, G and B become 255 − v; alpha is left unchanged",
                        [("before", [str(v) for v in before], "a")], ("after", [str(v) for v in after], "c"),
                        "255 − v (A kept)", n_label="byte", warp_note=False,
                        note_lines=["Bytes 0–3 are pixel 0 (R, G, B, A) and bytes 4–7 pixel 1.",
                                    "One thread loads a whole pixel as one 32-bit word: v ^ 0x00FFFFFF",
                                    "flips R, G and B in a single instruction and keeps A."])
    return s


def fig_008_matrix_addition(name):
    s = Svg(name, W, 100, "Matrix addition on the flattened array: float4 groups plus a scalar tail")
    N, cw = 4, 34
    x0, y0 = 40, 44
    grp = lambda k: "f-hl" if k >= 12 else ("f-a2" if (k // 4) % 2 == 0 else "f-a")
    matrix(s, x0, y0, 3, 5, cw, fill_fn=lambda r, c: grp(r * 5 + c), role="a")
    for r in range(3):
        for c in range(5):
            s.text(x0 + c * cw + cw / 2, y0 + r * cw + cw / 2, str(r * 5 + c), size="tiny")
    s.text(x0 + 5 * cw / 2, y0 - 14, "3 × 5 matrix (15 floats)", size="small", role="a", bold=True)
    fx = x0 + 5 * cw + 60
    for k in range(15):
        s.box(fx + k * 28, y0 + 34, 26, 30, str(k), role="a" if k < 12 else "hl", fill=grp(k), size="tiny", rx=2)
    s.arrow(x0 + 5 * cw + 8, y0 + 49, fx - 6, y0 + 49, role="muted", sw=1.2)
    s.text(fx + 7 * 28, y0 + 14, "row-major memory order", size="small", role="muted")
    for g in range(3):
        s.brace_h(fx + g * 112, fx + g * 112 + 110, y0 + 76, "", role="a")
        s.text(fx + g * 112 + 55, y0 + 94, f"float4 #{g}", size="tiny", role="a")
    s.brace_h(fx + 336, fx + 418, y0 + 76, "", role="hl")
    s.text(fx + 377, y0 + 94, "tail R = 3", size="tiny", role="hl")
    return finish(s, note(s, 40, y0 + 3 * cw + 50,
                          ["N² = 4V + R. Thread t adds float4 group t with one 16-byte load from A, one from B and",
                           "one 16-byte store to C; the last R ≤ 3 elements are handled one by one."]))


def fig_009_1d_convolution(name):
    xs = [2, 1, 0, 3, 1, 2, 4, 1, 0, 2]
    w = [1, 0, -1]
    ys = [sum(xs[i + j] * w[j] for j in range(3)) for i in range(len(xs) - 2)]
    return kit.window_1d(name, "Valid 1-D convolution: y_i = Σ_{j} x_{i+j} · w_j", xs, ys, 3, out_index=3,
                         weights=[fmt(v) for v in w],
                         note_lines=["Output i needs inputs i … i+K−1. A block stages its T outputs' inputs plus a",
                                     "K−1 halo in shared memory once, so every input is read from DRAM about once."])


def fig_010_2d_convolution(name):
    return kit.window_2d(name, "Valid 2-D convolution: each output pixel is a K×K window of the input", 6, 8, 3,
                         out_rc=(1, 3), out_rows=4, out_cols=6, op_text="Y_{ij} = Σ_{m,n} X_{i+m, j+n} · w_{mn}",
                         note_lines=["A block computes a 32 × 32 output tile and stages the (32+K−1)² input window",
                                     "(tile plus halo) in shared memory; neighbouring outputs reuse the same pixels."])


def fig_011_3d_convolution(name):
    s = Svg(name, W, 100, "3-D convolution: a K_d × K_r × K_c box of voxels makes one output voxel")
    cw = 22
    x0, y0 = 40, 50
    for d in range(3):
        xx = x0 + d * 150
        matrix(s, xx, y0, 5, 5, cw,
               fill_fn=lambda r, c: "f-a2" if 1 <= r <= 3 and 2 <= c <= 4 else None, role="a")
        s.rect(xx + 2 * cw, y0 + cw, 3 * cw, 3 * cw, fill="fig-none", stroke="s-hl", sw=2)
        s.text(xx + 2.5 * cw, y0 - 14, f"depth slice z+{d}", size="small", role="a")
    ox = x0 + 3 * 150 + 30
    matrix(s, ox, y0 + cw, 3, 3, cw, fill_fn=lambda r, c: "f-c2" if (r, c) == (1, 2) else None, role="c")
    s.text(ox + 1.5 * cw, y0 + cw - 14, "output slice z", size="small", role="c")
    s.arrow(x0 + 2 * 150 + 5 * cw + 6, y0 + 2.5 * cw, ox + 2 * cw + 11 - 20, y0 + 2.5 * cw, role="hl", sw=1.3)
    return finish(s, note(s, 40, y0 + 5 * cw + 34,
                          ["Y[z, r, c] = Σ_{a,b,e} X[z+a, r+b, c+e] · w[a, b, e]  (here K = 3: 27 taps).",
                           "With at most 5³ = 125 taps the input slices stay in L1/L2, so one thread per",
                           "output voxel with coalesced reads along c is enough."]))


def fig_012_multi_head_attention(name):
    s = Svg(name, W, 100, "Multi-head attention: heads are column blocks of Q, K and V, addressed by stride")
    x0, y0, cw, ch = 60, 50, 22, 20
    roles = ["a", "b", "c", "d"]
    for mi, nm in enumerate(["Q", "K", "V"]):
        xx = x0 + mi * 200
        for r in range(6):
            for c in range(8):
                s.rect(xx + c * cw, y0 + r * ch, cw, ch, fill=f"f-{roles[c // 2]}", stroke="s-line", sw=0.6)
        s.rect(xx, y0, 8 * cw, 6 * ch, fill="fig-none", stroke="s-ink", sw=1.3)
        s.text(xx + 4 * cw, y0 - 14, f"{nm}  (N × d_model)", size="small", bold=True)
    for h in range(4):
        s.text(x0 + h * 2 * cw + cw, y0 + 6 * ch + 14, f"h{h}", size="tiny", role=roles[h])
    y1 = y0 + 6 * ch + 40
    for h in range(4):
        s.box(x0 + h * 150, y1, 130, 40, f"head {h}\nsoftmax(QKᵀ/√d_k)V", role=roles[h], size="tiny")
    s.box(x0, y1 + 64, 580, 30, "output = Concat(H₀, …, H₃): head i writes columns i·d_k … (i+1)·d_k − 1",
          role="ink", size="small")
    for h in range(4):
        s.arrow(x0 + h * 150 + 65, y1 + 42, x0 + h * 150 + 65, y1 + 62, role="muted", sw=1)
    return finish(s, note(s, 40, y1 + 126, ["Element (r, c) of head i is at r·d_model + i·d_k + c: a head is only a base",
                                            "offset and a row stride, so splitting and concatenating heads move no data."]))


def fig_013_histogramming(name):
    s = Svg(name, W, 100, "Histogram with privatisation: per-block shared-memory counts, then one merge")
    data = [2, 0, 3, 2, 1, 2, 0, 2, 3, 1, 2, 2]
    x0, y0 = 60, 40
    row_label(s, x0 - 10, y0 + 14, "x", role="a", bold=True)
    for i, v in enumerate(data):
        s.box(x0 + i * 32, y0, 28, 28, str(v), role="a" if i < 6 else "b", size="small", rx=2)
    s.brace_h(x0, x0 + 6 * 32 - 4, y0 + 40, "block 0", role="a")
    s.brace_h(x0 + 6 * 32, x0 + 12 * 32 - 4, y0 + 40, "block 1", role="b")
    hb = y0 + 80
    for g, (part, role) in enumerate(((data[:6], "a"), (data[6:], "b"))):
        cnt = [part.count(k) for k in range(4)]
        kit.bars(s, x0 + 10 + g * 192, hb, 60, cnt, 5, bw=28, gap=12, role=role, labels=[f"bin {k}" for k in range(4)])
    tot = [data.count(k) for k in range(4)]
    kit.bars(s, 500, hb, 60, tot, 7, bw=28, gap=12, role="c", labels=[f"bin {k}" for k in range(4)])
    s.text(80 + 70, hb + 94, "shared-memory atomics (block 0)", size="tiny", role="a")
    s.text(80 + 192 + 70, hb + 94, "(block 1)", size="tiny", role="b")
    s.text(500 + 76, hb + 94, "global h = Σ blocks", size="tiny", role="c")
    s.arrow(430, hb + 40, 490, hb + 40, role="c", sw=1.2)
    return finish(s, note(s, 40, hb + 124, ["Contention is confined to fast shared-memory atomics inside a block; each",
                                            "block then adds its B counts to global memory once (B atomics per block)."]))


def fig_014_multi_agent_sim(name):
    s = Svg(name, W, 100, "Boids alignment: agent i steers 5% towards the mean velocity of its neighbours")
    p = Plot(s, 40, 20, 360, 250, (0, 18), (0, 12.5))
    s.rect(p.x, p.y, p.w, p.h, fill="fig-panel", stroke="s-line", sw=0.6)
    agents = [(8, 6, 1.2, 0.4), (10, 8, 0.8, 1.0), (6, 4.5, 1.4, -0.2), (11, 4.5, 1.0, 0.6), (3, 10, -1, 0.5),
              (15, 2, 0.2, 1.2), (14, 10.5, -0.8, -0.6)]
    ci = 0
    cx, cy = p.px(agents[ci][0]), p.py(agents[ci][1])
    rpx = 5 * p.w / 18
    s.add(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rpx:.1f}" ry="{5 * p.h / 12.5:.1f}" class="f-a s-a" '
          f'stroke-width="1" stroke-dasharray="4 3" fill-opacity="0.5"/>')
    nb = []
    for k, (x, y, vx, vy) in enumerate(agents):
        near = k != ci and (x - agents[ci][0]) ** 2 + (y - agents[ci][1]) ** 2 < 25
        if near:
            nb.append((vx, vy))
        role = "hl" if k == ci else ("b" if near else "muted")
        s.circle(p.px(x), p.py(y), 5, fill=f"k-{role}")
        s.arrow(p.px(x), p.py(y), p.px(x + vx * 1.6), p.py(y + vy * 1.6), role=role, sw=1.4)
    s.text(cx + rpx - 20, cy - 5 * p.h / 12.5 + 12, "r = 5", size="small", role="a", plate=True)
    mvx = sum(v[0] for v in nb) / len(nb)
    mvy = sum(v[1] for v in nb) / len(nb)
    vx, vy = agents[ci][2], agents[ci][3]
    tx = 440
    lines = [("velocity update of agent i (red)", "ink"), (f"neighbours (orange): {len(nb)}", "b"),
             (f"v̄ = ({fmt(mvx)}, {fmt(mvy)})", "b"), (f"v = ({fmt(vx)}, {fmt(vy)})", "hl"),
             (f"v′ = v + 0.05 (v̄ − v) = ({fmt(vx + 0.05 * (mvx - vx), 3)}, {fmt(vy + 0.05 * (mvy - vy), 3)})", "c"),
             ("p′ = p + v′", "c")]
    for i, (t, role) in enumerate(lines):
        s.text(tx, 40 + i * 24, t, anchor="start", size="small", role=role, bold=(i == 0))
    return finish(s, note(s, tx, 200, ["All N threads read the old state;", "positions are staged in shared",
                                       "memory tile by tile (N-body style)."]))


def fig_015_sorting(name):
    s = Svg(name, W, 100, "Sorting floats with LSD radix sort: an order-preserving map from float bits to keys")
    vals = [-2.5, 3.0, -0.5, 1.0]
    import struct
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in vals]
    keys = [(b ^ 0xFFFFFFFF) if b >> 31 else (b | 0x80000000) for b in bits]
    x0, y0 = 40, 40
    hdr = ["float x", "bits(x)", "key f(bits)", "rank"]
    xs = [x0, x0 + 110, x0 + 250, x0 + 390]
    for h, xx in zip(hdr, xs):
        s.text(xx, y0, h, anchor="start", size="small", bold=True)
    order = sorted(range(4), key=lambda i: keys[i])
    for i, v in enumerate(vals):
        yy = y0 + 28 + i * 26
        s.text(xs[0], yy, fmt(v), anchor="start", size="small", role="a")
        s.text(xs[1], yy, f"0x{bits[i]:08X}", anchor="start", size="small", mono=True)
        s.text(xs[2], yy, f"0x{keys[i]:08X}", anchor="start", size="small", mono=True, role="c")
        s.text(xs[3], yy, str(order.index(i)), anchor="start", size="small", role="c")
    s.text(x0 + 470, y0 + 28, "negative: flip all bits", anchor="start", size="small", role="b")
    s.text(x0 + 470, y0 + 52, "non-negative: set the sign bit", anchor="start", size="small", role="b")
    y1 = y0 + 150
    steps = [("digit 0 (bits 0–7)", "d"), ("digit 1", "d"), ("digit 2", "d"), ("digit 3 (top)", "d")]
    kit.pipeline(s, x0, y1, steps, w=140, h=32, gap=26)
    s.text(x0, y1 - 14, "4 stable counting-sort passes: histogram per tile → scan → scatter", anchor="start",
           size="small", role="muted")
    return finish(s, note(s, x0, y1 + 58, ["Unsigned key order equals float order, so four 8-bit passes sort the data;",
                                           "stability keeps the order set by the lower digits."]))


def fig_016_prefix_sum(name):
    xs = [3, 1, 4, 1, 5, 9, 2, 6]
    ys, acc = [], 0
    for v in xs:
        acc += v
        ys.append(acc)
    return kit.scan(name, "Inclusive prefix sum: y_i adds up every input up to and including x_i", xs, ys,
                    "y_i = x_0 + … + x_i", highlight=(0, 5, 5),
                    chunk=(4, ["chunk 0: offset 0", "chunk 1: offset 9"]),
                    note_lines=["Reduce-then-scan: each chunk scans itself locally, then adds the total of all",
                                "earlier chunks (its offset). Here y_5 = 9 + (5 + 9) = 23."])


def fig_017_dot_product(name):
    a = [1, 2, -1, 3, 0.5, 2, -2, 1]
    b = [2, 1, 3, 1, 4, -1, 1, 2]
    return kit.reduce_tree(name, "Dot product: multiply pairwise (fused multiply-add), then reduce as a tree",
                           [x * y for x, y in zip(a, b)], lambda p, q: p + q, "a_i · b_i",
                           map_row=("a_i, b_i", [f"{fmt(x)}·{fmt(y)}" if y >= 0 else f"{fmt(x)}·({fmt(y)})" for x, y in zip(a, b)]),
                           result_label="s = a · b",
                           final_lines=["Each thread accumulates acc = fma(a_i, b_i, acc) over its grid-stride slice:",
                                        "one instruction and one rounding per element, then the usual tree reduction."])


def fig_018_sparse_matrix_vector_multiplication(name):
    s = Svg(name, W, 100, "Dense-stored sparse GEMV: one warp per row streams the row and reduces with shuffles")
    import random
    rng = random.Random(18)
    R, C, cw = 6, 12, 28
    A = [[(rng.randint(1, 9) if rng.random() < 0.35 else 0) for _ in range(C)] for _ in range(R)]
    x = [rng.randint(1, 3) for _ in range(C)]
    x0, y0 = 90, 70
    for c in range(C):
        s.box(x0 + c * cw, 30, cw - 2, 24, str(x[c]), role="b", size="tiny", rx=2)
    row_label(s, x0 - 10, 42, "x", role="b", bold=True)
    for r in range(R):
        for c in range(C):
            v = A[r][c]
            f = ("f-a2" if v else "f-a") if r == 2 else ("fig-panel" if not v else None)
            s.box(x0 + c * cw, y0 + r * cw, cw - 2, cw - 2, str(v), role="a" if v or r == 2 else "line", fill=f,
                  size="tiny", rx=1, text_role=None if v else "muted")
    row_label(s, x0 - 10, y0 + 2 * cw + 13, "warp 2", role="a", bold=True)
    y = [sum(A[r][c] * x[c] for c in range(C)) for r in range(R)]
    yx = x0 + C * cw + 70
    for r in range(R):
        s.box(yx, y0 + r * cw, 40, cw - 2, str(y[r]), role="c", fill="f-c2" if r == 2 else None, size="tiny", rx=2)
    s.text(yx + 20, y0 - 14, "y = A x", size="small", role="c", bold=True)
    s.arrow(x0 + C * cw + 6, y0 + 2 * cw + 13, yx - 6, y0 + 2 * cw + 13, role="a", sw=1.2)
    return finish(s, note(s, 40, y0 + R * cw + 30,
                          ["Zeros (grey) are stored too: without an index structure every byte of A must be read,",
                           "so the kernel is bandwidth-bound. Lanes stride along the row with float4 loads and",
                           "combine their partial sums with __shfl_down_sync."]))


def fig_019_reverse_array(name):
    s = Svg(name, W, 100, "In-place reversal: thread i swaps x[i] with its mirror x[N−1−i]")
    vals = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    n, x0, cw, y0 = len(vals), 120, 56, 90
    for i in range(n):
        s.text(x0 + i * cw + 24, y0 - 60, str(i), size="small", role="muted")
    row_label(s, x0 - 14, y0 + 14, "before", role="a", bold=True)
    cells(s, x0, y0, vals, cw=48, gap=8, role="a", fills=["f-hl" if i == n // 2 else None for i in range(n)])
    for i in range(n // 2):
        j = n - 1 - i
        xa, xb = x0 + i * cw + 24, x0 + j * cw + 24
        top = y0 - 8 - 9 * (n // 2 - i)
        s.path(f"M{xa},{y0 - 2} C{xa},{top} {xb},{top} {xb},{y0 - 2}", stroke="s-b", sw=1.2, arrow="b", both=True)
    y1 = y0 + 70
    row_label(s, x0 - 14, y1 + 14, "after", role="c", bold=True)
    cells(s, x0, y1, vals[::-1], cw=48, gap=8, role="c", fills=["f-hl" if i == n // 2 else None for i in range(n)])
    return finish(s, note(s, 40, y1 + 58, ["Only ⌊N/2⌋ threads run and each owns one pair, so no two threads touch the",
                                           "same element: no race and no second buffer. The middle element (odd N) stays."]))


def fig_020_kmeans_clustering(name):
    s = Svg(name, W, 100, "Lloyd's k-means: assign every point to its nearest centroid, then move the centroids")
    import random
    rng = random.Random(20)
    centres = [(3, 3), (8, 7), (12, 3)]
    pts = []
    for cx, cy in centres:
        for _ in range(7):
            pts.append((cx + rng.uniform(-2, 2), cy + rng.uniform(-1.8, 1.8)))
    init = [(5, 5), (7, 9), (10, 1)]
    roles = ["a", "b", "c"]
    for panel, (title, cents) in enumerate((("1. assign: label = nearest centroid", init),
                                            ("2. update: centroid = mean of its points", None))):
        p = Plot(s, 30 + panel * 350, 40, 320, 200, (0, 15), (0, 10))
        s.rect(p.x, p.y, p.w, p.h, fill="fig-panel", stroke="s-line", sw=0.6)
        s.text(p.x + p.w / 2, 24, title, size="small", bold=True)
        lab = [min(range(3), key=lambda c: (x - init[c][0]) ** 2 + (y - init[c][1]) ** 2) for x, y in pts]
        for (x, y), l in zip(pts, lab):
            s.circle(p.px(x), p.py(y), 4, fill=f"k-{roles[l]}")
        new = []
        for c in range(3):
            mem = [pts[i] for i in range(len(pts)) if lab[i] == c]
            new.append((sum(q[0] for q in mem) / len(mem), sum(q[1] for q in mem) / len(mem)))
        for c in range(3):
            ox, oy = init[c]
            if panel == 0:
                s.rect(p.px(ox) - 7, p.py(oy) - 7, 14, 14, fill=f"f-{roles[c]}2", stroke="s-ink", sw=1.4)
            else:
                nx, ny = new[c]
                s.rect(p.px(ox) - 6, p.py(oy) - 6, 12, 12, fill="fig-paper", stroke="s-muted", sw=1, extra=' stroke-dasharray="3 2"')
                s.arrow(p.px(ox), p.py(oy), p.px(nx), p.py(ny), role="ink", sw=1.3)
                s.rect(p.px(nx) - 7, p.py(ny) - 7, 14, 14, fill=f"f-{roles[c]}2", stroke="s-ink", sw=1.4)
    return finish(s, note(s, 40, 272, ["Assignment: one thread per point, centroids in shared memory. Update: sums and",
                                       "counts per cluster via block-private accumulators and atomics; repeat T times."]))


# ----- 021-040 -----------------------------------------------------------------------------------------------

def fig_021_relu(name):
    return kit.function_plot(name, "ReLU keeps positive inputs and replaces negative ones with 0",
                             [(lambda x: max(0.0, x), "a", None)], (-3, 3), (-1, 3), [-3, -2, -1, 0, 1, 2, 3],
                             [-1, 0, 1, 2, 3], ["y = max(0, x)", "one thread per element:", "one load, one max, one store"],
                             points=[(-2, 0, "b"), (1.5, 1.5, "hl")],
                             side_lines=["x = −2 → 0 (orange dot)", "x = 1.5 → 1.5 (red dot)",
                                         "", "Same memory pattern as vector", "addition with one input stream."])


def fig_022_gemm(name):
    return kit.gemm(name, "FP16 GEMM on tensor cores: each warp owns 16 × 16 fragments of C", cw=18, m=8, n=8, k=8,
                    tile=(1, 2), kstep=2,
                    epilogue=["per warp: wmma::mma_sync", "D = A_f · B_f + C_f", "A_f, B_f: fp16 16 × 16",
                              "C_f, D: fp32 accumulators", "epilogue: C ← α·D + β·C", "rounded once to fp16"],
                    note_lines=["Each highlighted 2 × 2 cell block stands for one 16 × 16 fragment; K advances 16 at a time."])


def fig_023_leaky_relu(name):
    return kit.function_plot(name, "Leaky ReLU keeps a small slope α = 0.01 for negative inputs",
                             [(lambda x: max(0.0, x), "line", "4 3"), (lambda x: x if x > 0 else 0.01 * x, "a", None)],
                             (-40, 10), (-0.6, 10), [-40, -30, -20, -10, 0, 10], [0, 5, 10],
                             ["y = x (x > 0),  y = α x (x ≤ 0)", "α = 0.01, i.e. y = max(x, α x)"],
                             points=[(-40, -0.4, "hl")],
                             legend=[("f-a2", "a", "Leaky ReLU"), ("fig-panel", "line", "ReLU (dashed)")],
                             side_lines=["x = −40 → −0.4 (red dot): the", "gradient never becomes exactly 0,",
                                         "so units cannot \"die\"."])


def fig_024_rainbow_table(name):
    s = Svg(name, W, 100, "Rainbow-table chain: R rounds of 32-bit FNV-1a, one thread per input word")
    x0, y0 = 30, 40
    steps = [("x_i", "a"), ("H", "d"), ("H", "d"), ("…", "muted"), ("H", "d"), ("y_i", "c")]
    for i, (lab, role) in enumerate(steps):
        bx = x0 + i * 115
        s.box(bx, y0, 80, 34, lab, role=role, size="small", bold=True)
        if i:
            s.arrow(bx - 33, y0 + 17, bx - 2, y0 + 17, role="muted", sw=1.1)
    s.brace_h(x0 + 115, x0 + 4 * 115 + 80, y0 + 48, "R rounds (R ≤ 100)", role="d")
    y1 = y0 + 100
    s.text(x0, y1 - 14, "one round H(x): consume the 4 bytes of x, least significant first", anchor="start",
           size="small", bold=True)
    bs = [("h = 0x811C9DC5", "ink")] + [(f"h ^= byte {b}\nh *= P", "b") for b in range(4)]
    for i, (lab, role) in enumerate(bs):
        bx = x0 + i * 134
        s.box(bx, y1, 120, 40, lab, role=role, size="tiny", mono=True)
        if i:
            s.arrow(bx - 12, y1 + 17, bx - 2, y1 + 17, role="muted", sw=1)
    return finish(s, note(s, x0, y1 + 68, ["P = 16777619 (0x01000193), all arithmetic mod 2³². 8 bytes of memory traffic",
                                           "buy up to 800 integer operations per thread: compute-bound, not memory-bound."]))


def fig_025_categorical_cross_entropy_loss(name):
    s = Svg(name, W, 100, "Cross-entropy: per row, log-sum-exp of the logits minus the logit of the true class")
    Z = [[2.0, 0.5, -1.0, 1.0], [0.0, 3.0, 1.0, -2.0], [1.0, 1.0, 1.0, 1.0]]
    y = [0, 2, 3]
    x0, y0, cw = 130, 50, 58
    for c in range(4):
        s.text(x0 + c * cw + cw / 2 - 3, y0 - 14, f"class {c}", size="tiny", role="muted")
    losses = []
    for r in range(3):
        row_label(s, x0 - 12, y0 + r * 36 + 14, f"sample {r}", role="muted")
        cells(s, x0, y0 + r * 36, [fmt(v) for v in Z[r]], cw=cw - 6, gap=6, role="a",
              fills=["f-hl" if c == y[r] else None for c in range(4)])
        m = max(Z[r])
        lse = m + math.log(sum(math.exp(v - m) for v in Z[r]))
        l = lse - Z[r][y[r]]
        losses.append(l)
        s.arrow(x0 + 4 * cw + 6, y0 + r * 36 + 14, x0 + 4 * cw + 40, y0 + r * 36 + 14, role="muted", sw=1)
        s.text(x0 + 4 * cw + 48, y0 + r * 36 + 14, f"ℓ = {fmt(lse, 3)} − {fmt(Z[r][y[r]])} = {fmt(l, 3)}",
               anchor="start", size="small", role="b")
    s.text(x0 + 4 * cw + 48, y0 - 14, "ℓ_{j} = LSE(z_j) − z_{j, y_j}", anchor="start", size="small", bold=True)
    yb = y0 + 3 * 36 + 20
    s.text(x0, yb, f"L = mean ℓ_{{j}} = {fmt(sum(losses) / 3, 3)}", anchor="start", size="small", role="c", bold=True)
    s.legend(x0 + 260, yb, [("f-hl", "hl", "logit of the true class y_j")])
    return finish(s, note(s, 40, yb + 34, ["One warp per row computes LSE = m + log Σ e^{z − m} in one pass with the online",
                                           "(m, s) merge; the per-row losses are then summed and divided by N."]))


def fig_026_multi_head_cross_attention(name):
    return kit.attention(name, "Cross-attention: M decoder queries attend to all N encoder keys, no mask", 6, 12,
                         lambda i, j: True, hl_row=2,
                         legend=[("f-a2", "a", "every score is visible"), ("f-c2", "c", "query 2 sees all N keys")],
                         pipeline_steps=["s = q·k / √D (per head h)", "softmax over j", "O_{i,h} = Σ p · V_{j,h}"],
                         note_lines=["Q is (M, H, D) and K, V are (N, H, D): head h is base offset h·D with row stride H·D,",
                                     "so the transposes of the reference cost nothing. M ≠ N in general."])


def fig_027_mean_squared_error(name):
    p = [1.0, 2.5, 0.0, 4.0, 3.0, 1.5, 2.0, 0.5]
    t = [1.5, 2.0, 1.0, 3.0, 3.0, 0.5, 2.5, 0.0]
    sq = [(a - b) ** 2 for a, b in zip(p, t)]
    return kit.reduce_tree(name, "Mean squared error: square each difference, reduce, divide by N",
                           sq, lambda a, b: a + b, "(p − t)²",
                           map_row=("p, t", [f"{fmt(a)}, {fmt(b)}" for a, b in zip(p, t)]),
                           tail=("÷ N", fmt(sum(sq) / len(sq), 4)), value_fmt=lambda v: fmt(v, 3),
                           result_label="MSE",
                           final_lines=["Per-thread partial sums stay in float32 (a few hundred terms each); the",
                                        "levels across threads and blocks accumulate in float64 to avoid swamping."])


def fig_028_gaussian_blur(name):
    return kit.window_2d(name, "\"Same\" blur with zero padding: windows near the edge read padded zeros", 5, 7, 3,
                         out_rc=(0, 0), pad=1, out_rows=5, out_cols=7,
                         op_text="Y_{ij} = Σ w_{mn} · X̃_{i+m−h, j+n−h}",
                         pad_note=[("fig-panel", "line", "zero padding"), ("f-hl", "hl", "tap on padding (= 0)")],
                         note_lines=["The output has the input's size. Output (0, 0) uses a 3 × 3 window centred on",
                                     "pixel (0, 0); its five taps outside the image read 0. Weights sum to 1."])


def fig_029_top_k_selection(name):
    s = Svg(name, W, 100, "Top-k by radix select: find the k-th largest value τ, keep everything above it")
    vals = [3.1, 7.4, 1.2, 9.0, 5.5, 7.4, 2.2, 8.1, 4.0, 6.3]
    k = 4
    tau = sorted(vals, reverse=True)[k - 1]
    x0, y0, h = 60, 40, 150
    fills = ["f-c2" if v > tau else ("f-b2" if v == tau else "f-a") for v in vals]
    kit.bars(s, x0, y0, h, vals, 10, bw=34, gap=12, fills=fills, labels=[str(i) for i in range(len(vals))])
    ty = y0 + h - h * tau / 10
    s.line(x0 - 10, ty, x0 + len(vals) * 46, ty, stroke="s-b", sw=1.4, dash="5 4")
    s.text(x0 + len(vals) * 46 + 8, ty, f"τ = {fmt(tau)}", anchor="start", size="small", role="b", bold=True)
    tx = x0 + len(vals) * 46 + 8
    note(s, tx, y0 + 104, [f"k = {k}", "green: x > τ", "orange: x = τ", "(ties fill the rest)"], role="ink")
    out = sorted(vals, reverse=True)[:k]
    yb = y0 + h + 44
    row_label(s, x0 + 90, yb + 14, "output", role="c", bold=True)
    cells(s, x0 + 104, yb, [fmt(v) for v in out], cw=48, gap=8, role="c")
    return finish(s, note(s, 40, yb + 58, ["Floats are mapped to order-preserving unsigned keys; four 8-bit histogram passes",
                                           "fix τ digit by digit (read-only), then only the k survivors are sorted."]))


def fig_030_batched_matrix_multiplication(name):
    return kit.gemm(name, "Batched GEMM: the batch index is simply a third grid dimension", batch=2,
                    epilogue=["grid = (N/64, M/64, B)", "blockIdx.z = b selects", "A_b = A + b·M·K",
                              "B_b = B + b·K·N", "C_b = C + b·M·N"],
                    note_lines=["Every batch entry is an independent GEMM with the same tiled kernel."])


def fig_031_matrix_copy(name):
    s = Svg(name, W, 100, "Matrix copy at peak bandwidth: 16-byte vector loads and stores, fully coalesced")
    x0, y0 = 90, 40
    row_label(s, x0 - 10, y0 + 14, "A", role="a", bold=True)
    row_label(s, x0 - 10, y0 + 94, "B", role="c", bold=True)
    for k in range(16):
        f = "f-a2" if (k // 4) % 2 == 0 else "f-a"
        s.box(x0 + k * 34, y0, 32, 28, str(k), role="a", fill=f, size="tiny", rx=2)
        s.box(x0 + k * 34, y0 + 80, 32, 28, str(k), role="c", fill="f-c2" if (k // 4) % 2 == 0 else "f-c",
              size="tiny", rx=2)
    for g in range(4):
        cx = x0 + g * 136 + 66
        s.arrow(cx, y0 + 30, cx, y0 + 78, role="muted", sw=1.2)
        s.text(cx + 6, y0 + 54, f"thread {g}", anchor="start", size="tiny", role="muted")
    return finish(s, note(s, 40, y0 + 140, ["Each thread moves one float4 (16 bytes); a warp moves 512 contiguous bytes per",
                                            "instruction. There is no arithmetic: the achieved GB/s is the ceiling for every",
                                            "memory-bound kernel on this GPU."]))


def fig_032_int8_quantized_matmul(name):
    return kit.gemm(name, "INT8 GEMM: exact integer accumulation, then a float requantisation epilogue",
                    epilogue=["S_{ij} = Σ_{k} (A_{ik} − z_A)(B_{kj} − z_B)", "(int32, exact; zero points folded", " out as row/column sums)",
                              "C = clamp(rne(S·s_A·s_B / s_C)", "      + z_C, −128, 127)"],
                    note_lines=["The int8 × int8 products run on integer tensor cores; only the epilogue uses float32,",
                                "in the reference's exact operation order, so the result is bit-exact."])


def fig_033_ordinary_least_squares(name):
    s = Svg(name, W, 100, "Least squares via the normal equations: Gram matrix, Cholesky, two triangular solves")
    x0, y0 = 20, 40
    steps = [("X, y", "a"), ("G = XᵀX\nb = Xᵀy", "b"), ("G = L Lᵀ\n(Cholesky)", "d"), ("L z = b\nforward", "c"),
             ("Lᵀ β = z\nbackward", "c"), ("β", "c")]
    kit.pipeline(s, x0, y0, steps, w=96, h=46, gap=20)
    y1 = y0 + 96
    cw = 16
    mats = [("G (f × f)", lambda r, c: "f-b2"), ("L", lambda r, c: "f-d2" if c <= r else "fig-panel"),
            ("Lᵀ", lambda r, c: "f-c2" if c >= r else "fig-panel")]
    for i, (lab, fn) in enumerate(mats):
        xx = x0 + 130 + i * 170
        matrix(s, xx, y1, 5, 5, cw, fill_fn=fn, role="ink")
        s.text(xx + 40, y1 + 5 * cw + 14, lab, size="small")
    return finish(s, note(s, x0, y1 + 5 * cw + 46,
                          ["G is symmetric positive definite for full-rank X. The Gram matrix is GEMM-like and fully",
                           "parallel; Cholesky is sequential in its column k but parallel within each step."]))


def fig_034_logistic_regression(name):
    s = Svg(name, W, 100, "Logistic regression by Newton's method (IRLS): repeat until the step is tiny")
    x0, y0 = 40, 50
    steps = [("p = σ(Xβ)", "a"), ("g = Xᵀ(p − y)\n+ λβ", "b"), ("H = XᵀWX\n+ λI", "b"), ("solve H Δ = g", "d"),
             ("β ← β − Δ", "c")]
    end = kit.pipeline(s, x0, y0, steps, w=110, h=40, gap=20)
    s.path(f"M{end - 55:.1f},{y0 + 40:.1f} L{end - 55:.1f},{y0 + 70:.1f} L{x0 + 55:.1f},{y0 + 70:.1f} "
           f"L{x0 + 55:.1f},{y0 + 42:.1f}", stroke="s-hl", sw=1.3, arrow="hl")
    s.text((x0 + end) / 2, y0 + 84, "until ‖Δ‖ < 10⁻⁸", size="small", role="hl")
    p = Plot(s, 90, y0 + 120, 260, 120, (-6, 6), (0, 1))
    p.axes([-6, 0, 6], [0, 0.5, 1], "xᵀβ", "p")
    p.curve(sigmoid, role="a")
    note(s, 400, y0 + 140, ["W = diag(max(p(1 − p), 10⁻⁸)).", "Each iteration is two GEMM-like",
                            "reductions over the n samples and one", "small f × f Cholesky solve."])
    return finish(s, p.y + p.h + 36)


def fig_035_monte_carlo_integration(name):
    s = Svg(name, W, 100, "Monte Carlo integration: (b − a) times the mean of the sampled function values")
    import random
    rng = random.Random(35)
    f = lambda x: 1 + 0.8 * math.sin(x) + 0.1 * x
    a, b = 0.0, 6.0
    p = Plot(s, 70, 30, 360, 190, (0, 6), (0, 2.5))
    p.axes([0, 2, 4, 6], [0, 1, 2], "x", "f(x)")
    pts = " L".join(f"{p.px(a + (b - a) * t / 100):.1f},{p.py(f(a + (b - a) * t / 100)):.1f}" for t in range(101))
    s.path(f"M{p.px(a):.1f},{p.py(0):.1f} L{pts} L{p.px(b):.1f},{p.py(0):.1f} Z", stroke="s-a", fill="f-a", sw=0)
    p.curve(f, role="a")
    xs = [rng.uniform(a, b) for _ in range(12)]
    for x in xs:
        s.line(p.px(x), p.py(0), p.px(x), p.py(f(x)), stroke="s-b", sw=0.9, dash="2 2")
        p.dot(x, f(x), role="b", r=3.5)
    mean = sum(f(x) for x in xs) / len(xs)
    s.line(p.px(0), p.py(mean), p.px(6), p.py(mean), stroke="s-c", sw=1.6, dash="6 3")
    tx = 460
    note(s, tx, 50, ["samples y_i = f(x_i), x_i ~ U[a, b]", f"mean ȳ = {fmt(mean, 3)} (green line)",
                     f"Î = (b − a) · ȳ = {fmt((b - a) * mean, 3)}"], role="ink")
    note(s, tx, 128, ["The GPU work is only the mean:", "a sum reduction, then one multiply.",
                      "Error ∝ σ_{f} / √n: 100× more", "samples → 10× more accurate."])
    return finish(s, p.y + p.h + 36)


def fig_036_radix_sort(name):
    s = Svg(name, W, 100, "One LSD radix-sort pass: count digits, scan the counts, scatter stably")
    keys = [0x2B, 0x13, 0x21, 0x3A, 0x11, 0x2C, 0x03, 0x32]
    digit = lambda k: k >> 4  # the digit of this pass (top nibble, for the example)
    x0, y0 = 90, 40
    roles = ["a", "b", "c", "d"]
    row_label(s, x0 - 12, y0 + 14, "keys", bold=True)
    for i, k in enumerate(keys):
        s.box(x0 + i * 56, y0, 50, 28, f"{k:02X}", role=roles[digit(k)], size="small", mono=True, rx=2)
    cnt = [sum(1 for k in keys if digit(k) == d) for d in range(4)]
    off = [sum(cnt[:d]) for d in range(4)]
    y1 = y0 + 60
    row_label(s, x0 - 12, y1 + 14, "count", bold=True)
    row_label(s, x0 - 12, y1 + 48, "offset", bold=True)
    for d in range(4):
        s.box(x0 + d * 80, y1, 72, 28, f"digit {d}: {cnt[d]}", role=roles[d], size="tiny", rx=2)
        s.box(x0 + d * 80, y1 + 34, 72, 28, str(off[d]), role=roles[d], size="small", rx=2)
    s.text(x0 + 4 * 80 + 10, y1 + 48, "← exclusive scan of the counts", anchor="start", size="small", role="muted")
    out = sorted(keys, key=digit)
    y2 = y1 + 100
    row_label(s, x0 - 12, y2 + 14, "output", bold=True)
    for i, k in enumerate(out):
        s.box(x0 + i * 56, y2, 50, 28, f"{k:02X}", role=roles[digit(k)], size="small", mono=True, rx=2)
    return finish(s, note(s, 40, y2 + 56, ["The example sorts by the high hex digit. A key goes to offset[d] + (its rank among",
                                           "equal digits); equal digits keep their order (2B before 21 before 2C): stable.",
                                           "The real kernel uses 8-bit digits, tiles of 2048 keys and 4 passes."]))


def fig_037_matrix_power(name):
    s = Svg(name, W, 100, "Matrix power by repeated squaring: A²⁰ with 5 GEMMs instead of 19")
    x0, y0 = 40, 60
    pw = ["A", "A²", "A⁴", "A⁸", "A¹⁶"]
    bits = [0, 0, 1, 0, 1]
    for i, lab in enumerate(pw):
        bx = x0 + i * 120
        s.box(bx, y0, 80, 36, lab, role="c" if bits[i] else "a", size=None, bold=True,
              fill="f-c2" if bits[i] else None)
        s.text(bx + 40, y0 - 16, f"bit {i} = {bits[i]}", size="small", role="c" if bits[i] else "muted")
        if i:
            s.arrow(bx - 38, y0 + 18, bx - 2, y0 + 18, role="a", sw=1.2)
            s.text(bx - 20, y0 + 32, "²", size="small", role="a")
    s.text(x0 + 600, y0 + 18, "", size="small")
    y1 = y0 + 90
    s.box(x0 + 240, y1, 200, 36, "A²⁰ = A⁴ · A¹⁶", role="c", fill="f-c2", bold=True)
    s.arrow(x0 + 280, y0 + 38, x0 + 300, y1 - 2, role="c", sw=1.2)
    s.arrow(x0 + 520, y0 + 38, x0 + 400, y1 - 2, role="c", sw=1.2)
    return finish(s, note(s, 40, y1 + 66, ["20 = 10100₂: 4 squarings + popcount(20) − 1 = 1 extra multiply = 5 GEMMs.",
                                           "The multiplication order mirrors torch.linalg.matrix_power exactly, so float32",
                                           "rounding matches the reference despite the huge dynamic range of A²⁰."]))


def fig_038_nearest_neighbor(name):
    s = Svg(name, W, 100, "Nearest neighbour: every point scans all others and keeps the smallest distance")
    import random
    rng = random.Random(38)
    pts = [(rng.uniform(1, 13), rng.uniform(1, 9)) for _ in range(11)]
    p = Plot(s, 40, 30, 380, 240, (0, 14), (0, 10))
    s.rect(p.x, p.y, p.w, p.h, fill="fig-panel", stroke="s-line", sw=0.6)
    for i, (x, y) in enumerate(pts):
        j = min((k for k in range(len(pts)) if k != i), key=lambda k: (pts[k][0] - x) ** 2 + (pts[k][1] - y) ** 2)
        s.arrow(p.px(x), p.py(y), p.px(x + 0.85 * (pts[j][0] - x)), p.py(y + 0.85 * (pts[j][1] - y)),
                role="b", sw=1.1)
    for x, y in pts:
        s.circle(p.px(x), p.py(y), 5, fill="k-a")
    note(s, 450, 50, ["nn(i) = argmin_{j ≠ i} d_{ij} (ties →", "smallest j), d = Δx² + Δy² + Δz².",
                                       "", "Exact match with PyTorch: each of", "Δx², Δy², Δz² is rounded separately",
                                       "and summed left to right (no FMA", "contraction). Blocks tile the O(N²)",
                                       "loop through shared memory."], role="ink")
    return finish(s, p.y + p.h)


def fig_039_fast_fourier_transform(name):
    s = Svg(name, W, 100, "Radix-2 FFT: log₂N stages of butterflies X_k = E_k ± ω^k O_k (N = 8 shown)")
    x0, y0, dx, dy = 70, 40, 150, 30
    N = 8
    for st in range(4):
        for i in range(N):
            s.circle(x0 + st * dx, y0 + i * dy, 4, fill="k-a" if st < 3 else "k-c")
    for i in range(N):
        s.text(x0 - 14, y0 + i * dy, f"x{i}", anchor="end", size="small", role="a")
        s.text(x0 + 3 * dx + 14, y0 + i * dy, f"X{i}", anchor="start", size="small", role="c")
    for st in range(3):
        half = N >> (st + 1)
        for i in range(N):
            j = i ^ half
            s.line(x0 + st * dx + 5, y0 + i * dy, x0 + (st + 1) * dx - 5, y0 + j * dy,
                   stroke="s-b" if i & half else "s-muted", sw=1)
            s.line(x0 + st * dx + 5, y0 + i * dy, x0 + (st + 1) * dx - 5, y0 + i * dy, stroke="s-muted", sw=1)
        s.text(x0 + st * dx + dx / 2, y0 + N * dy + 4, f"stage {st + 1}", size="small", role="muted")
    return finish(s, note(s, 40, y0 + N * dy + 36, [
        "Stockham form: every stage reads one buffer and writes the other in natural order, so no bit-reversal",
        "pass is needed. Arbitrary N: Bluestein's algorithm turns the DFT into a power-of-two convolution."]))


def fig_040_batch_normalization(name):
    return kit.normalize(name, "BatchNorm: statistics per column (channel), taken over the batch", 6, 6,
                         lambda r, c: c, 2, ["channel j = 2 (column)", "μ_{j} = mean over N rows", "σ_{j}² = biased variance",
                                              "Welford + Chan merge"],
                         ["y_{ij} = γ_{j} (x_{ij} − μ_{j})", "  / √(σ_{j}² + ε) + β_{j}"],
                         col_names="C channels →", row_names="N rows",
                         note_lines=["Threads of a warp take consecutive columns, so every row read is coalesced while",
                                     "each thread walks down its own column; partial (n, mean, M2) states merge exactly."])


# ----- 041-060 -----------------------------------------------------------------------------------------------

def fig_041_simple_inference(name):
    return kit.gemm(name, "A linear layer is one GEMM with a bias epilogue: Y = X Wᵀ + b",
                    a_label="X", b_label="Wᵀ", c_label="Y", dims=("B", "d_out", "d_in"),
                    epilogue=["torch.addmm(b, X, W.T, out=Y)", "one library call: GEMM", "with the bias add fused",
                              "and written straight into", "the output buffer"],
                    note_lines=["W is stored [out, in] (PyTorch layout), so the GEMM reads it transposed for free."])


def fig_042_2d_max_pooling(name):
    return kit.window_2d(name, "2-D max pooling: k × k windows moved by the stride; padding never wins", 6, 6, 3,
                         out_rc=(1, 2), stride=2, pad=1, out_rows=3, out_cols=3, op_text="Y = max over the window",
                         pad_note=[("fig-panel", "line", "padding (acts as −∞)")],
                         note_lines=["Here k = 3, s = 2, p = 1: H_o = ⌊(6 + 2 − 3)/2⌋ + 1 = 3. Output (1, 2) reads input",
                                     "rows 1 … 3 and columns 3 … 5; one thread per output element, per (n, c) plane."])


def fig_043_count_array_element(name):
    xs = [4, 7, 4, 1, 4, 9, 2, 4]
    K = 4
    return kit.reduce_tree(name, "Counting equal elements: compare, then an exact integer reduction",
                           [1 if v == K else 0 for v in xs], lambda a, b: a + b, "[x_i = K]",
                           map_row=("x (K = 4)", [str(v) for v in xs]), result_label="count",
                           final_lines=["Integer addition is exact and associative, so any order (and atomics) gives the",
                                        "same result. A warp sums its flags with one __reduce_add_sync instruction."])


def _count_grid(name, title, dims_note, grid_title="matrix (row-major)", slices=False):
    s = Svg(name, W, 100, title)
    import random
    rng = random.Random(44)
    R, C, cw = 4, 8, 30
    vals = [[rng.choice([1, 1, 2, 3, 5]) for _ in range(C)] for _ in range(R)]
    x0, y0 = 40, 50
    gap = lambda r: (12 if slices and r >= 2 else 0)
    for r in range(R):
        for c in range(C):
            v = vals[r][c]
            s.box(x0 + c * cw, y0 + r * cw + gap(r), cw, cw, str(v), role="c" if v == 1 else "line",
                  fill="f-c2" if v == 1 else "fig-paper", size="tiny", rx=0)
    if slices:
        for k in range(2):
            s.rect(x0, y0 + k * (2 * cw + 12), C * cw, 2 * cw, fill="fig-none", stroke="s-ink", sw=1.4)
            s.text(x0 + C * cw + 8, y0 + k * (2 * cw + 12) + cw, f"a = {k}", anchor="start", size="tiny",
                   role="muted")
    else:
        s.rect(x0, y0, C * cw, R * cw, fill="fig-none", stroke="s-ink", sw=1.4)
    s.text(x0 + C * cw / 2, y0 - 14, grid_title, size="small", bold=True)
    fx = x0 + C * cw + 60
    flat = [v for row in vals for v in row]
    s.arrow(x0 + C * cw + 8, y0 + 2 * cw, fx - 8, y0 + 2 * cw, role="muted", sw=1.2)
    for i in range(16):
        v = flat[i]
        s.box(fx + i * 20, y0 + 2 * cw - 12, 20, 24, str(v), role="c" if v == 1 else "line",
              fill="f-c2" if v == 1 else "fig-paper", size="tiny", rx=0)
    s.text(fx + 160, y0 + 2 * cw - 30, "same bytes, flat index i", size="small", role="muted")
    s.text(fx + 160, y0 + 2 * cw + 30, f"count(K = 1) = {sum(1 for v in flat if v == 1)} for the whole matrix",
           size="small", role="c", bold=True)
    return finish(s, note(s, 40, y0 + R * cw + 34, dims_note))


def fig_044_count_2d_array_element(name):
    return _count_grid(name, "Counting in a 2-D array: the matrix is contiguous, so count over the flat array",
                       ["The row/column structure only fixes the element count N·M; the kernel is the 1-D count",
                        "with a grid-stride loop, int4 (4 × int32) loads and a warp-level integer reduction."])


def fig_045_count_3d_array_element(name):
    return _count_grid(name, "Counting in a 3-D array: still one flat, contiguous count",
                       ["An N × M × K tensor is the same flat array. N·M·K ≤ 10⁹ elements fits in 32 bits, but the",
                        "byte offset 4·N·M·K does not, so pointers are advanced with 64-bit arithmetic."],
                       grid_title="two slices of the 3-D tensor", slices=True)


def fig_046_bfs_shortest_path(name):
    s = Svg(name, W, 100, "BFS on a grid: each frontier F_ℓ holds the cells at distance ℓ from the start")
    grid = ["....#...",
            ".##.#.#.",
            "...#..#.",
            ".#...##.",
            ".#.#....",
            "...#.##."]
    R, C, cw = len(grid), len(grid[0]), 34
    start, goal = (0, 0), (5, 7)
    dist = {start: 0}
    q = [start]
    for cur in q:
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == "." and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[cur] + 1
                q.append((nr, nc))
    x0, y0 = 40, 40
    for r in range(R):
        for c in range(C):
            if grid[r][c] == "#":
                s.rect(x0 + c * cw, y0 + r * cw, cw, cw, fill="f-ink2", stroke="s-line", sw=0.6)
                continue
            d = dist.get((r, c))
            role = "hl" if (r, c) in (start, goal) else ("a" if d is not None and d % 2 == 0 else "b")
            s.box(x0 + c * cw, y0 + r * cw, cw, cw, "" if d is None else str(d), role=role, rx=0,
                  fill="f-hl" if (r, c) in (start, goal) else None, size="small")
    s.rect(x0, y0, C * cw, R * cw, fill="fig-none", stroke="s-ink", sw=1.4)
    tx = x0 + C * cw + 30
    note(s, tx, y0 + 10, ["numbers: BFS distance from the", "start (top-left, red); dark cells",
                          "are obstacles.", "", f"goal (bottom-right): distance {dist[goal]}",
                          "", "One persistent kernel expands a", "whole frontier per level and uses",
                          "atomics to claim new cells; the", "levels themselves stay sequential."], role="ink")
    return finish(s, y0 + R * cw + 10)


def _range_sum(name, title, dims_note, box):
    s = Svg(name, W, 100, title)
    x0, y0, cw = 40, 50, 30
    R, C = 5, 9
    (r0, r1), (c0, c1) = box
    tot = 0
    for r in range(R):
        for c in range(C):
            v = 1 + (r * 7 + c * 3) % 9
            inside = r0 <= r <= r1 and c0 <= c <= c1
            tot += v if inside else 0
            s.box(x0 + c * cw, y0 + r * cw, cw, cw, str(v), role="a" if inside else "line",
                  fill="f-a2" if inside else "fig-paper", size="tiny", rx=0)
    s.rect(x0 + c0 * cw, y0 + r0 * cw, (c1 - c0 + 1) * cw, (r1 - r0 + 1) * cw, fill="fig-none", stroke="s-hl", sw=2)
    s.rect(x0, y0, C * cw, R * cw, fill="fig-none", stroke="s-ink", sw=1.4)
    tx = x0 + C * cw + 40
    note(s, tx, y0 + 10, [f"sum of the box = {tot}", "", *dims_note], role="ink")
    return finish(s, y0 + R * cw + 10)


def fig_047_subarray_sum(name):
    s = Svg(name, W, 100, "Subarray sum: add the values x[S] … x[E], an exact integer reduction")
    xs = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    S, E = 2, 7
    x0, cw, y0 = 110, 52, 50
    for i in range(len(xs)):
        s.text(x0 + i * cw + 22, y0 - 16, str(i), size="small", role="muted")
    cells(s, x0, y0, [str(v) for v in xs], cw=44, gap=8, role="a",
          fills=["f-a2" if S <= i <= E else "fig-panel" for i in range(len(xs))])
    s.brace_h(x0 + S * cw, x0 + E * cw + 44, y0 + 42, f"S = {S} … E = {E}:  sum = {sum(xs[S:E + 1])}", role="hl")
    return finish(s, note(s, 40, y0 + 90, ["Only the E − S + 1 elements in range are read. Threads map a flat index i to",
                                           "x[S + i]; per-thread int sums, a warp __reduce_add_sync, then one atomicAdd.",
                                           "The maximum 10 · 10⁸ = 10⁹ < 2³¹, so int32 cannot overflow."]))


def fig_048_2d_subarray_sum(name):
    return _range_sum(name, "2-D subarray sum: add every value inside the rectangle", [
        "rows r₀ … r₁, columns c₀ … c₁", "(red box). A flat index i over", "the h × w box maps to",
        "(r₀ + i / w, c₀ + i mod w):", "consecutive threads read", "consecutive columns."], ((1, 3), (2, 6)))


def fig_049_3d_subarray_sum(name):
    s = Svg(name, W, 100, "3-D subarray sum: add every value inside a box of depth slices")
    x0, y0, cw = 40, 50, 26
    tot = 0
    for d in range(3):
        xx = x0 + d * 170
        for r in range(4):
            for c in range(5):
                v = 1 + (d * 5 + r * 7 + c * 3) % 9
                inside = d >= 1 and 1 <= r <= 2 and 1 <= c <= 3
                tot += v if inside else 0
                s.box(xx + c * cw, y0 + r * cw, cw, cw, str(v), role="a" if inside else "line",
                      fill="f-a2" if inside else "fig-paper", size="tiny", rx=0)
        s.rect(xx, y0, 5 * cw, 4 * cw, fill="fig-none", stroke="s-ink", sw=1.4)
        s.text(xx + 2.5 * cw, y0 - 14, f"depth {d}" + (" (in box)" if d >= 1 else ""), size="small",
               role="a" if d >= 1 else "muted")
    s.text(x0 + 3 * 170 + 10, y0 + 2 * cw, f"box sum = {tot}", anchor="start", size="small", role="hl", bold=True)
    return finish(s, note(s, 40, y0 + 4 * cw + 34, ["Flat index i over the d × h × w box → (a₀ + i/(hw), b₀ + (i/w) mod h, c₀ + i mod w).",
                                                    "Consecutive threads read consecutive columns; the reduction is exact int32."]))


def fig_050_rms_normalization(name):
    s = Svg(name, W, 100, "RMS normalisation of one vector: a global reduction, then an elementwise pass")
    xs = [2.0, -1.0, 3.0, 0.0, -2.0, 1.0]
    rms = math.sqrt(sum(v * v for v in xs) / len(xs) + 1e-5)
    g, b = 1.0, 0.0
    x0, cw, y0 = 130, 62, 40
    row_label(s, x0 - 12, y0 + 14, "x", role="a", bold=True)
    cells(s, x0, y0, [fmt(v) for v in xs], cw=cw - 8, gap=8, role="a")
    s.box(x0 + 100, y0 + 56, 240, 30, f"rms = √(Σx²/N + ε) = {fmt(rms, 3)}", role="d", size="small")
    for i in range(len(xs)):
        s.line(x0 + i * cw + 27, y0 + 28, x0 + 220, y0 + 56, stroke="s-d", sw=0.7)
    y1 = y0 + 120
    row_label(s, x0 - 12, y1 + 14, "y", role="c", bold=True)
    cells(s, x0, y1, [fmt(g * v / rms + b, 3) for v in xs], cw=cw - 8, gap=8, role="c")
    s.arrow(x0 + 220, y0 + 88, x0 + 220, y1 - 2, role="d", sw=1.2)
    s.text(x0 + 230, y0 + 104, "y_i = γ · x_i / rms + β  (γ = 1, β = 0)", anchor="start", size="small", role="c")
    return finish(s, note(s, 40, y1 + 58, ["Pass 1 reduces Σx² over the whole vector (grid-wide, float64 across blocks);",
                                           "pass 2 scales every element by the same γ / rms and adds β."]))


def fig_051_max_subarray_sum(name):
    s = Svg(name, W, 100, "Maximum window sum: every window is a difference of two prefix sums")
    xs = [2, -3, 4, -1, 5, -2, 3, -4]
    w = 3
    P = [0]
    for v in xs:
        P.append(P[-1] + v)
    sums = [P[i + w] - P[i] for i in range(len(xs) - w + 1)]
    best = max(range(len(sums)), key=lambda i: sums[i])
    x0, cw, y0 = 190, 56, 40
    row_label(s, x0 - 44, y0 + 14, "x", role="a", bold=True)
    cells(s, x0, y0, [fmt(v) for v in xs], cw=cw - 8, gap=8, role="a",
          fills=["f-a2" if best <= i < best + w else None for i in range(len(xs))])
    y1 = y0 + 50
    row_label(s, x0 - 44, y1 + 14, "P (exclusive)", role="d", bold=True)
    cells(s, x0 - cw / 2, y1, [fmt(v) for v in P], cw=cw - 8, gap=8, role="d",
          fills=["f-d2" if i in (best, best + w) else None for i in range(len(P))])
    y2 = y1 + 50
    row_label(s, x0 - 44, y2 + 14, f"window (w = {w})", role="c", bold=True)
    cells(s, x0, y2, [fmt(v) for v in sums], cw=cw - 8, gap=8, role="c",
          fills=["f-c2" if i == best else None for i in range(len(sums))])
    s.text(x0 + len(sums) * cw + 10, y2 + 14, f"max = {sums[best]} = P[{best + w}] − P[{best}]", anchor="start",
           size="small", role="c", bold=True)
    return finish(s, note(s, 40, y2 + 58, ["After one scan, the N − w + 1 window sums are independent subtractions, so the",
                                           "answer is a plain max-reduction; all arithmetic is exact int32."]))


def fig_052_silu(name):
    return kit.function_plot(name, "SiLU (Swish-1): x · σ(x), smooth and slightly negative below 0",
                             [(lambda x: max(0.0, x), "line", "4 3"), (silu, "a", None)], (-6, 4), (-1, 4),
                             [-6, -4, -2, 0, 2, 4], [-1, 0, 1, 2, 3, 4],
                             ["SiLU(x) = x · σ(x) = x / (1 + e⁻ˣ)", "→ x for large x, → 0⁻ for very negative x"],
                             points=[(-1.278, silu(-1.278), "hl")],
                             legend=[("f-a2", "a", "SiLU"), ("fig-panel", "line", "ReLU (dashed)")],
                             side_lines=["red dot: minimum ≈ −0.278 at", "x ≈ −1.278.", "",
                                         "Elementwise, one thread per value;", "expf is cheap next to the memory",
                                         "traffic."])


def fig_053_casual_attention(name):
    return kit.attention(name, "Causal attention: query i may only look at keys j ≤ i", 10, 10,
                         lambda i, j: j <= i, hl_row=6,
                         legend=[("f-a2", "a", "visible (j ≤ i)"), ("fig-panel", "line", "masked, never computed"),
                                 ("f-c2", "c", "row 6: keys 0 … 6")],
                         pipeline_steps=["s = q·k / √d", "online softmax", "O = Σ p · v"],
                         note_lines=["Row i has i + 1 visible keys, M(M+1)/2 pairs in total. Key tiles entirely above",
                                     "the diagonal are skipped instead of computed and discarded: half the work."])


def _gated(name, title, act, act_name, first_is_gate, note_lines):
    s = Svg(name, W, 100, title)
    xs = [1.0, -2.0, 0.5, 3.0, 2.0, 1.5, -1.0, 0.5]
    n = len(xs)
    h = n // 2
    x0, cw, y0 = 110, 64, 44
    for i in range(n):
        s.text(x0 + i * cw + 28, y0 - 14, str(i), size="small", role="muted")
    row_label(s, x0 - 12, y0 + 14, "x", bold=True)
    gate_role, val_role = ("b", "a") if first_is_gate else ("a", "b")
    cells(s, x0, y0, [fmt(v) for v in xs], cw=cw - 8, gap=8, role=[gate_role] * h + [val_role] * h)
    s.brace_h(x0, x0 + h * cw - 8, y0 + 40, ("gate: x₁ (first half)" if first_is_gate else "value: x₁ (first half)"),
              role=gate_role if first_is_gate else val_role)
    s.brace_h(x0 + h * cw, x0 + n * cw - 8, y0 + 40, ("value: x₂ (second half)" if first_is_gate
                                                      else "gate: x₂ (second half)"),
              role=val_role if first_is_gate else gate_role)
    y1 = y0 + 110
    ys = []
    for i in range(h):
        g, v = (xs[i], xs[i + h]) if first_is_gate else (xs[i + h], xs[i])
        ys.append(act(g) * v)
        s.line(x0 + i * cw + 28, y0 + 60, x0 + i * cw + 28 + 40, y1, stroke="s-muted", sw=0.9)
        s.line(x0 + (i + h) * cw + 28, y0 + 60, x0 + i * cw + 28 + 40, y1, stroke="s-muted", sw=0.9)
    row_label(s, x0 + 28, y1 + 14, "y", role="c", bold=True)
    cells(s, x0 + 40, y1, [fmt(v, 3) for v in ys], cw=cw - 8, gap=8, role="c")
    s.text(x0 + 40 + h * cw + 10, y1 + 14, f"y_i = {act_name}", anchor="start", size="small", role="c")
    return finish(s, note(s, 40, y1 + 58, note_lines))


def fig_054_swiglu(name):
    return _gated(name, "SwiGLU gate: SiLU of the first half times the second half", silu,
                  "SiLU(x_i) · x_{i+N/2}", True,
                  ["Thread i reads x_i and x_{i+N/2} (two coalesced streams) and writes y_i: the output has N/2",
                   "elements. In an LLM MLP the halves are the \"gate\" and \"up\" projections."])


def fig_055_attn_w_linear_bias(name):
    s = Svg(name, W, 100, "ALiBi: add α · (i − j) to every score before the softmax (α = −0.5 shown)")
    n, cw, x0, y0 = 7, 40, 110, 56
    alpha = -0.5
    for i in range(n):
        for j in range(n):
            b = alpha * (i - j)
            f = "f-a2" if b == 0 else ("f-a" if abs(b) <= 1 else ("fig-panel" if b < 0 else "f-b"))
            s.box(x0 + j * cw, y0 + i * cw, cw, cw, fmt(b), role="line", fill=f, size="tiny", rx=0)
    s.rect(x0, y0, n * cw, n * cw, fill="fig-none", stroke="s-ink", sw=1.4)
    s.text(x0 + n * cw / 2, y0 - 26, "key j →", size="small", role="muted")
    s.text(x0 - 8, y0 - 26, "query i ↓", size="small", role="muted", anchor="end")
    for j in range(n):
        s.text(x0 + j * cw + cw / 2, y0 - 9, str(j), size="tiny", role="muted")
        s.text(x0 - 8, y0 + j * cw + cw / 2, str(j), size="tiny", role="muted", anchor="end")
    tx = x0 + n * cw + 40
    note(s, tx, y0 + 10, ["S_{ij} = q_i · k_j / √d + α (i − j)", "", "cells: the bias α (i − j)",
                          "α < 0 penalises distant past keys", "linearly (bias grows with i − j);",
                          "keys after i get a positive bias,", "since this problem has no mask.", "",
                          "The bias is computed on the fly", "inside the flash-attention loop:",
                          "no position embeddings, no extra", "memory traffic."], role="ink")
    return finish(s, y0 + n * cw + 10)


def fig_056_linear_attention(name):
    s = Svg(name, W, 100, "Linear attention: associativity turns an M × M product into a d × d state")
    x0, y0 = 40, 50
    s.text(x0, y0 - 18, "softmax-free kernel attention, computed two ways", anchor="start", size="small", bold=True)
    s.box(x0, y0, 150, 40, "(φ(Q) φ(K)ᵀ) V", role="hl", size="small")
    s.text(x0 + 170, y0 + 20, "M × M matrix first:  O(M² d)", anchor="start", size="small", role="hl")
    s.box(x0, y0 + 60, 150, 40, "φ(Q) (φ(K)ᵀ V)", role="c", size="small")
    s.text(x0 + 170, y0 + 80, "d × d state first:  O(M d²)", anchor="start", size="small", role="c")
    y1 = y0 + 165
    steps = [("S = Σ_{j} φ(k_j) v_jᵀ\n(d × d)", "d"), ("z = Σ_{j} φ(k_j)\n(length d)", "d"),
             ("O_i = φ(q_i)ᵀS\n/ φ(q_i)ᵀz", "c")]
    kit.pipeline(s, x0, y1, steps, w=190, h=46, gap=30)
    p = Plot(s, 500, y0 + 10, 170, 100, (-3, 2), (0, 3))
    p.axes([-3, 0, 2], [0, 1, 3], "x", "φ")
    p.curve(lambda x: x + 1 if x > 0 else math.exp(x), role="b")
    s.text(585, p.y - 12, "φ(x) = ELU(x) + 1 > 0", size="small", role="b")
    return finish(s, note(s, 40, y1 + 76, ["Pass 1 reduces φ(K)ᵀV and Σφ(k) over all M rows (a d × d reduction); pass 2 is",
                                           "one small matrix-vector product per query. Nothing of size M × M is ever built."]))


def fig_057_fp16_batched_matmul(name):
    return kit.gemm(name, "FP16 batched GEMM: tensor-core fragments, fp32 accumulation, one fp16 rounding", batch=2,
                    epilogue=["grid.z = batch index b", "wmma 16 × 16 × 16 per warp", "fp16 inputs, fp32 accumulators",
                              "C_b = fp16(accumulator)"],
                    note_lines=["Same tensor-core kernel as the fp16 GEMM, with per-batch base pointers."])


def fig_058_fp16_dot_product(name):
    a = [0.5, 1.5, -1.0, 2.0, 0.25, 1.0, -0.5, 3.0]
    b = [2.0, 1.0, 1.0, 0.5, 4.0, -2.0, 2.0, 1.0]
    return kit.reduce_tree(name, "FP16 dot product: half the bytes of fp32, but accumulate in fp32",
                           [x * y for x, y in zip(a, b)], lambda p, q: p + q, "fp32(a)·fp32(b)",
                           map_row=("a_i, b_i (fp16)", [f"{fmt(x)}, {fmt(y)}" for x, y in zip(a, b)]),
                           result_label="s → fp16 once",
                           final_lines=["fp16 has 11 significant bits: once a running sum reaches ~2048, adding values",
                                        "below 1 changes nothing. Widening to fp32 before the multiply avoids it; loads",
                                        "use half2 pairs so the kernel still streams at full bandwidth."])


def fig_059_sliding_window_attn(name):
    w = 2
    return kit.attention(name, "Sliding-window attention: query i sees keys within distance w", 12, 12,
                         lambda i, j: abs(i - j) <= w, hl_row=5,
                         legend=[("f-a2", "a", f"visible: |i − j| ≤ w = {w}"), ("fig-panel", "line", "outside the band"),
                                 ("f-c2", "c", "row 5: keys 3 … 7")],
                         pipeline_steps=["only key tiles that meet the band", "online softmax over ≤ 2w + 1 keys",
                                         "O_i = Σ p · v"],
                         note_lines=["The kernel loops only over the key tiles that intersect the band, so the cost is",
                                     "O(M·w·d) instead of O(M²·d); the band is clipped at both ends of the sequence."])


def fig_060_top_p_sampling(name):
    s = Svg(name, W, 100, "Top-p (nucleus) sampling: keep the most likely tokens until their mass reaches p")
    probs = [0.30, 0.22, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04]
    p_ = 0.7
    cum, keep = 0, []
    for v in probs:
        keep.append(cum < p_)
        cum += v
    x0, y0, h = 70, 40, 150
    fills = ["f-c2" if k else "f-a" for k in keep]
    kit.bars(s, x0, y0, h, probs, 0.35, bw=40, gap=14, fills=fills, labels=[f"t{i}" for i in range(len(probs))],
             vfmt=lambda v: fmt(v, 2))
    cx = x0 + len(probs) * 54 + 20
    tot = sum(v for v, k in zip(probs, keep) if k)
    note(s, cx, y0 + 10, ["tokens by probability (desc.)", f"p = {p_}: keep t0 … t3", f"cumulative mass = {fmt(tot, 2)} ≥ p",
                          "threshold T = 0.10", "", "renormalise the kept mass", "and sample with the seed"], role="ink")
    return finish(s, note(s, 40, y0 + h + 44, ["No full sort: the nucleus is {t : π_{t} ≥ T}, so T is found by a radix-select style",
                                               "search on the probability bits, with block-wide sums of the mass above each candidate."]))


# ----- 061-085 -----------------------------------------------------------------------------------------------

def fig_061_rope_embedding(name):
    s = Svg(name, W, 100, "RoPE (half-split layout): element j is rotated together with element j + D/2")
    D, x0, cw, y0 = 8, 60, 50, 50
    roles = ["a", "b", "c", "d"]
    for j in range(D):
        s.text(x0 + j * cw + 22, y0 - 14, str(j), size="small", role="muted")
        s.box(x0 + j * cw, y0, 44, 30, f"x{j}", role=roles[j % 4], size="small", rx=2)
    for j in range(D // 2):
        xa, xb = x0 + j * cw + 22, x0 + (j + D // 2) * cw + 22
        s.path(f"M{xa},{y0 + 32} C{xa},{y0 + 70 + 8 * j} {xb},{y0 + 70 + 8 * j} {xb},{y0 + 32}",
               stroke=f"s-{roles[j]}", sw=1.2, arrow=roles[j], both=True)
    y1 = y0 + 130
    note(s, x0, y1, ["y_j = x_j·cos_j − x_{j+h}·sin_j", "y_{j+h} = x_{j+h}·cos_{j+h} + x_j·sin_{j+h}", "h = D/2",
                     "", "one thread per pair (j, j + h):", "two loads, four multiplies, two stores"], role="ink")
    cx, cy, r = 520, 130, 80
    s.circle(cx, cy, r, fill="fig-panel", stroke="s-line", sw=1)
    s.line(cx - r - 10, cy, cx + r + 10, cy, stroke="s-muted", sw=0.8)
    s.line(cx, cy - r - 10, cx, cy + r + 10, stroke="s-muted", sw=0.8)
    th = 0.5
    s.arrow(cx, cy, cx + r * 0.9, cy - r * 0.35, role="a", sw=1.6)
    s.arrow(cx, cy, cx + r * (0.9 * math.cos(th) - 0.35 * math.sin(th)) * 1.0,
            cy - r * (0.9 * math.sin(th) + 0.35 * math.cos(th)), role="c", sw=1.6)
    s.text(cx + r * 0.9 + 12, cy - r * 0.35 + 8, "(x_j, x_{j+h})", anchor="start", size="small", role="a")
    s.text(cx + 30, cy - r - 4, "rotated by mθ_{j}", anchor="start", size="small", role="c", plate=True)
    return finish(s, max(y1 + 5 * 17 + 8, cy + r + 10))


def fig_062_value_clipping(name):
    lo, hi = -1.0, 2.0
    return kit.function_plot(name, "Value clipping: values below ℓ become ℓ, values above h become h",
                             [(lambda x: min(max(x, lo), hi), "a", None)], (-3, 4), (-2, 3), [-3, -2, -1, 0, 1, 2, 3, 4],
                             [-2, -1, 0, 1, 2, 3], ["y = min(max(x, ℓ), h)", "ℓ = −1, h = 2 in this plot"],
                             points=[(-2.5, lo, "b"), (1.0, 1.0, "hl"), (3.5, hi, "b")],
                             side_lines=["orange dots: clipped to a bound", "red dot: inside, passed through",
                                         "", "fminf / fmaxf, one thread per", "element: purely memory-bound."])


def fig_063_interleave(name):
    s = Svg(name, W, 100, "Interleave (SoA → AoS): o[2i] = a[i], o[2i + 1] = b[i]")
    x0, y0, cw = 80, 40, 36
    n = 6
    row_label(s, x0 - 10, y0 + 14, "A", role="a", bold=True)
    row_label(s, x0 + 330 - 10, y0 + 14, "B", role="b", bold=True)
    for i in range(n):
        s.box(x0 + i * cw, y0, cw - 4, 28, f"a{i}", role="a", size="small", rx=2)
        s.box(x0 + 330 + i * cw, y0, cw - 4, 28, f"b{i}", role="b", size="small", rx=2)
    y1 = y0 + 110
    row_label(s, x0 - 10, y1 + 14, "out", role="c", bold=True)
    for k in range(2 * n):
        s.box(x0 + k * 46, y1, 42, 28, f"{'ab'[k % 2]}{k // 2}", role="a" if k % 2 == 0 else "b", size="small", rx=2)
    for i in range(n):
        s.line(x0 + i * cw + 16, y0 + 28, x0 + 2 * i * 46 + 21, y1, stroke="s-a", sw=0.9)
        s.line(x0 + 330 + i * cw + 16, y0 + 28, x0 + (2 * i + 1) * 46 + 21, y1, stroke="s-b", sw=0.9)
    return finish(s, note(s, 40, y1 + 58, ["Thread i loads a[i] and b[i] and stores them as one float2 (8 bytes) at o[2i]:",
                                           "both reads and the write are contiguous across the warp, so it streams at full",
                                           "bandwidth (this is how complex numbers or (x, y) points are packed)."]))


def fig_064_weight_dequantization(name):
    s = Svg(name, W, 100, "Tile-wise dequantisation: every T × T tile of X shares one scale from S")
    x0, y0, cw = 40, 50, 22
    R, C, T = 6, 10, 3
    roles = ["a", "b", "c", "d"]
    for r in range(R):
        for c in range(C):
            t = (r // T) * 4 + (c // T)
            s.rect(x0 + c * cw, y0 + r * cw, cw, cw, fill=f"f-{roles[t % 4]}" if (c // T) % 2 == (r // T) % 2
                   else f"f-{roles[(t + 1) % 4]}2", stroke="s-line", sw=0.6)
    s.rect(x0, y0, C * cw, R * cw, fill="fig-none", stroke="s-ink", sw=1.4)
    for k in range(1, 4):
        s.line(x0 + k * T * cw, y0, x0 + k * T * cw, y0 + R * cw, stroke="s-ink", sw=1.4)
    s.line(x0, y0 + T * cw, x0 + C * cw, y0 + T * cw, stroke="s-ink", sw=1.4)
    s.text(x0 + C * cw / 2, y0 - 14, "X (M × N), tiles of T × T (T = 3 here)", size="small", bold=True)
    sx = x0 + C * cw + 90
    for r in range(2):
        for c in range(4):
            t = r * 4 + c
            s.box(sx + c * 40, y0 + 20 + r * 40, 40, 40, f"s{r}{c}", role="ink",
                  fill=f"f-{roles[t % 4]}" if c % 2 == r % 2 else f"f-{roles[(t + 1) % 4]}2", size="small", rx=0)
    s.text(sx + 80, y0 - 14, "S (⌈M/T⌉ × ⌈N/T⌉)", size="small", bold=True)
    s.text(sx + 80, y0 + 120, "the last tile column is partial", size="tiny", role="muted")
    return finish(s, note(s, 40, y0 + R * cw + 34, ["Y_{ij} = X_{ij} · S[⌊i/T⌋, ⌊j/T⌋]. Threads walk rows of X with float4 loads; the scale is",
                                                    "one cached read per T columns, so the kernel runs at copy bandwidth."]))


def fig_065_geglu(name):
    return _gated(name, "GEGLU: the first half times GELU of the second half", gelu, "x_i · GELU(x_{i+N/2})", False,
                  ["Unlike SwiGLU, the second half is the gate here. GELU(u) = u·Φ(u) = ½u(1 + erf(u/√2))",
                   "uses the exact erf, as in the reference."])


def fig_066_rgb_to_grayscale(name):
    s = Svg(name, W, 100, "RGB to grayscale: three interleaved channels per pixel, one weighted sum")
    px = [(200, 100, 50), (10, 220, 30), (90, 90, 250)]
    x0, y0 = 60, 50
    roles = ["hl", "c", "a"]
    for p, rgb in enumerate(px):
        for c in range(3):
            s.box(x0 + (3 * p + c) * 54, y0, 50, 28, str(rgb[c]), role=roles[c], size="small", rx=2)
        s.brace_h(x0 + 3 * p * 54, x0 + (3 * p + 3) * 54 - 4, y0 + 40, f"pixel {p}", role="muted")
    y1 = y0 + 110
    for p, (r, g, b) in enumerate(px):
        yv = 0.299 * r + 0.587 * g + 0.114 * b
        cx = x0 + (3 * p + 1) * 54 + 25
        s.arrow(cx, y0 + 64, cx, y1 - 2, role="muted", sw=1)
        s.box(cx - 40, y1, 80, 28, fmt(yv, 2), role="ink", size="small", rx=2)
    s.text(x0, y1 + 58, "Y = 0.299 R + 0.587 G + 0.114 B  (ITU-R BT.601)", anchor="start", size="small", bold=True)
    return finish(s, note(s, x0, y1 + 84, ["Thread p reads x[3p], x[3p+1], x[3p+2]: a stride-3 pattern that the L1/L2 caches",
                                           "absorb, since neighbouring threads share the same cache lines."]))


def fig_067_moe_topk_gating(name):
    s = Svg(name, W, 100, "MoE gating: pick the k best experts per token, softmax only their logits")
    z = [0.5, 2.1, -0.3, 1.2, 2.1, 0.0, 1.8, -1.0]
    k = 2
    order = sorted(range(len(z)), key=lambda i: (-z[i], i))[:k]
    x0, y0, cw = 110, 50, 60
    for i in range(len(z)):
        s.text(x0 + i * cw + 26, y0 - 14, f"e{i}", size="small", role="muted")
    row_label(s, x0 - 12, y0 + 14, "logits", bold=True)
    cells(s, x0, y0, [fmt(v) for v in z], cw=cw - 8, gap=8, role="a",
          fills=["f-c2" if i in order else None for i in range(len(z))])
    m = z[order[0]]
    ws = [math.exp(z[i] - m) for i in order]
    ws = [v / sum(ws) for v in ws]
    y1 = y0 + 90
    for t, i in enumerate(order):
        bx = x0 + 120 + t * 170
        s.box(bx, y1, 150, 34, f"expert {i}: w = {fmt(ws[t], 3)}", role="c", size="small")
        s.arrow(x0 + i * cw + 26, y0 + 30, bx + 75, y1 - 2, role="c", sw=1.1)
    return finish(s, note(s, 40, y1 + 64, ["Ties go to the lower index (e1 before e4, as torch.topk does). The weights are a",
                                           "softmax over the k selected logits only; one warp handles one token's E logits."]))


def fig_068_sigmoid(name):
    return kit.function_plot(name, "The logistic sigmoid squashes any input into (0, 1)",
                             [(sigmoid, "a", None)], (-8, 8), (0, 1), [-8, -4, 0, 4, 8], [0, 0.5, 1],
                             ["σ(x) = 1 / (1 + e⁻ˣ)", "σ(−x) = 1 − σ(x),  σ(0) = 0.5"],
                             points=[(0, 0.5, "hl")],
                             side_lines=["Safe without branches in float32:", "x → −∞: e⁻ˣ → ∞ and 1/∞ = 0;",
                                         "x → +∞: e⁻ˣ → 0 and σ = 1.", "", "One thread per element (float4)."])


def fig_069_jacobi_stencil_2d(name):
    taps = {(2, 3), (1, 3), (3, 3), (2, 2), (2, 4)}
    s = kit.window_2d(name, "Jacobi 5-point stencil: each interior cell becomes the mean of its 4 neighbours", 6, 8,
                      3, out_rc=(2, 3), taps=taps - {(2, 3)}, op_text="u′_{ij} = ¼ (N + S + W + E)",
                      note_lines=["Boundary cells (outer ring of the output) are copied unchanged. Every input value is",
                                  "read by up to 4 neighbours, so a tiled kernel loads each row once through shared",
                                  "memory or L1 and writes a separate output grid (no in-place update)."])
    return s


def fig_070_segmented_prefix_sum(name):
    xs = [3, 1, 4, 1, 5, 9, 2, 6]
    flags = [1, 0, 0, 1, 0, 1, 0, 0]
    ys, acc = [], 0
    for v, f in zip(xs, flags):
        if f:
            acc = 0
        ys.append(acc)
        acc += v
    return kit.scan(name, "Segmented exclusive scan: the running sum restarts at every flagged head", xs, ys,
                    "exclusive, per segment", flags=flags, highlight=(3, 3, 4), cw=48,
                    note_lines=["Scan over pairs (f, s) with (f₁, s₁) ⊕ (f₂, s₂) = (f₁ | f₂, f₂ ? s₂ : s₁ + s₂): the",
                                "operator is associative, so the ordinary decoupled look-back scan applies unchanged."])


def fig_071_parallel_merge(name):
    s = Svg(name, W, 100, "Merge path: each thread finds where its output range starts by a binary search")
    A = [1, 3, 5, 8, 9]
    B = [2, 4, 6, 7, 10]
    x0, y0, cw = 110, 60, 44
    for j, v in enumerate(B):
        s.box(x0 + j * cw, y0 - 36, cw - 4, 28, str(v), role="b", size="small", rx=2)
    for i, v in enumerate(A):
        s.box(x0 - 44, y0 + i * cw + 6, 36, cw - 12, str(v), role="a", size="small", rx=2)
    s.grid(x0, y0, len(A), len(B), cw, fill_fn=lambda r, c: "f-c" if A[r] > B[c] else None)
    # merge path: walk the grid
    i = j = 0
    pts = [(x0, y0)]
    while i < len(A) or j < len(B):
        if j < len(B) and (i >= len(A) or B[j] < A[i]):
            j += 1
        else:
            i += 1
        pts.append((x0 + j * cw, y0 + i * cw))
    s.path("M" + " L".join(f"{a},{b}" for a, b in pts), stroke="s-hl", sw=3)
    for d in (4, 8):
        s.line(x0 + d * cw if d <= 5 else x0 + 5 * cw, y0 + (0 if d <= 5 else (d - 5) * cw),
               x0 + (0 if d <= 5 else d - 5) * cw, y0 + (d if d <= 5 else 5) * cw, stroke="s-d", sw=1.4, dash="5 3")
    tx = x0 + 5 * cw + 40
    note(s, tx, y0, ["red: the merge path (down = take A,", "right = take B); shaded cells: A_i > B_j.", "",
                     "Dashed diagonals: output positions", "k = 4 and k = 8. Where a diagonal",
                     "crosses the path gives the co-rank", "i (elements from A before k): a", "binary search on the diagonal.",
                     "", "Every thread then merges its own", "k-range sequentially, independently."], role="ink")
    return finish(s, y0 + 5 * cw + 10)


def fig_072_stream_compaction(name):
    s = Svg(name, W, 100, "Stream compaction: a predicate, an exclusive scan, and a scatter")
    A = [3, -1, 0, 7, 2, -4, 5, -2]
    p = [1 if v > 0 else 0 for v in A]
    o, acc = [], 0
    for v in p:
        o.append(acc)
        acc += v
    x0, cw, y0 = 150, 56, 40
    rows = [("A", [fmt(v) for v in A], "a"), ("p = [A > 0]", [str(v) for v in p], "b"),
            ("o = excl. scan", [str(v) for v in o], "d")]
    for r, (nm, labs, role) in enumerate(rows):
        row_label(s, x0 - 12, y0 + r * 40 + 14, nm, role=role, bold=True)
        cells(s, x0, y0 + r * 40, labs, cw=cw - 8, gap=8, role=role,
              fills=["f-" + role + "2" if p[i] else "fig-panel" for i in range(len(A))])
    y1 = y0 + 3 * 40 + 50
    out = [v for v in A if v > 0] + [0] * (len(A) - acc)
    row_label(s, x0 - 12, y1 + 14, "out", role="c", bold=True)
    cells(s, x0, y1, [fmt(v) for v in out], cw=cw - 8, gap=8, role="c",
          fills=["f-c2" if i < acc else "fig-panel" for i in range(len(A))])
    for i in range(len(A)):
        if p[i]:
            s.line(x0 + i * cw + 24, y0 + 2 * 40 + 28, x0 + o[i] * cw + 24, y1, stroke="s-c", sw=1)
    return finish(s, note(s, 40, y1 + 58, ["Kept element i goes to out[o_i]: unique and order-preserving (stable). The",
                                           f"remaining N − k slots (k = {acc}) are filled with 0."]))


def fig_073_all_pairs_shortest_paths(name):
    s = Svg(name, W, 100, "Blocked Floyd–Warshall: for each diagonal block, three dependent phases")
    x0, y0, cw = 60, 50, 44
    nb = 5
    kb = 2

    def f(r, c):
        if r == kb and c == kb:
            return "f-hl"
        if r == kb or c == kb:
            return "f-b2"
        return "f-a"

    matrix(s, x0, y0, nb, nb, cw, fill_fn=f, role="ink")
    s.text(x0 + nb * cw / 2, y0 - 14, "distance matrix in B × B blocks (k-block = 2)", size="small", bold=True)
    tx = x0 + nb * cw + 40
    s.legend(tx, y0 + 12, [("f-hl", "hl", "phase 1: pivot block (k, k)"),
                           ("f-b2", "b", "phase 2: pivot row and column"),
                           ("f-a", "a", "phase 3: all other blocks")])
    note(s, tx, y0 + 90, ["d_{ij} = min(d_{ij}, d_{ik} + d_{kj})", "for every k inside the k-block.", "",
                          "Phase 3 is a (min, +) \"matrix", "product\" of the pivot column and", "pivot row: tiled like GEMM,",
                          "in shared memory."], role="ink")
    return finish(s, y0 + nb * cw + 10)


def fig_074_gpt2_block(name):
    rows = [[("x", "a"), ("LayerNorm 1", "d"), ("QKV GEMM\n+ bias", "b"), ("causal\nattention", "c"),
             ("out proj\n+ residual", "b")],
            [("LayerNorm 2", "d"), ("FC GEMM\n+ bias, GELU", "b"), ("proj GEMM\n+ residual", "b"), ("y", "a")]]
    return kit.layer_flow(name, "GPT-2 block: pre-LayerNorm attention and MLP, each with a residual add", rows,
                          residuals=[(0, 0, 4, "residual x"), (1, 0, 2, "residual x′")],
                          note_lines=["Fusion: biases, GELU and the residual adds run in GEMM epilogues, and LayerNorm reads",
                                      "the residual stream once, so the only full-size round trips are the GEMM outputs.",
                                      "d = 768, 12 heads of 64, MLP width 3072."])


def fig_075_sparse_matrix_dense_matrix_multiplication(name):
    import random
    rng = random.Random(75)
    mask = {(r, c) for r in range(6) for c in range(6) if rng.random() < 0.65}
    return kit.gemm(name, "Sparse × dense at 35% density: a dense tiled GEMM still wins",
                    a_mask=lambda r, c: (r, c) in mask,
                    epilogue=["dense work 2MNK FLOPs", "sparse ideal 2ρMNK (ρ ≈ 0.35)", "but: no index structure,",
                              "irregular gathers of B rows", "→ a dense tensor-core-free", "SGEMM is faster here"],
                    note_lines=["Grey cells of A are zeros. Sparse kernels only pay off far below ~10% density."])


def fig_076_adder_transformer(name):
    s = Svg(name, W, 100, "Autoregressive decoding with a KV cache: prefill once, then one position per step")
    x0, y0, cw = 40, 60, 18
    for t in range(31):
        s.rect(x0 + t * cw, y0, cw - 2, 26, fill="f-a2", stroke="s-a", sw=0.8, rx=2)
    s.text(x0 + 31 * cw / 2, y0 - 14, "prompt: 31 digit tokens → prefill writes K, V for every position", size="small",
           role="a")
    for t in range(4):
        xx = x0 + (31 + t) * cw
        s.rect(xx, y0, cw - 2, 26, fill="f-c2", stroke="s-c", sw=0.8, rx=2)
    s.text(x0 + 35 * cw + 6, y0 + 13, "…", anchor="start", size="small", role="c")
    y1 = y0 + 70
    steps = [("new token t_p", "c"), ("embed, UnitRMS,\nq/k/v (d = 2)", "d"), ("attend to cached\nK, V (0 … p)", "b"),
             ("logits over\n10 digits", "c"), ("argmax →\nnext token", "hl")]
    end = kit.pipeline(s, x0 - 16, y1, steps, w=124, h=44, gap=16)
    s.path(f"M{end - 62:.1f},{y1 + 44:.1f} L{end - 62:.1f},{y1 + 70:.1f} L{x0 + 46:.1f},{y1 + 70:.1f} "
           f"L{x0 + 46:.1f},{y1 + 46:.1f}", stroke="s-hl", sw=1.2, arrow="hl")
    s.text((x0 + end) / 2, y1 + 84, "11 decode steps; K and V of the new token are appended to the cache", size="small",
           role="hl")
    return finish(s, note(s, x0, y1 + 116, ["Only the last position is computed per step (O(p) instead of O(p²)); the output is",
                                            "the logits of every step, [batch, 11, 10]. The model has 10 parameters."]))


def fig_078_2d_fft(name):
    s = Svg(name, W, 100, "2-D FFT by the row–column method: 1-D FFTs along rows, transpose, rows again")
    x0, y0, cw = 30, 50, 16
    stages = [("x (M × N)", lambda r, c: "f-a"), ("row FFTs", lambda r, c: "f-a2" if r == 2 else "f-a"),
              ("transpose", lambda r, c: "f-b"), ("row FFTs", lambda r, c: "f-b2" if r == 2 else "f-b"),
              ("transpose → X", lambda r, c: "f-c")]
    for i, (lab, fn) in enumerate(stages):
        xx = x0 + i * 138
        matrix(s, xx, y0, 6, 6, cw, fill_fn=fn, role="ink")
        s.text(xx + 3 * cw, y0 + 6 * cw + 14, lab, size="small")
        if i:
            s.arrow(xx - 40, y0 + 3 * cw, xx - 4, y0 + 3 * cw, role="muted", sw=1.1)
    return finish(s, note(s, 30, y0 + 6 * cw + 44, ["X_{uv} = Σ_{m} ω_{M}^{um} (Σ_{n} x_{mn} ω_{N}^{vn}): the inner sum is a DFT of each row, the outer",
                                                    "one a DFT of each column. Transposing (through shared-memory tiles) turns the column",
                                                    "pass into contiguous row FFTs, so every FFT pass reads memory coalesced."]))


def fig_080_grouped_query_attention(name):
    s = Svg(name, W, 100, "Grouped-query attention: G consecutive query heads share one K/V head")
    x0, y0 = 40, 50
    Hq, Hkv = 8, 2
    G = Hq // Hkv
    roles = ["a", "b"]
    for h in range(Hq):
        s.box(x0 + h * 80, y0, 70, 32, f"Q head {h}", role=roles[h // G], size="small")
    for g in range(Hkv):
        bx = x0 + g * 320 + 85
        s.box(bx, y0 + 110, 150, 34, f"K, V head {g}", role=roles[g], size="small", fill=f"f-{roles[g]}2")
        for h in range(g * G, (g + 1) * G):
            s.arrow(bx + 75, y0 + 108, x0 + h * 80 + 35, y0 + 34, role=roles[g], sw=1)
    return finish(s, note(s, 40, y0 + 180, [f"H_{{q}} = {Hq}, H_{{kv}} = {Hkv}, group size G = {G}: head h reads K/V head ⌊h / G⌋.",
                                            "The KV cache shrinks G×. A kernel can process a whole group per block so each",
                                            "K/V tile loaded from memory serves G query heads."]))


def fig_081_int4_matmul(name):
    s = Svg(name, W, 100, "W4A16: two 4-bit weights per byte, one fp16 scale per group of g weights")
    x0, y0 = 40, 50
    s.text(x0, y0 - 16, "one packed byte b = 0xA3", anchor="start", size="small", bold=True)
    bits = "10100011"
    for i, bt in enumerate(bits):
        s.box(x0 + i * 30, y0, 28, 28, bt, role="a" if i < 4 else "b", size="small", mono=True, rx=2)
    s.brace_h(x0, x0 + 118, y0 + 40, "high: q = 10", role="a")
    s.brace_h(x0 + 120, x0 + 238, y0 + 40, "low: q = 3", role="b")
    y1 = y0 + 90
    s.box(x0, y1, 118, 30, "(10 − 8)·s = 2s", role="a", size="small")
    s.box(x0 + 120, y1, 118, 30, "(3 − 8)·s = −5s", role="b", size="small")
    s.text(x0 + 119, y1 + 50, "element 2i (high), 2i + 1 (low)", size="small", role="muted")
    gx = 340
    s.text(gx, y0 - 16, "row n of W along K: groups of g share a scale", anchor="start", size="small", bold=True)
    for k in range(16):
        s.rect(gx + k * 22, y0, 20, 28, fill="f-d" if (k // 4) % 2 == 0 else "f-d2", stroke="s-d", sw=0.8, rx=2)
    for g in range(4):
        s.text(gx + g * 88 + 43, y0 + 44, f"s_{{n,{g}}}", size="small", role="d")
    s.text(gx + 176, y1 + 15, "y = x · Wᵀ, dequantised in registers", size="small", role="c")
    s.text(gx + 176, y1 + 38, "right before the tensor-core MMA", size="small", role="c")
    return finish(s, note(s, 40, y1 + 84, ["The weights cross memory as 4 bits each (4× less than fp16). Unpacking and scaling",
                                           "happen after the load, so the kernel reads a quarter of the bytes of an fp16 GEMM."]))


def fig_082_linear_recurrence(name):
    s = Svg(name, W, 100, "Linear recurrence h_t = a_t h_{t−1} + x_t as a scan over affine maps")
    x0, y0 = 40, 60
    for t in range(5):
        bx = x0 + t * 130
        s.box(bx, y0, 90, 34, f"h_{t}", role="c", size="small", bold=True)
        if t:
            s.arrow(bx - 38, y0 + 17, bx - 2, y0 + 17, role="muted", sw=1.2)
            s.text(bx - 20, y0 - 10, f"× a_{t}", size="small", role="b")
        s.arrow(bx + 45, y0 + 76, bx + 45, y0 + 36, role="a", sw=1.1)
        s.text(bx + 45, y0 + 88, f"+ x_{t}", size="small", role="a")
    y1 = y0 + 130
    note(s, x0, y1, ["Each step is the map f_t(h) = a_t h + x_t, stored as the pair (a_t, x_t). Composing two maps:",
                     "(A₁, X₁) then (A₂, X₂) = (A₁A₂, A₂X₁ + X₂) — associative, so a parallel scan applies:",
                     "per-thread chunks, a warp scan of the pairs, then the carry from earlier blocks."], role="ink")
    return finish(s, y1 + 40)


def fig_083_fused_residual_add_rms_norm(name):
    s = Svg(name, W, 100, "Fused add & RMSNorm: z = x + r stays on chip, only y is written")
    x0, y0 = 40, 50
    s.box(x0, y0, 110, 34, "x (sublayer out)", role="a", size="small")
    s.box(x0, y0 + 60, 110, 34, "r (residual)", role="b", size="small")
    s.box(x0 + 170, y0 + 30, 120, 34, "z = x + r", role="d", size="small", fill="f-d2")
    s.arrow(x0 + 112, y0 + 17, x0 + 168, y0 + 40, role="a", sw=1.1)
    s.arrow(x0 + 112, y0 + 77, x0 + 168, y0 + 56, role="b", sw=1.1)
    s.box(x0 + 350, y0 + 30, 140, 34, "rms_i = √(mean z² + ε)", role="d", size="tiny")
    s.arrow(x0 + 292, y0 + 47, x0 + 348, y0 + 47, role="d", sw=1.1)
    s.box(x0 + 530, y0 + 30, 110, 34, "y = z / rms · w", role="c", size="small")
    s.arrow(x0 + 492, y0 + 47, x0 + 528, y0 + 47, role="c", sw=1.1)
    s.rect(x0 + 160, y0 + 10, 340, 74, fill="fig-none", stroke="s-hl", sw=1.2, extra=' stroke-dasharray="5 3"')
    s.text(x0 + 330, y0 + 100, "registers / shared memory only", size="small", role="hl")
    return finish(s, note(s, 40, y0 + 140, ["One block per row: read x and r once, add, reduce Σz², normalise and write y.",
                                            "Unfused, z would be written and read back: 2 extra full-size memory passes."]))


def fig_084_swiglu_mlp_block(name):
    rows = [[("X (M × d)", "a"), ("gate GEMM\nG = X W_g", "b"), ("up GEMM\nU = X W_u", "b"), ("H = SiLU(G) ⊙ U\n(fused)", "d"),
             ("down GEMM\nY = H W_d", "b")],
            [("Y (M × d)", "c")]]
    return kit.layer_flow(name, "SwiGLU MLP: two input projections, a fused gate, one output projection", rows,
                          note_lines=["The gate and up GEMMs share X, so they run as one GEMM against [W_g | W_u]; SiLU(G)⊙U is",
                                      "applied in its epilogue, so G and U never reach memory. LLaMA-3 8B: d = 4096, d_f = 14336."])


def fig_085_lora_linear(name):
    s = Svg(name, W, 100, "LoRA: a frozen full-rank path plus a scaled rank-r update")
    x0, y0 = 40, 60
    s.box(x0, y0 + 40, 90, 40, "x (b × d_in)", role="a", size="small")
    s.box(x0 + 170, y0, 150, 40, "x Wᵀ (frozen, d_out)", role="b", size="small")
    s.box(x0 + 170, y0 + 80, 110, 40, "t = x Aᵀ (r)", role="d", size="small")
    s.box(x0 + 330, y0 + 80, 120, 40, "s · t Bᵀ (d_out)", role="d", size="small")
    s.box(x0 + 520, y0 + 40, 110, 40, "Y = sum", role="c", size="small", fill="f-c2")
    s.arrow(x0 + 92, y0 + 56, x0 + 168, y0 + 22, role="a", sw=1.1)
    s.arrow(x0 + 92, y0 + 66, x0 + 168, y0 + 98, role="a", sw=1.1)
    s.arrow(x0 + 282, y0 + 100, x0 + 328, y0 + 100, role="d", sw=1.1)
    s.arrow(x0 + 322, y0 + 20, x0 + 518, y0 + 56, role="b", sw=1.1)
    s.arrow(x0 + 452, y0 + 100, x0 + 518, y0 + 66, role="d", sw=1.1)
    return finish(s, note(s, 40, y0 + 160, ["Y = [x | s·xAᵀ] · [W | B]ᵀ: the small projection t (rank r = 64) is computed first and",
                                            "appended to the K dimension, so the whole layer is one GEMM with K = d_in + r."]))


# ----- 087-118 -----------------------------------------------------------------------------------------------

def fig_087_speculative_decoding_verification(name):
    s = Svg(name, W, 100, "Speculative decoding: accept draft tokens left to right, stop at the first rejection")
    x0, y0 = 40, 60
    toks = [("t0", 0.9, 0.62, True), ("t1", 0.5, 0.31, True), ("t2", 0.2, 0.74, False), ("t3", None, None, None)]
    for i, (t, a, u, acc) in enumerate(toks):
        bx = x0 + i * 150
        role = "c" if acc else ("hl" if acc is False else "line")
        s.box(bx, y0, 120, 34, f"draft {t}", role=role, size="small", fill=None if acc is not None else "fig-panel")
        if a is not None:
            s.text(bx + 60, y0 + 52, f"α = {fmt(a)}, u = {fmt(u)}", size="small", role=role)
            s.text(bx + 60, y0 + 70, "u < α: accept" if acc else "u ≥ α: reject", size="small", role=role, bold=True)
        else:
            s.text(bx + 60, y0 + 52, "never examined", size="small", role="muted")
    y1 = y0 + 110
    s.box(x0 + 300, y1, 260, 34, "resample from r ∝ max(0, q − p)", role="hl", size="small", fill="f-hl")
    s.arrow(x0 + 360, y0 + 80, x0 + 400, y1 - 2, role="hl", sw=1.2)
    s.text(x0, y1 + 17, "output row: t0, t1, r, 0 (zero-padded)", anchor="start", size="small", role="ink", bold=True)
    return finish(s, note(s, 40, y1 + 64, ["α_i = min(1, q_i(t_i) / p_i(t_i)) with target q and draft p. If all T drafts pass, a bonus",
                                           "token is drawn from q at position T. This rule makes the output distribution exactly the",
                                           "target model's. One block per sequence; the scan over positions stops early."]))


def fig_090_causal_depthwise_conv1d(name):
    xs = [2, 1, 3, 0, 2, 1, 4, 2]
    w = [0.5, 0.25, 0.25]  # w0 applies to the current position
    ys = []
    for l in range(len(xs)):
        ys.append(sum(w[k] * (xs[l - k] if l - k >= 0 else 0) for k in range(3)))
    return kit.window_1d(name, "Causal depthwise conv1d: position l of channel d sees only l−K+1 … l", xs, ys, 3,
                         out_index=4, pad=2, causal=True, weights=["w₂", "w₁", "w₀"],
                         op_text="y = β_d + Σ_{k} w_{d,k} x_{l−k}", x_name="x (1 ch.)",
                         note_lines=["Two zeros are padded on the left only, so no output looks into the future. Every channel",
                                     "d has its own K ≤ 8 taps; with channels-last storage, threads take consecutive d and",
                                     "each thread slides its window along l."])


def fig_092_decaying_causal_attention(name):
    g = 0.7

    def weight(i, j):
        v = g ** (i - j)
        return "f-a2" if v > 0.6 else ("f-a" if v > 0.3 else "f-d")
    return kit.attention(name, "Retention: causal scores scaled by γ^(n−m), no softmax", 9, 9, lambda i, j: j <= i,
                         weight=weight,
                         legend=[("f-a2", "a", "γ^{n−m} > 0.6 (recent)"), ("f-a", "a", "0.3 … 0.6"),
                                 ("f-d", "d", "< 0.3 (distant past)"), ("fig-panel", "line", "future: 0")],
                         pipeline_steps=["q_n · k_m / √d", "× γ^{n − m}", "O_n = Σ_{m} (…) v_m"],
                         row_lab="n", col_lab="m",
                         note_lines=["γ = 0.7 in the picture. The same result has a recurrent form S_n = γ S_{n−1} + k_nᵀ v_n with",
                                     "O(1) state per step; the parallel form here is a masked, decayed GEMM chain."])


def fig_093_llama_transformer_block(name):
    rows = [[("x", "a"), ("RMSNorm", "d"), ("Q, K, V GEMM\n(8 q, 2 kv heads)", "b"), ("RoPE on\nQ and K", "d"),
             ("causal GQA\nattention", "c")],
            [("O proj\n+ residual", "b"), ("RMSNorm", "d"), ("gate / up GEMM\nSiLU ⊙", "b"), ("down GEMM\n+ residual", "b"),
             ("y", "a")]]
    return kit.layer_flow(name, "LLaMA block: RMSNorm, RoPE, grouped-query causal attention, SwiGLU MLP", rows,
                          residuals=[(1, 1, 3, "residual")], bw=120, gap=12, x0=20,
                          note_lines=["d = 512, 8 query heads and 2 KV heads of width 64, MLP width 1408, no biases. Compared with",
                                      "GPT-2 every piece is the modern variant: RMSNorm, rotary positions, GQA and SwiGLU.",
                                      "The first residual (x added after the O projection) is fused into that GEMM's epilogue."])


def fig_094_ssm_selective_scan(name):
    s = Svg(name, W, 100, "Mamba selective scan: an input-dependent linear recurrence per (channel, state)")
    x0, y0 = 40, 60
    for t in range(4):
        bx = x0 + t * 160
        s.box(bx, y0, 110, 36, f"h_{t} (N states)", role="c", size="small")
        if t:
            s.arrow(bx - 48, y0 + 18, bx - 2, y0 + 18, role="muted", sw=1.2)
            s.text(bx - 25, y0 - 12, f"× Ā_{t}", size="small", role="b")
        s.arrow(bx + 55, y0 + 84, bx + 55, y0 + 38, role="a", sw=1.1)
        s.text(bx + 55, y0 + 96, f"+ B̄_{t} u_{t}", size="small", role="a")
    y1 = y0 + 140
    note(s, x0, y1, ["Ā = exp(Δ_t A), B̄ = Δ_t B_t (input-dependent), h_t = Ā h_{t−1} + B̄ u_t,",
                     "y_t = C_t · h_t + s · u_t.  One thread owns one (b, d, n) state and walks t sequentially",
                     "in registers; the N ≤ 64 states of a channel are reduced for y_t with warp shuffles."], role="ink")
    return finish(s, y1 + 40)


def fig_096_int8_kv_cache_attention(name):
    s = Svg(name, W, 100, "Decode attention over an int8 KV cache, split across blocks (flash-decoding)")
    x0, y0 = 40, 50
    s.box(x0, y0 + 40, 70, 36, "q_h", role="a", size="small", bold=True)
    for sp in range(4):
        bx = x0 + 120 + sp * 110
        for t in range(6):
            s.rect(bx + t * 16, y0, 14, 26, fill="f-b2", stroke="s-b", sw=0.8, rx=1)
        s.text(bx + 48, y0 + 40, f"split {sp}", size="small", role="b")
        s.box(bx, y0 + 60, 96, 40, f"(m, ℓ, a)_{sp}", role="d", size="small")
        s.arrow(bx + 48, y0 + 102, x0 + 330, y0 + 150, role="d", sw=1)
    s.text(x0 + 120 + 220, y0 - 14, "int8 K̂, V̂ with per-token scales κ, ν", size="small", role="b")
    s.box(x0 + 250, y0 + 152, 160, 34, "merge → o_h", role="c", size="small", fill="f-c2")
    return finish(s, note(s, 40, y0 + 220, ["s_j = κ_j (q · K̂_j) / √D: the scale is applied once per dot product, and V̂ rows are",
                                            "scaled by ν_j inside the weighted sum. Each block scans one split of the S cached tokens",
                                            "(8 GB/s-bound, not compute-bound); the partial softmax states merge exactly."]))


def fig_105_group_normalization(name):
    return kit.normalize(name, "GroupNorm: statistics per (sample, group of channels) over all H·W positions", 8, 8,
                         lambda r, c: r // 2, 1, ["sample n, group g = 1", "channels 2 … 3, all H·W", "μ_{n,g}, σ²_{n,g}"],
                         ["y = γ_c (x − μ_{n,g})", "  / √(σ²_{n,g} + ε) + β_c"],
                         row_names="C channels", col_names="H · W positions →",
                         note_lines=["C = 8 channels in G = 4 groups of 2 (rows), one sample shown. The (C/G)·H·W elements of a",
                                     "group are contiguous in NCHW, so one block reduces one group. G = 1 is LayerNorm, G = C",
                                     "InstanceNorm; γ and β stay per channel."])


def fig_106_token_embedding_layer(name):
    s = Svg(name, W, 100, "Token embedding: gather two table rows, add them, LayerNorm the sum")
    x0, y0, cw = 40, 50, 20
    for tbl, (lab, role, hl) in enumerate((("token table E_T (V × D)", "a", 3), ("position table E_P (P × D)", "b", 1))):
        xx = x0 + tbl * 230
        for r in range(6):
            for c in range(8):
                s.rect(xx + c * cw, y0 + r * 18, cw, 18, fill=f"f-{role}2" if r == hl else "fig-paper",
                       stroke="s-line", sw=0.6)
        s.rect(xx, y0, 8 * cw, 6 * 18, fill="fig-none", stroke=f"s-{role}", sw=1.3)
        s.text(xx + 4 * cw, y0 - 14, lab, size="small", role=role, bold=True)
        s.text(xx + 8 * cw + 8, y0 + hl * 18 + 9 - (12 if tbl == 0 else 0), "τ = 3" if tbl == 0 else "π = 1",
               anchor="start", size="tiny", role=role)
    s.box(x0 + 480, y0 + 20, 170, 34, "s = E_T[τ] + E_P[π]", role="d", size="small")
    s.box(x0 + 480, y0 + 90, 170, 34, "y = LayerNorm(s; γ, β)", role="c", size="small")
    s.arrow(x0 + 565, y0 + 56, x0 + 565, y0 + 88, role="d", sw=1.2)
    s.path(f"M{x0 + 8 * cw + 2},{y0 + 3 * 18 + 9} L{x0 + 8 * cw + 30},{y0 + 3 * 18 + 9} L{x0 + 8 * cw + 30},{y0 + 126} "
           f"L{x0 + 462},{y0 + 126} L{x0 + 462},{y0 + 44} L{x0 + 476},{y0 + 44}", stroke="s-a", sw=1.1, arrow="a")
    s.arrow(x0 + 230 + 8 * cw + 50, y0 + 18 + 9, x0 + 478, y0 + 30, role="b", sw=1)
    return finish(s, note(s, 40, y0 + 160, ["One block (or warp) per token: two coalesced row gathers of D floats, the sum kept in",
                                            "registers, a mean/variance reduction over D, and one write of the normalised row."]))


def _ppo_obj(r, A, eps=0.2):
    return min(r * A, min(max(r, 1 - eps), 1 + eps) * A)


def fig_107_ppo_clipped_surrogate_loss(name):
    return kit.function_plot(name, "PPO's clipped objective: no extra reward for moving the ratio beyond 1 ± ε",
                             [(lambda r: _ppo_obj(r, 1.0), "c", None), (lambda r: _ppo_obj(r, -1.0), "hl", None)],
                             (0, 2), (-2, 1.5), [0, 0.5, 0.8, 1, 1.2, 1.5, 2], [-2, -1, 0, 1],
                             ["min(r A, clip(r, 1 − ε, 1 + ε) A)", "ε = 0.2, r = π / π_{old} = exp(log π − log π_{old})"],
                             legend=[("f-c2", "c", "advantage A = +1"), ("f-hl", "hl", "advantage A = −1")],
                             side_lines=["The loss is −mean over all B·S tokens:", "an elementwise map (exp, clip, min)",
                                         "fused into one sum reduction."], xlabel="ratio r", ylabel="objective")


def fig_108_dpo_sequence_loss(name):
    return kit.function_plot(name, "DPO loss −log σ(z) = softplus(−z): small once the policy prefers the chosen answer",
                             [(lambda z: math.log1p(math.exp(-z)), "a", None)], (-4, 4), (0, 4.5), [-4, -2, 0, 2, 4],
                             [0, 1, 2, 3, 4], ["loss = mean softplus(−z)", "z = β[(ℓ⁺ − ℓ⁻) − (ℓ⁺_{ref} − ℓ⁻_{ref})]"],
                             points=[(0, math.log(2), "hl")],
                             side_lines=["red dot: z = 0 → log 2 ≈ 0.693.", "",
                                         "Stable form: softplus(x) =", "max(x, 0) + log(1 + e^{−|x|}).",
                                         "One thread per pair, then a mean."], xlabel="z", ylabel="loss")


def fig_109_grpo_surrogate_loss(name):
    s = Svg(name, W, 100, "GRPO: advantages are rewards standardised within each group of G responses")
    R = [1.0, 0.0, 0.5, 0.0]
    mu = sum(R) / len(R)
    sd = math.sqrt(sum((r - mu) ** 2 for r in R) / len(R))
    A = [(r - mu) / (sd + 1e-8) for r in R]
    x0, y0, cw = 150, 50, 90
    row_label(s, x0 - 12, y0 + 14, "reward R_{b,g}", role="a", bold=True)
    cells(s, x0, y0, [fmt(v) for v in R], cw=cw - 10, gap=10, role="a")
    s.text(x0 + 4 * cw + 10, y0 + 14, f"μ = {fmt(mu, 3)}, σ = {fmt(sd, 3)}", anchor="start", size="small", role="d")
    y1 = y0 + 60
    row_label(s, x0 - 12, y1 + 14, "advantage A", role="c", bold=True)
    cells(s, x0, y1, [fmt(v, 2) for v in A], cw=cw - 10, gap=10, role="c")
    y2 = y1 + 60
    row_label(s, x0 - 12, y2 + 30, "S tokens each", role="muted")
    for g in range(4):
        for t in range(6):
            s.rect(x0 + g * cw + t * 13, y2, 11, 56, fill="f-c" if A[g] >= 0 else "f-hl", stroke="s-line", sw=0.5)
        s.arrow(x0 + g * cw + 40, y1 + 30, x0 + g * cw + 40, y2 - 2, role="muted", sw=1)
    return finish(s, note(s, 40, y2 + 86, ["A_{b,g} is broadcast to all S tokens of response g, which then use the PPO clipped term plus",
                                           "a KL penalty K = e^d − d − 1 (d = log π_{ref} − log π). No critic network is needed."], lh=20))


def fig_110_gae_reverse_scan(name):
    d = [0.5, -0.2, 0.3, 1.0, -0.4, 0.2]
    c = 0.9
    A = [0.0] * len(d)
    acc = 0.0
    for t in range(len(d) - 1, -1, -1):
        acc = d[t] + c * acc
        A[t] = acc
    return kit.scan(name, "GAE: a discounted scan that runs from the end of the trajectory backwards", d, A,
                    "A_t = δ_t + c · A_{t+1}", x_name="δ", y_name="A", reverse=True, highlight=(2, 5, 2), note_lh=20,
                    note_lines=["δ_t = r_t + γ V_{t+1} − V_t, and c = γλ = 0.9 here. The pair (c, δ_t) is an affine map, so",
                                "the recurrence is a scan (run right to left): A_2 = Σ_{k} c^k δ_{2+k} over the bracketed inputs."])


def fig_111_softmax_attention_backward(name):
    s = Svg(name, W, 100, "Attention backward without storing P: recompute it tile by tile from Q, K and the LSE")
    x0, y0 = 30, 50
    boxes = {"Q, K, LSE": (x0, y0, "a"), "P = exp(QKᵀ/√d − LSE)": (x0 + 150, y0, "d"),
             "dV += Pᵀ dO": (x0 + 400, y0 - 20, "c"), "dP = dO Vᵀ": (x0 + 150, y0 + 80, "b"),
             "dS = P ⊙ (dP − D)": (x0 + 400, y0 + 60, "d"), "dQ += dS K / √d": (x0 + 400, y0 + 130, "c"),
             "dK += dSᵀ Q / √d": (x0 + 400, y0 + 190, "c"), "D = rowsum(dO ⊙ O)": (x0 + 150, y0 + 160, "b")}
    for lab, (bx, by, role) in boxes.items():
        s.box(bx, by, 200 if bx > x0 else 110, 34, lab, role=role, size="small")
    s.arrow(x0 + 112, y0 + 17, x0 + 148, y0 + 17, role="muted", sw=1.1)
    s.arrow(x0 + 352, y0 + 12, x0 + 398, y0 - 3, role="muted", sw=1.1)
    s.arrow(x0 + 352, y0 + 24, x0 + 398, y0 + 72, role="muted", sw=1.1)
    s.arrow(x0 + 352, y0 + 97, x0 + 398, y0 + 82, role="muted", sw=1.1)
    s.arrow(x0 + 352, y0 + 172, x0 + 398, y0 + 90, role="muted", sw=1.1)
    s.arrow(x0 + 500, y0 + 96, x0 + 500, y0 + 128, role="muted", sw=1.1)
    s.path(f"M{x0 + 602},{y0 + 77} L{x0 + 640},{y0 + 77} L{x0 + 640},{y0 + 207} L{x0 + 602},{y0 + 207}",
           stroke="s-muted", sw=1.1, arrow="muted")
    return finish(s, note(s, 40, y0 + 250, ["D_i = dO_i · O_i is precomputed per row. The forward pass saves only the row log-sum-exp,",
                                            "so the backward kernel rebuilds each P tile on the fly and never writes an M × N matrix."]))


def fig_112_attention_with_sinks(name):
    ns, w = 2, 3
    return kit.attention(name, "Attention with sinks: the first n_s tokens plus a sliding window of the last w", 12, 12,
                         lambda i, j: j <= i and (j < ns or j >= i - w + 1),
                         weight=lambda i, j: "f-b2" if j < ns else ("f-c2" if i == 9 else "f-a2"),
                         legend=[("f-b2", "b", f"sink tokens (n_s = {ns})"), ("f-a2", "a", f"window (w = {w})"),
                                 ("f-c2", "c", "row 9: keys 0, 1, 7, 8, 9"), ("fig-panel", "line", "evicted / future")],
                         pipeline_steps=["visit sink tiles + window tiles", "online softmax", "O_i = Σ p · v"],
                         note_lines=["The KV cache stays bounded (n_s + w entries) for any stream length; keeping the first tokens",
                                     "avoids the collapse of plain windowing, because they absorb a lot of attention mass."])


def fig_113_layer_normalization(name):
    return kit.normalize(name, "LayerNorm: statistics per row over its C features, then a per-feature affine", 6, 8,
                         lambda r, c: r, 2, ["row i = 2", "μ_{i} = mean of the C values", "σ²_{i} = mean (x − μ_{i})²",
                                              "(two passes over registers)"],
                         ["y_{ij} = w_{j} (x_{ij} − μ_{i})", "  / √(σ²_{i} + ε) + b_{j}"],
                         row_names="N rows", col_names="C features →",
                         note_lines=["One warp per row (C ≤ 4096): the row is held in registers, so the exact two-pass",
                                     "variance costs no extra memory traffic and avoids E[x²] − μ² cancellation."])


def fig_114_multi_head_latent_attention(name):
    s = Svg(name, W, 100, "MLA decode: attention runs in the compressed latent space of the KV cache")
    x0, y0 = 40, 50
    s.text(x0, y0 - 16, "one cache row per position t", anchor="start", size="small", bold=True)
    for k in range(10):
        s.rect(x0 + k * 22, y0, 20, 26, fill="f-b2", stroke="s-b", sw=0.8, rx=1)
    for k in range(3):
        s.rect(x0 + 226 + k * 22, y0, 20, 26, fill="f-d2", stroke="s-d", sw=0.8, rx=1)
    s.brace_h(x0, x0 + 218, y0 + 38, "latent c_t (width R)", role="b")
    s.brace_h(x0 + 226, x0 + 290, y0 + 38, "k_{t}^{pe} (r)", role="d")
    y1 = y0 + 90
    steps = [("q_h = [q^{nope} | q^{pe}]", "a"), ("q̃_h = q^{nope} W_{UK,h}\n(absorbed)", "a"),
             ("s = q̃·c_t + q^{pe}·k^{pe}_{t}\n× scale", "d"), ("softmax, then\no = Σ p c_t · W_{UV,h}", "c")]
    kit.pipeline(s, x0, y1, steps, w=150, h=46, gap=16)
    return finish(s, note(s, 40, y1 + 76, ["Per-head keys and values are never materialised: W_{UK} is folded into the query and W_{UV} is",
                                           "applied after the weighted sum. The cache holds R + r numbers per token instead of",
                                           "2 · heads · d_{head}, a >10× reduction. One block per head scans the whole cache."]))


def fig_116_dit_block(name):
    rows = [[("c (condition)", "a"), ("SiLU, Linear", "d"), ("β₁, γ₁, g₁,\nβ₂, γ₂, g₂", "d")],
            [("x", "a"), ("LN, no affine;\n(1 + γ₁)· + β₁", "d"), ("self-attention", "c"), ("x + g₁ ⊙ attn", "b")],
            [("LN, no affine;\n(1 + γ₂)· + β₂", "d"), ("MLP (GELU)", "c"), ("x + g₂ ⊙ mlp", "b"), ("y", "a")]]
    return kit.layer_flow(name, "DiT block with adaLN-Zero: the condition predicts scale, shift and residual gates", rows,
                          bw=130, gap=26, x0=20, breaks=(0,),
                          note_lines=["Top row, once per sample: the condition c yields six vectors that modulate the",
                                      "two LayerNorms (γ, β) and gate the two residual branches (g), so every sample in",
                                      "the batch is normalised differently. The gates start at zero (\"Zero\"), which makes",
                                      "a freshly initialised block the identity."])


def fig_118_vit_patch_embedding(name):
    s = Svg(name, W, 100, "ViT patch embedding: cut the image into P × P patches and project each one (a GEMM)")
    x0, y0, cw = 40, 50, 16
    roles = ["a", "b", "c", "d"]
    for r in range(8):
        for c in range(8):
            p = (r // 4) * 2 + (c // 4)
            s.rect(x0 + c * cw, y0 + r * cw, cw, cw, fill=f"f-{roles[p]}", stroke="s-line", sw=0.5)
    for k in (4,):
        s.line(x0 + k * cw, y0, x0 + k * cw, y0 + 8 * cw, stroke="s-ink", sw=1.4)
        s.line(x0, y0 + k * cw, x0 + 8 * cw, y0 + k * cw, stroke="s-ink", sw=1.4)
    s.rect(x0, y0, 8 * cw, 8 * cw, fill="fig-none", stroke="s-ink", sw=1.4)
    s.text(x0 + 4 * cw, y0 - 14, "image (C × H × W)", size="small", bold=True)
    s.text(x0 + 4 * cw, y0 + 8 * cw + 14, "P = 4: 2 × 2 patches", size="small", role="muted")
    fx = x0 + 8 * cw + 60
    s.arrow(fx - 50, y0 + 4 * cw, fx - 6, y0 + 4 * cw, role="muted", sw=1.2)
    for p in range(4):
        s.rect(fx, y0 + p * 32, 120, 26, fill=f"f-{roles[p]}2", stroke=f"s-{roles[p]}", sw=1, rx=2)
        s.text(fx + 60, y0 + p * 32 + 13, f"patch {p}: C·P² values", size="tiny")
    gx = fx + 150
    s.box(gx, y0 + 40, 110, 40, "× Wᵀ + β\n(C·P² → D)", role="d", size="small")
    s.arrow(fx + 122, y0 + 60, gx - 2, y0 + 60, role="muted", sw=1.2)
    tx = gx + 150
    s.box(tx, y0, 100, 24, "CLS + E_0", role="hl", size="tiny")
    for p in range(4):
        s.box(tx, y0 + 28 + p * 28, 100, 24, f"t_{p} + E_{p + 1}", role=roles[p], size="tiny")
    s.arrow(gx + 112, y0 + 60, tx - 2, y0 + 60, role="muted", sw=1.2)
    return finish(s, note(s, 40, y0 + 8 * cw + 44, ["A convolution with kernel = stride = P is exactly im2col + GEMM: the patch gather is a",
                                                    "pure index transform, fused into the GEMM's A-tile loads; bias and positional embeddings",
                                                    "are added in the epilogue, and row 0 of every image is the learned CLS token."]))
