"""One figure per Tensara problem, written to tensara/<dir>/figure.svg (see build_figures.py).

Like figures/leetgpu.py, every figure uses small concrete numbers computed
here, so the values shown are always consistent with the formulas.
"""
import math
import struct

from . import kit
from .kit import W, Svg, fmt, cells, row_label, note, matrix, finish, Plot
from .leetgpu import sigmoid, silu, gelu, _count_grid, _range_sum  # noqa: F401  (shared helpers)


def _act(name, title, f, xr, yr, xt, yt, lines, side, points=(), legend=None, extra=()):
    """Activation plot for the Tensara M × N elementwise problems."""
    return kit.function_plot(name, title, [*extra, (f, "a", None)], xr, yr, xt, yt, lines, points=points,
                             legend=legend, side_lines=[*side, "", "C = f(A) on an M × N matrix: float4",
                                                        "loads and stores, 4 elements per thread."])


# ----- activations -------------------------------------------------------------------------------------------

def fig_elu(name):
    elu = lambda x: x if x > 0 else math.exp(x) - 1
    return _act(name, "ELU: identity for x > 0, a smooth exponential that saturates at −α below", elu, (-4, 3),
                (-1.5, 3), [-4, -2, 0, 2], [-1, 0, 1, 2, 3], ["ELU(x) = x (x > 0)", "α (eˣ − 1) (x ≤ 0),  α = 1"],
                ["The dashed line is the limit −α.", "expm1f keeps e^x − 1 accurate", "near 0."],
                extra=[(lambda x: -1.0, "line", "4 3")])


def fig_gelu(name):
    tanh_gelu = lambda x: 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    return _act(name, "GELU with the tanh approximation (practically on top of the exact erf curve)", tanh_gelu,
                (-4, 3), (-0.5, 3), [-4, -2, 0, 2], [0, 1, 2, 3],
                ["GELU(x) = x Φ(x) ≈ ½ x (1 + tanh(u))", "u = √(2/π) (x + 0.044715 x³)"],
                ["dashed: the exact x·Φ(x); the curves", "differ by < 0.001 on this range."],
                extra=[(gelu, "line", "4 3")])


def fig_hard_sigmoid(name):
    hs = lambda x: min(1.0, max(0.0, x / 6 + 0.5))
    return _act(name, "Hard sigmoid: a straight ramp from (−3, 0) to (3, 1), clamped outside", hs, (-5, 5), (-0.2, 1.2),
                [-5, -3, 0, 3, 5], [0, 0.5, 1], ["hsig(x) = min(1, max(0, x/6 + ½))", "no exponential at all"],
                ["dashed: the smooth σ(x) for comparison."], points=[(-3, 0, "hl"), (3, 1, "hl")],
                extra=[(sigmoid, "line", "4 3")])


def fig_leaky_relu(name):
    return _act(name, "Leaky ReLU with a runtime slope α (α = 0.2 drawn, the largest test value)",
                lambda x: x if x > 0 else 0.2 * x, (-4, 3), (-1, 3), [-4, -2, 0, 2], [-1, 0, 1, 2, 3],
                ["C = max(x, 0) + α min(x, 0)", "α ∈ [0.01, 0.2] passed at run time"],
                ["x = −4 → −0.8 (red dot)."], points=[(-4, -0.8, "hl")])


def fig_relu(name):
    return _act(name, "ReLU on a matrix: the simplest bandwidth benchmark", lambda x: max(0.0, x), (-3, 3), (-1, 3),
                [-3, -2, -1, 0, 1, 2, 3], [-1, 0, 1, 2, 3], ["C = max(A, 0)", "one fmaxf per element"],
                ["8 bytes of traffic per element", "(one read, one write)."])


def fig_selu(name):
    lam, al = 1.0507009873554805, 1.6732632423543772
    selu = lambda x: lam * (x if x > 0 else al * (math.exp(x) - 1))
    return _act(name, "SELU: a scaled ELU whose constants make activations self-normalising", selu, (-4, 3),
                (-2, 3.5), [-4, -2, 0, 2], [-2, -1, 0, 1, 2, 3], ["SELU(x) = λ x (x > 0), λ α (eˣ − 1) (x ≤ 0)",
                                                                   "λ ≈ 1.0507, α ≈ 1.6733"],
                ["The dashed line is the limit −λα ≈ −1.758."], extra=[(lambda x: -lam * al, "line", "4 3")])


def fig_sigmoid(name):
    return _act(name, "Sigmoid on a matrix: σ(x) = 1 / (1 + e⁻ˣ)", sigmoid, (-8, 8), (0, 1), [-8, -4, 0, 4, 8],
                [0, 0.5, 1], ["C = 1 / (1 + e^{−A})", "range (0, 1),  σ(0) = ½"],
                ["Overflow of e⁻ˣ for very negative x", "gives 1/∞ = 0: the right limit."], points=[(0, 0.5, "hl")])


def fig_soft_plus(name):
    sp = lambda x: math.log1p(math.exp(x)) if x <= 20 else x
    return _act(name, "Softplus: a smooth ReLU, ln(1 + eˣ), switched to x above the threshold 20", sp, (-4, 4),
                (-0.5, 4.5), [-4, -2, 0, 2, 4], [0, 1, 2, 3, 4], ["softplus(x) = ln(1 + eˣ)", "= x for x > 20 (as PyTorch)"],
                ["dashed: ReLU. The gap at 0 is", "ln 2 ≈ 0.693 (red dot)."], points=[(0, math.log(2), "hl")],
                extra=[(lambda x: max(0.0, x), "line", "4 3")])


def fig_swish(name):
    return _act(name, "Swish (SiLU) on a matrix: x · σ(x)", silu, (-6, 4), (-1, 4), [-6, -4, -2, 0, 2, 4],
                [-1, 0, 1, 2, 3, 4], ["C = x σ(x) = x / (1 + e⁻ˣ)", "minimum ≈ −0.2785 at x ≈ −1.2785"],
                ["red dot: the minimum."], points=[(-1.2785, silu(-1.2785), "hl")])


def fig_tanh(name):
    return _act(name, "tanh squashes into (−1, 1); it is 2σ(2x) − 1", math.tanh, (-4, 4), (-1.2, 1.2),
                [-4, -2, 0, 2, 4], [-1, 0, 1], ["C = tanh(A) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)", "odd function, tanh(0) = 0"],
                ["tanhf is accurate across the whole", "range; no overflow handling needed."])


def fig_conv2d_relu_hardswish(name):
    def f(c):
        r = max(0.0, c)
        return r * min(6.0, max(0.0, r + 3)) / 6
    return kit.function_plot(name, "Conv2d, then ReLU, then HardSwish: the epilogue applied to each conv output",
                             [(f, "a", None)], (-4, 5), (-0.5, 5), [-4, -2, 0, 2, 4], [0, 1, 2, 3, 4, 5],
                             ["O = R · ReLU6(R + 3) / 6", "R = max(0, C),  C = conv2d(I, κ)"],
                             side_lines=["After ReLU, R ≥ 0, so HardSwish is", "R(R + 3)/6 for R < 3 and R for R ≥ 3.",
                                         "", "The \"same\" convolution is tiled with", "a shared-memory halo; both",
                                         "activations run in registers before", "the single store (fused)."],
                             xlabel="conv output C", ylabel="O")


# ----- elementwise and data movement ---------------------------------------------------------------------------

def fig_vector_addition(name):
    a = [1.5, 2, -1, 0.5, 3, 4, -2, 1]
    b = [0.5, 1, 2, 2.5, -1, 0, 3, 1]
    return kit.elementwise(name, "Vector addition at up to 2³⁰ elements: 64-bit indices and a grid-stride loop",
                           [("a", [fmt(v) for v in a], "a"), ("b", [fmt(v) for v in b], "b")],
                           ("c", [fmt(x + y) for x, y in zip(a, b)], "c"), "+",
                           note_lines=["n = 2³⁰ floats is 4 GB per vector: indices are 64-bit, and a fixed grid",
                                       "walks the array with a grid-stride loop of float4 loads and stores."])


def fig_matrix_scalar(name):
    A = [0.5, -2, 4, 1, -1, 3, 2.5, -0.5]
    return kit.elementwise(name, "Matrix × scalar: every element is multiplied by the same s (s = −0.3 here)",
                           [("A", [fmt(v) for v in A], "a")], ("C", [fmt(-0.3 * v, 2) for v in A], "c"), "× s",
                           note_lines=["The n × n matrix is contiguous, so it is treated as a flat array of n² floats",
                                       "(float4 accesses); s is a kernel argument held in a register."])


def fig_vector_multiply_ff(name):
    p = 2 ** 31 - 1
    a = [3, p - 1, 2 ** 30, 12345]
    b = [5, 2, 4, 67890]
    return kit.elementwise(name, "Multiplication in F_p, p = 2³¹ − 1: a 62-bit product folded without division",
                           [("a", [str(v) for v in a], "a"), ("b", [str(v) for v in b], "b")],
                           ("c", [str(x * y % p) for x, y in zip(a, b)], "c"), "×", cw=130, x0=150, warp_note=False,
                           note_lines=["x = a·b < 2⁶²; since 2³¹ ≡ 1 (mod p): x₁ = (x & p) + (x ≫ 31),",
                                       "x₂ = (x₁ & p) + (x₁ ≫ 31), then subtract p once if x₂ ≥ p.",
                                       "Example: (p − 1) · 2 = 2p − 2 ≡ p − 2 = 2147483645."])


def fig_ecc_point_negation(name):
    s = Svg(name, W, 100, "Elliptic-curve point negation: −(x, y) is the mirror image (x, −y mod p)")
    p = Plot(s, 60, 20, 330, 250, (-2.5, 4), (-9, 9))
    p.axes([-2, 0, 2, 4], [-8, -4, 0, 4, 8], "x", "y")
    x_min = -(7 ** (1 / 3))
    p.curve(lambda x: math.sqrt(x ** 3 + 7) if x ** 3 + 7 >= 0 else None, role="a")
    p.curve(lambda x: -math.sqrt(x ** 3 + 7) if x ** 3 + 7 >= 0 else None, role="a")
    s.line(p.px(x_min), p.py(-0.4), p.px(x_min), p.py(0.4), stroke="s-a", sw=2.2)
    px_ = 2.0
    py_ = math.sqrt(px_ ** 3 + 7)
    p.dot(px_, py_, role="c", r=5)
    p.dot(px_, -py_, role="hl", r=5)
    s.line(p.px(px_), p.py(py_) + 6, p.px(px_), p.py(-py_) - 6, stroke="s-muted", sw=1, dash="4 3")
    s.text(p.px(px_) + 10, p.py(py_ - 1.6), "P = (x, y)", anchor="start", size="small", role="c")
    s.text(p.px(px_) + 10, p.py(-py_ + 1.6), "−P = (x, −y)", anchor="start", size="small", role="hl")
    note(s, 430, 44, ["curve y² = x³ + 7 (drawn over the reals)", "", "over F_p, p = 2⁶¹ − 1:",
                      "−y mod p = (p − (y mod p)) mod p", "(y = 0 maps to 0, not to p)", "",
                      "one thread per point: read (x, y),", "write (x, p − y) interleaved;", "bandwidth-bound, exact."],
         role="ink")
    return finish(s, p.y + p.h + 36)


def fig_grayscale(name):
    s = Svg(name, W, 100, "Grayscale conversion of an HWC image: Y = 0.299 R + 0.587 G + 0.114 B")
    px = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
    x0, y0 = 40, 50
    roles = ["hl", "c", "a"]
    for p_, rgb in enumerate(px):
        for c in range(3):
            s.box(x0 + (3 * p_ + c) * 54, y0, 50, 28, str(rgb[c]), role=roles[c], size="small", rx=2)
        s.brace_h(x0 + 3 * p_ * 54, x0 + (3 * p_ + 3) * 54 - 4, y0 + 40, f"pixel {p_}", role="muted")
    y1 = y0 + 110
    for p_, (r, g, b) in enumerate(px):
        yv = 0.299 * r + 0.587 * g + 0.114 * b
        cx = x0 + (3 * p_ + 1) * 54 + 25
        s.arrow(cx, y0 + 64, cx, y1 - 2, role="muted", sw=1)
        s.box(cx - 40, y1, 80, 28, fmt(yv, 3), role="ink", size="small", rx=2)
    return finish(s, note(s, x0, y1 + 58, ["Pure red, green and blue give 76.245, 149.685 and 29.07: green dominates perceived",
                                           "brightness. Thread (i, j) reads the 3 interleaved floats at 3(iw + j) and writes one."]))


def fig_threshold(name):
    return kit.function_plot(name, "Binary thresholding: 255 where the pixel is strictly above θ, else 0",
                             [(lambda v: 255.0 if v > 128 else 0.0, "a", None)], (0, 255), (-20, 275),
                             [0, 64, 128, 192, 255], [0, 128, 255],
                             ["out = 255 if I > θ else 0", "θ = 128 drawn (tests: 64, 128, 192)"],
                             points=[(128, 0, "hl")], xlabel="input pixel I", ylabel="out",
                             side_lines=["red dot: I = θ maps to 0 (strictly", "greater is required).", "",
                                         "Exact comparison, float4 per", "thread: a pure streaming kernel."])


def fig_diagonal_matmul(name):
    s = Svg(name, W, 100, "diag(a) · B: never build the N × N diagonal matrix — just scale row i of B by a_i")
    a = [2, -1, 0.5, 3]
    B = [[1, 2, 0, -1, 3], [4, 0, 1, 2, -2], [2, 2, 6, 0, 4], [1, -1, 0, 1, 2]]
    x0, y0, cw = 40, 50, 34
    for i, v in enumerate(a):
        s.box(x0, y0 + i * cw, cw - 4, cw - 4, fmt(v), role="a", size="small", rx=2)
    s.text(x0 + 15, y0 - 14, "a", size="small", role="a", bold=True)
    bx = x0 + 90
    for i in range(4):
        for j in range(5):
            s.box(bx + j * cw, y0 + i * cw, cw - 4, cw - 4, fmt(B[i][j]), role="b", size="small", rx=2)
    s.text(bx + 2.5 * cw, y0 - 14, "B (N × M)", size="small", role="b", bold=True)
    s.text(x0 + 60, y0 + 2 * cw - 2, "⊙", size="big")
    cx = bx + 5 * cw + 70
    for i in range(4):
        for j in range(5):
            s.box(cx + j * cw, y0 + i * cw, cw - 4, cw - 4, fmt(a[i] * B[i][j]), role="c", size="small", rx=2)
    s.text(cx + 2.5 * cw, y0 - 14, "C_{ij} = a_i · B_{ij}", size="small", role="c", bold=True)
    s.arrow(bx + 5 * cw + 8, y0 + 2 * cw - 2, cx - 10, y0 + 2 * cw - 2, role="muted", sw=1.2)
    return finish(s, note(s, 40, y0 + 4 * cw + 30, ["The reference's torch.diag(A) @ B costs N³ FLOPs; the direct form is N·M multiplies",
                                                    "and pure streaming: one broadcast a_i per row, float4 loads of B."]))


# ----- losses --------------------------------------------------------------------------------------------------

def fig_hinge_loss(name):
    return kit.function_plot(name, "Hinge loss: zero once the prediction is on the right side with margin ≥ 1",
                             [(lambda m: max(0.0, 1 - m), "a", None)], (-2, 3), (-0.3, 3.2), [-2, -1, 0, 1, 2, 3],
                             [0, 1, 2, 3], ["ℓ_i = max(0, 1 − x_i y_i)", "y_i ∈ {−1, +1}"],
                             points=[(1, 0, "hl")], xlabel="margin x·y", ylabel="loss",
                             side_lines=["red dot: margin 1, where the loss", "reaches 0 (the SVM margin).", "",
                                         "Output is per element (no mean):", "one fused multiply, subtract and max."])


def fig_huber_loss(name):
    hub = lambda d: 0.5 * d * d if abs(d) < 1 else abs(d) - 0.5
    return kit.function_plot(name, "Smooth L1 (Huber, β = 1): quadratic near 0, linear beyond |d| = β",
                             [(lambda d: 0.5 * d * d, "line", "4 3"), (hub, "a", None)], (-3, 3), (-0.2, 3),
                             [-3, -2, -1, 0, 1, 2, 3], [0, 1, 2, 3],
                             ["z = d²/2 (|d| < 1), |d| − ½ otherwise", "d = x − y"],
                             points=[(-1, 0.5, "hl"), (1, 0.5, "hl")], xlabel="residual d", ylabel="loss",
                             legend=[("f-a2", "a", "Smooth L1"), ("fig-panel", "line", "d²/2 (dashed)")],
                             side_lines=["Red dots: the pieces meet with equal", "value (½) and slope (±1) at |d| = 1."])


def fig_kl_loss(name):
    pp = 0.3
    term = lambda q: pp * (math.log(pp) - math.log(q))
    return kit.function_plot(name, "One KL term p·(log p − log q) for p = 0.3, as the prediction q varies",
                             [(term, "a", None)], (0.01, 1), (-0.5, 1.2), [0.01, 0.3, 0.6, 1], [-0.5, 0, 0.5, 1],
                             ["out_i = p_i (log p_i − log q_i)", "both clamped at 10⁻¹⁰; 0 where p_i ≤ 0"],
                             points=[(0.3, 0, "hl")], xlabel="predicted q", ylabel="term",
                             side_lines=["red dot: q = p gives 0. A single", "term can be negative (q > p); only",
                                         "the sum over i is ≥ 0.", "", "The output is element-wise: the", "sum is not required."])


def fig_cosine_similarity(name):
    s = Svg(name, W, 100, "Cosine distance per row: 1 − (p · t) / (‖p‖ ‖t‖)")
    cx, cy = 150, 220
    pvec, tvec = (3.2, 1.2), (1.6, 2.8)
    sc = 50
    s.line(cx - 20, cy, cx + 200, cy, stroke="s-muted", sw=0.8)
    s.line(cx, cy + 20, cx, cy - 170, stroke="s-muted", sw=0.8)
    s.arrow(cx, cy, cx + pvec[0] * sc, cy - pvec[1] * sc, role="a", sw=2)
    s.arrow(cx, cy, cx + tvec[0] * sc, cy - tvec[1] * sc, role="b", sw=2)
    s.text(cx + pvec[0] * sc + 8, cy - pvec[1] * sc, "p_i", anchor="start", size="small", role="a", bold=True)
    s.text(cx + tvec[0] * sc + 8, cy - tvec[1] * sc - 6, "t_i", anchor="start", size="small", role="b", bold=True)
    a1, a2 = math.atan2(pvec[1], pvec[0]), math.atan2(tvec[1], tvec[0])
    r = 40
    s.path(f"M{cx + r * math.cos(a1):.1f},{cy - r * math.sin(a1):.1f} A{r},{r} 0 0 0 "
           f"{cx + r * math.cos(a2):.1f},{cy - r * math.sin(a2):.1f}", stroke="s-hl", sw=1.4)
    s.text(cx + 54 * math.cos((a1 + a2) / 2), cy - 54 * math.sin((a1 + a2) / 2), "θ", size="small", role="hl")
    dot = pvec[0] * tvec[0] + pvec[1] * tvec[1]
    cos = dot / math.hypot(*pvec) / math.hypot(*tvec)
    note(s, 400, 60, [f"p · t = {fmt(dot, 2)}", f"‖p‖ = {fmt(math.hypot(*pvec), 3)},  ‖t‖ = {fmt(math.hypot(*tvec), 3)}",
                      f"cos θ = {fmt(cos, 3)}", f"out = 1 − cos θ = {fmt(1 - cos, 3)}", "",
                      "One block per row: three sums", "(p·t, ‖p‖², ‖t‖²) in a single pass,", "with ε = 10⁻⁸ guarding the",
                      "product of the norms."], role="ink")
    return finish(s, cy + 30)


def fig_triplet_margin(name):
    s = Svg(name, W, 100, "Triplet margin loss: pull the positive closer than the negative by at least m")
    cx, cy = 200, 150
    a, pp, n = (0, 0), (1.3, 0.8), (-0.4, -2.1)
    sc = 50
    P = lambda q: (cx + q[0] * sc, cy - q[1] * sc)
    dp = math.hypot(pp[0] - a[0], pp[1] - a[1])
    dn = math.hypot(n[0] - a[0], n[1] - a[1])
    m = 1.0
    s.add(f'<circle cx="{cx}" cy="{cy}" r="{(dp + m) * sc:.1f}" class="fig-none s-hl" stroke-width="1.2" '
          f'stroke-dasharray="5 4"/>')
    s.line(*P(a), *P(pp), stroke="s-c", sw=1.6)
    s.line(*P(a), *P(n), stroke="s-b", sw=1.6)
    for q, role, lab in ((a, "a", "anchor"), (pp, "c", "positive"), (n, "b", "negative")):
        x, y = P(q)
        s.circle(x, y, 6, fill=f"k-{role}")
        s.text(x + 10, y - 12, lab, anchor="start", size="small", role=role, bold=True, plate=True)
    loss = max(0.0, dp - dn + m)
    note(s, 420, 50, [f"d(a, p) = {fmt(dp, 3)} (green)", f"d(a, n) = {fmt(dn, 3)} (orange)", f"margin m = {fmt(m)}",
                      f"ℓ = max(0, d_p − d_n + m) = {fmt(loss, 3)}", "",
                      "Red ring: radius d_p + m. A negative", "inside it still costs loss.", "",
                      "One block per triplet reduces the two", "squared distances (with +ε per term),",
                      "then the B losses are averaged."], role="ink")
    return finish(s, cy + (dp + m) * sc + 20)


# ----- reductions along one axis -------------------------------------------------------------------------------

_GRID = [[3, 1, 4, 1, 5, 9], [2, 6, 5, 3, 5, 8], [9, 7, 9, 3, 2, 3], [8, 4, 6, 2, 6, 4]]
_AXIS_NOTE = ["The tensor is viewed as x[o, j, i] = in[(oR + j)·I + i]: O = product of the axes before dim,",
              "R = the reduced length, I = product of the axes after it. Threads take consecutive i, so every",
              "step down j is a coalesced row read; with I = 1 a warp reduces one contiguous row instead."]


def _cols(fn):
    return [fn([_GRID[j][i] for j in range(len(_GRID))]) for i in range(len(_GRID[0]))]


def fig_sum_dim(name):
    return kit.reduce_axis(name, "Sum over one dimension (keepdim): reduce down j for every (o, i)", _GRID, "Σ over j",
                           [fmt(v) for v in _cols(sum)], note_lines=_AXIS_NOTE)


def fig_mean_dim(name):
    return kit.reduce_axis(name, "Mean over one dimension (keepdim): the column sums divided by R", _GRID, "Σ / R",
                           [fmt(v, 2) for v in _cols(lambda c: sum(c) / len(c))], note_lines=_AXIS_NOTE)


def fig_max_dim(name):
    return kit.reduce_axis(name, "Max over one dimension (keepdim): the largest value down each column", _GRID,
                           "max over j", [fmt(v) for v in _cols(max)], note_lines=_AXIS_NOTE)


def fig_min_dim(name):
    return kit.reduce_axis(name, "Min over one dimension (keepdim): the smallest value down each column", _GRID,
                           "min over j", [fmt(v) for v in _cols(min)], note_lines=_AXIS_NOTE)


def fig_product_dim(name):
    return kit.reduce_axis(name, "Product over one dimension (keepdim): multiply down each column", _GRID,
                           "∏ over j", [fmt(v) for v in _cols(lambda c: math.prod(c))],
                           note_lines=_AXIS_NOTE[:2] + ["step down j is a coalesced row read. Products over- and",
                                                        "underflow fast, hence the loose tolerance of this problem."])


def fig_argmax(name):
    return kit.reduce_axis(name, "Argmax over one dimension: the index j of the largest value (first on ties)", _GRID,
                           "argmax over j", [str(max(range(4), key=lambda j: (_GRID[j][i], -j))) for i in range(6)],
                           highlight_col=2,
                           note_lines=["Column i = 2 holds 4, 5, 9, 6 → index 2. Pairs (value, j) are merged with \"larger value,",
                                       "or equal value and smaller j\", which is associative, so any reduction tree returns",
                                       "the first occurrence like PyTorch. Output: int32 with the reduced dimension removed."])


def fig_argmin(name):
    return kit.reduce_axis(name, "Argmin over one dimension: the index j of the smallest value (first on ties)", _GRID,
                           "argmin over j", [str(min(range(4), key=lambda j: (_GRID[j][i], j))) for i in range(6)],
                           highlight_col=4,
                           note_lines=["Column i = 4 holds 5, 5, 2, 6 → index 2. The merge keeps the smaller value, or the",
                                       "smaller j on equal values, so the first occurrence always wins regardless of the order",
                                       "in which threads combine their partial results."])


def fig_softmax(name):
    col = [_GRID[j][1] for j in range(4)]
    m = max(col)
    ex = [math.exp(v - m) for v in col]
    return kit.reduce_axis(name, "Softmax along any dimension: max and sum down j, then normalise every element",
                           _GRID, "(m, s) online", ["", f"m = {fmt(m)}", "", "", "", ""], highlight_col=1,
                           note_lines=[f"Column i = 1: m = {fmt(m)}, s = Σ e^{{x − m}} = {fmt(sum(ex), 3)}, so the output column is "
                                       + ", ".join(fmt(v / sum(ex), 3) for v in ex) + ".",
                                       "The output keeps the full shape. Contiguous dims (I = 1) use a warp per row;",
                                       "strided dims let consecutive threads own consecutive i, each walking down j."])


# ----- whole-tensor reductions ---------------------------------------------------------------------------------

def fig_frobenius_norm(name):
    xs = [1, -2, 2, 0, 1, -1, 3, 2]
    return kit.reduce_tree(name, "Frobenius normalisation: reduce Σx² over the whole tensor, then divide every element",
                           [v * v for v in xs], lambda a, b: a + b, "x²", map_row=("x", [fmt(v) for v in xs]),
                           tail=("√", fmt(math.sqrt(sum(v * v for v in xs)), 3)), result_label="‖X‖_F",
                           final_lines=["Kernel 1 writes one partial Σx² per block (accumulated in float64); kernel 2 adds",
                                        "them and divides every element by the norm. Here y = x / 4.899."])


def fig_mse_loss(name):
    p = [1.0, 2.5, 0.0, 4.0, 3.0, 1.5, 2.0, 0.5]
    t = [1.5, 2.0, 1.0, 3.0, 3.0, 0.5, 2.5, 0.0]
    sq = [(a - b) ** 2 for a, b in zip(p, t)]
    return kit.reduce_tree(name, "MSE over a tensor of any shape: squared differences, one scalar",
                           sq, lambda a, b: a + b, "(x − y)²", map_row=("x, y", [f"{fmt(a)}, {fmt(b)}" for a, b in zip(p, t)]),
                           tail=("÷ n", fmt(sum(sq) / len(sq), 4)), value_fmt=lambda v: fmt(v, 3), result_label="MSE",
                           final_lines=["n = Π shape[k] (64-bit). Per-block partial sums in float64, then a second tiny",
                                        "kernel (or the last block) adds them and divides by n."])


def fig_matmul_sigmoid_sum(name):
    return kit.gemm(name, "Σ σ(AB): each block reduces σ over its own output tile, then one atomicAdd",
                    epilogue=["per 64 × 64 output tile t:", "S_t = Σ σ(G_{ij}) in registers", "→ block reduction",
                              "→ atomicAdd(result, S_t)", "G = AB is never stored"],
                    note_lines=["The float atomics make the order of the tile sums nondeterministic, which the loose",
                                "tolerance (rtol 5e-2) allows."])


# ----- scans ---------------------------------------------------------------------------------------------------

def fig_cumsum(name):
    xs = [3, 1, 4, 1, 5, 9, 2, 6]
    ys = [sum(xs[:i + 1]) for i in range(len(xs))]
    return kit.scan(name, "Cumulative sum: a chunked inclusive scan with carries between chunks", xs, ys,
                    "y_i = x_0 + … + x_i", highlight=(0, 6, 6), chunk=(4, ["chunk 0: carry 0", "chunk 1: carry 9"]),
                    note_lines=["Each chunk (2048 elements in the kernel) scans locally; the chunk totals are scanned",
                                "(in float64) and added back as carries. Written once for any associative operator."])


def fig_cumprod(name):
    xs = [2, 0.5, 3, 1, -1, 2, 0.5, 4]
    ys, acc = [], 1
    for v in xs:
        acc *= v
        ys.append(acc)
    return kit.scan(name, "Cumulative product: the same scan with ⊗ = × and identity 1", xs, ys, "y_i = x_0 · … · x_i",
                    highlight=(0, 4, 4), chunk=(4, ["chunk 0: carry 1", "chunk 1: carry 3"]),
                    note_lines=["Generic scan over (⊗, e): chunk products T_c, exclusive carries E_c = ⊗ of earlier T,",
                                "y_i = E_c ⊗ (local prefix). Long products under/overflow, hence the loose tolerance."])


def fig_running_sum_1d(name):
    s = Svg(name, W, 100, "Sliding-window sum from prefix sums: out[i] = Π[i + h + 1] − Π[i − h]")
    xs = [2, 1, 3, 0, 4, 1, 2, 5]
    Wn, h = 3, 1
    Pi = [0]
    for v in xs:
        Pi.append(Pi[-1] + v)
    out = [sum(xs[max(0, i - h):min(len(xs), i + h + 1)]) for i in range(len(xs))]
    x0, cw, y0 = 190, 56, 40
    t = 4
    row_label(s, x0 - 44, y0 + 14, "x", role="a", bold=True)
    cells(s, x0, y0, [fmt(v) for v in xs], cw=cw - 8, gap=8, role="a",
          fills=["f-a2" if t - h <= i <= t + h else None for i in range(len(xs))])
    y1 = y0 + 50
    row_label(s, x0 - 44, y1 + 14, "Π (exclusive)", role="d", bold=True)
    cells(s, x0 - cw / 2, y1, [fmt(v) for v in Pi], cw=cw - 8, gap=8, role="d",
          fills=["f-d2" if i in (t - h, t + h + 1) else None for i in range(len(Pi))])
    y2 = y1 + 50
    row_label(s, x0 - 44, y2 + 14, f"out (W = {Wn})", role="c", bold=True)
    cells(s, x0, y2, [fmt(v) for v in out], cw=cw - 8, gap=8, role="c",
          fills=["f-c2" if i == t else None for i in range(len(out))])
    return finish(s, note(s, 40, y2 + 58, [f"out[{t}] = Π[{t + h + 1}] − Π[{t - h}] = {Pi[t + h + 1]} − {Pi[t - h]} = {out[t]}. With W = 8191 the",
                                           "direct window costs W adds per output; the scan costs O(1). Zero padding is handled",
                                           "by clamping the two prefix indices to [0, N]; float64 keeps the differences exact."]))


# ----- normalisation -------------------------------------------------------------------------------------------

def fig_batch_norm(name):
    return kit.normalize(name, "BatchNorm2d: one mean and variance per channel f, over the batch and both spatial axes",
                         6, 8, lambda r, c: r % 2, 1, ["channel f = 1", "all rows (b, f = 1)", "N = B·D₁·D₂ values",
                                                        "float64 sums"],
                         ["y = (x − μ_f)", "  / √(σ_f² + ε)"], row_names="(b, f) planes", col_names="D₁·D₂ positions →",
                         note_lines=["Rows are the (b, f) planes of a (B = 3, F = 2) tensor; channel 1 is spread over three",
                                     "separate planes. No affine and no running statistics (training-mode BatchNorm2d)."])


def fig_layer_norm(name):
    return kit.normalize(name, "LayerNorm over (F, D₁, D₂): one mean and variance per sample, elementwise γ, β", 5, 10,
                         lambda r, c: r, 3, ["sample b = 3", "G = F·D₁·D₂ values", "μ_b, σ_b² (float64)"],
                         ["y = (x − μ_b)", "  / √(σ_b² + ε) · γ_g + β_g"], row_names="B samples",
                         col_names="G = F·D₁·D₂ →",
                         note_lines=["Each sample is one contiguous group of G elements: a block per sample reduces it,",
                                     "then applies the elementwise affine (γ and β have the same shape (F, D₁, D₂))."])


def fig_rms_norm(name):
    return kit.normalize(name, "RMSNorm per row: rescale by the root mean square, no centring, no weight", 5, 10,
                         lambda r, c: r, 1, ["row b = 1", "RMS_b =", "√(mean x² + ε)"], ["y_{bn} = x_{bn} / RMS_b"],
                         row_names="B rows", col_names="N features →",
                         note_lines=["One block per row: a single Σx² reduction (no mean pass, unlike LayerNorm), then one",
                                     "multiply per element by 1 / RMS_b."])


def fig_l1_norm(name):
    return kit.normalize(name, "L1 normalisation per row: divide by the sum of absolute values", 5, 10,
                         lambda r, c: r, 2, ["row b = 2", "s_b = Σ |x_{bd}|"], ["y_{bd} = x_{bd} / (s_b + ε)", "ε = 10⁻¹⁰"],
                         row_names="B rows", col_names="D values →",
                         note_lines=["After normalisation Σ|y| ≈ 1 for every row. One block per row, a block reduction of",
                                     "|x|, then a scaling pass that re-reads the row (it is still in L2)."])


def fig_l2_norm(name):
    return kit.normalize(name, "L2 normalisation per row: divide by the Euclidean norm", 5, 10,
                         lambda r, c: r, 2, ["row b = 2", "n_b = √(Σ x²_{bd})"], ["y_{bd} = x_{bd} / (n_b + ε)", "ε = 10⁻¹⁰"],
                         row_names="B rows", col_names="D values →",
                         note_lines=["Each output row is a unit vector (Σ y² ≈ 1). ε is added after the square root, exactly",
                                     "as in the reference, so rows of zeros stay zero instead of producing NaN."])


def fig_log_softmax(name):
    s = Svg(name, W, 100, "Log-softmax per row: subtract the row's log-sum-exp from every entry")
    xs = [1.0, 3.0, 2.0, 5.0, 0.0, 4.0]
    m = max(xs)
    lse = m + math.log(sum(math.exp(v - m) for v in xs))
    x0, cw = 130, 70
    rows = [("x (row i)", [fmt(v) for v in xs], "a"), ("y = x − LSE", [fmt(v - lse, 3) for v in xs], "c")]
    y = 40
    row_label(s, x0 - 14, y + 14, rows[0][0], role="a", bold=True)
    cells(s, x0, y, rows[0][1], cw=cw - 8, gap=8, role="a")
    s.box(x0 + 120, y + 56, 260, 30, f"LSE = m + ln Σ e^{{x−m}} = {fmt(lse, 3)}", role="d", size="small")
    y2 = y + 120
    row_label(s, x0 - 14, y2 + 14, rows[1][0], role="c", bold=True)
    cells(s, x0, y2, rows[1][1], cw=cw - 8, gap=8, role="c")
    s.arrow(x0 + 250, y + 88, x0 + 250, y2 - 2, role="d", sw=1.2)
    return finish(s, note(s, 40, y2 + 58, ["One warp per row finds (m, s) in a single online pass; ln is taken once per row, so the",
                                           "output never computes ln(e^x / Σ) directly and cannot underflow to −∞."]))


# ----- windows: convolution and pooling ------------------------------------------------------------------------

def fig_conv_1d(name):
    xs = [2, 1, 0, 3, 1, 2, 4, 1]
    w = [1, 2, 1]
    xp = [0] + xs + [0]
    ys = [sum(xp[i + j] * w[j] for j in range(3)) for i in range(len(xs))]
    return kit.window_1d(name, "\"Same\" 1-D convolution: zero padding r = (K−1)/2 on both sides, no flip", xs, ys, 3,
                         out_index=0, pad=1, weights=[fmt(v) for v in w],
                         note_lines=["Output 0 uses the padded zero on the left. With K = 8191 taps the work is N·K FMAs:",
                                     "blocks stage an input tile plus a K−1 halo and the kernel taps in shared memory."])


def fig_conv_2d(name):
    return kit.window_2d(name, "\"Same\" 2-D convolution: zero padding keeps the output the size of the input", 5, 7, 3,
                         out_rc=(0, 3), pad=1, out_rows=5, out_cols=7, op_text="C[i, j] = Σ_{u,v} Ã[i+u−p, j+v−p] · B[u, v]",
                         pad_note=[("fig-panel", "line", "zero padding"), ("f-hl", "hl", "tap on padding")],
                         note_lines=["Kernels go up to 127 × 127, so the kernel is streamed through shared memory in",
                                     "chunks with the matching input halo: shared-memory use stays bounded for any K."])


def fig_conv_square_3d(name):
    s = Svg(name, W, 100, "\"Same\" 3-D convolution: a K³ box around each voxel, zero outside the volume")
    cw = 20
    x0, y0 = 40, 50
    for d in range(3):
        xx = x0 + d * 150
        tot = 7
        matrix(s, xx, y0, tot, tot, cw,
               fill_fn=lambda r, c, d=d: ("f-a2" if d and 0 <= r - 1 < 5 and 0 <= c - 1 < 5 else "f-hl")
               if (r <= 2 and c <= 2) else (None if 1 <= r <= 5 and 1 <= c <= 5 else "fig-panel"), role="a")
        s.rect(xx, y0, 3 * cw, 3 * cw, fill="fig-none", stroke="s-hl", sw=2)
        s.text(xx + 3.5 * cw, y0 - 14, ["slice z − 1 (padding)", "slice z", "slice z + 1"][d], size="small",
               role="muted" if d == 0 else "a")
    ox = x0 + 3 * 150 + 20
    matrix(s, ox, y0 + cw, 5, 5, cw, fill_fn=lambda r, c: "f-c2" if (r, c) == (0, 0) else None, role="c")
    s.text(ox + 2.5 * cw, y0 + cw - 14, "output slice z", size="small", role="c")
    return finish(s, note(s, 40, y0 + 7 * cw + 30, ["Output voxel (0, 0, 0) with K = 3: its window reaches one plane, row and column",
                                                    "outside the volume (red and grey cells read 0). p = (K−1)/2; K ≤ 11 gives ≤ 1331 taps."]))


def fig_box_blur(name):
    taps = {(r, c) for r in range(0, 2) for c in range(0, 2)}
    return kit.window_2d(name, "Box blur: average of the pixels that exist inside the K × K window", 5, 7, 3,
                         out_rc=(0, 0), taps=taps | {(r, c) for r in range(-1, 2) for c in range(-1, 2)} - taps
                         if False else taps, out_rows=5, out_cols=7, op_text="out = sum of existing pixels / N_{ij}",
                         note_lines=["At the corner a 3 × 3 window covers only 4 pixels, so the divisor is N = 4, not 9:",
                                     "nothing is padded. Separable form: a horizontal running sum, then a vertical one."])


def fig_edge_detect(name):
    taps = {(1, 3), (3, 3), (2, 2), (2, 4)}
    return kit.window_2d(name, "Edge detection: central differences, gradient magnitude, rescaled so the max is 255",
                         6, 8, 3, out_rc=(2, 3), taps=taps, op_text="M = √(G_x² + G_y²)",
                         note_lines=["G_x = (I[i, j+1] − I[i, j−1]) / 2 and G_y likewise from the pixels above and below;",
                                     "border pixels are 0. A global max-reduction (float atomicMax on the bit pattern, since",
                                     "M ≥ 0) runs first, then a second pass writes 255 · M / M_{max}."])


def fig_avg_pool_1d(name):
    xs = [2, 4, 6, 1, 3, 5, 8]
    k, S, P = 3, 2, 1
    xp = [0] * P + xs + [0] * P
    Ho = (len(xs) + 2 * P - k) // S + 1
    ys = [sum(xp[S * i + m] for m in range(k)) / k for i in range(Ho)]
    return kit.window_1d(name, "1-D average pooling: window k, stride S, zero padding P, divisor always k", xs, ys, k,
                         stride=S, pad=P, out_index=1, op_text="Σ / k",
                         note_lines=[f"k = {k}, S = {S}, P = {P}: H_{{out}} = ⌊(7 + 2 − 3)/2⌋ + 1 = {Ho}. Output 1 averages x[1 … 3].",
                                     "count_include_pad = True: padded zeros are in the sum and in the divisor k."])


def fig_avg_pool_2d(name):
    return kit.window_2d(name, "2-D average pooling: k × k windows at stride S; padding counts as zeros", 6, 6, 3,
                         out_rc=(0, 1), stride=2, pad=1, out_rows=3, out_cols=3, op_text="Σ / k²  (padding included)",
                         pad_note=[("fig-panel", "line", "zero padding"), ("f-hl", "hl", "padded tap (0, still ÷ 9)")],
                         note_lines=["k = 3, S = 2, P = 1: H_{out} = ⌊(6 + 2 − 3)/2⌋ + 1 = 3. The divisor is k² even where the",
                                     "window overlaps the padding (count_include_pad = True, PyTorch's default)."])


def fig_avg_pool_3d(name):
    s = Svg(name, W, 100, "3-D average pooling: a k³ box at stride S; the divisor is always k³")
    cw = 20
    x0, y0 = 40, 50
    for d in range(3):
        xx = x0 + d * 150
        matrix(s, xx, y0, 6, 6, cw, fill_fn=lambda r, c: "f-a2" if 2 <= r <= 4 and 2 <= c <= 4 else None, role="a")
        s.rect(xx + 2 * cw, y0 + 2 * cw, 3 * cw, 3 * cw, fill="fig-none", stroke="s-hl", sw=2)
        s.text(xx + 3 * cw, y0 - 14, f"depth {2 + d}", size="small", role="a")
    ox = x0 + 3 * 150 + 30
    matrix(s, ox, y0 + cw, 3, 3, cw, fill_fn=lambda r, c: "f-c2" if (r, c) == (1, 1) else None, role="c")
    s.text(ox + 1.5 * cw, y0 + cw - 14, "output (depth 1)", size="small", role="c")
    return finish(s, note(s, 40, y0 + 6 * cw + 30, ["k = 3, S = 2, P = 0 shown: output (1, 1, 1) averages input depths, rows and columns 2 … 4",
                                                    "(27 values, ÷ 27). One thread per output voxel; D is the contiguous axis."]))


def fig_max_pool_1d(name):
    xs = [2, 7, 1, 5, 3, 8, 4, 6, 0]
    k, S, P, dl = 3, 2, 1, 2
    xp = [-math.inf] * P + xs + [-math.inf] * P
    Ho = (len(xs) + 2 * P - dl * (k - 1) - 1) // S + 1
    ys = [max(xp[S * i + dl * m] for m in range(k)) for i in range(Ho)]
    return kit.window_1d(name, "1-D max pooling with dilation: taps are δ apart, padding never wins", xs, ys, k,
                         stride=S, pad=P, dilation=dl, out_index=1, op_text="max", pad_label="−∞",
                         note_lines=[f"k = {k}, S = {S}, P = {P}, δ = {dl}: the window spans δ(k−1) + 1 = 5 positions, so",
                                     f"H_{{out}} = ⌊(9 + 2 − 4 − 1)/2⌋ + 1 = {Ho}. Output 1 = max(x[1], x[3], x[5]) = {ys[1]}."])


def fig_max_pool_2d(name):
    taps = {(1 + 2 * a, 3 + 2 * b) for a in range(2) for b in range(2)}
    return kit.window_2d(name, "2-D max pooling with dilation δ = 2: a spread-out k × k window", 6, 6, 2,
                         out_rc=(1, 1), stride=2, pad=1, taps=taps, out_rows=3, out_cols=3, op_text="max over the 4 taps",
                         pad_note=[("fig-panel", "line", "padding (−∞, never wins)")],
                         note_lines=["k = 2, S = 2, P = 1, δ = 2: the window spans δ(k−1) + 1 = 3 cells per axis but reads",
                                     "only k² = 4 of them. X_{out} = ⌊(X + 2P − δ(k−1) − 1)/S⌋ + 1 = 3."])


def fig_max_pool_3d(name):
    s = Svg(name, W, 100, "3-D max pooling with dilation: k³ taps spread δ apart in every direction")
    cw = 20
    x0, y0 = 40, 50
    for d in range(3):
        xx = x0 + d * 150
        on = d != 1
        matrix(s, xx, y0, 6, 6, cw, fill_fn=lambda r, c: "f-a2" if on and r in (1, 3) and c in (2, 4) else None,
               role="a" if on else "line")
        s.text(xx + 3 * cw, y0 - 14, f"depth {1 + d}" + ("" if on else " (skipped)"), size="small",
               role="a" if on else "muted")
    ox = x0 + 3 * 150 + 30
    matrix(s, ox, y0 + cw, 3, 3, cw, fill_fn=lambda r, c: "f-c2" if (r, c) == (0, 1) else None, role="c")
    s.text(ox + 1.5 * cw, y0 + cw - 14, "output", size="small", role="c")
    return finish(s, note(s, 40, y0 + 6 * cw + 30, ["k = 2, δ = 2: the 8 taps sit at depths 1 and 3, rows 1 and 3, columns 2 and 4 (every",
                                                    "other cell). Padded positions act as −∞, so they never win the max."]))


# ----- graphs --------------------------------------------------------------------------------------------------

_NODES = {0: (80, 80), 1: (230, 50), 2: (380, 90), 3: (130, 220), 4: (290, 200), 5: (430, 230)}


def fig_shortest_path(name):
    s = Svg(name, W, 100, "Single-source shortest paths by Bellman–Ford: relax every edge until nothing changes")
    E = [(0, 1, 4), (0, 3, 2), (3, 1, 1), (1, 2, 5), (3, 4, 7), (1, 4, 3), (4, 2, 1), (4, 5, 2), (2, 5, 6)]
    dist = {0: 0}
    for _ in range(6):
        for u, v, w in E:
            if u in dist and dist[u] + w < dist.get(v, math.inf):
                dist[v] = dist[u] + w
    tree = set()
    for u, v, w in E:
        if u in dist and dist[u] + w == dist.get(v):
            tree.add((u, v))
    kit.graph(s, _NODES, E, directed=True, hl_edges=tree, node_role={0: "hl"})
    yb = 300
    row_label(s, 110, yb + 14, "vertex v", role="muted")
    row_label(s, 110, yb + 48, "distance d", role="c", bold=True)
    for v in range(len(_NODES)):
        s.box(124 + v * 50, yb, 44, 28, str(v), role="a", size="small", rx=2)
        s.box(124 + v * 50, yb + 34, 44, 28, str(dist.get(v, "−1")), role="c", size="small", rx=2)
    note(s, 490, 60, ["source s = 0 (red); red edges", "give the shortest-path tree.", "",
                      "Each sweep: one thread per", "target v takes min over u of", "d[u] + a[u][v] (a column of the",
                      "adjacency matrix). Stop early", "when a sweep changes nothing;", "unreachable vertices get −1."],
         role="ink")
    return finish(s, 362)


def fig_min_spanning_tree(name):
    s = Svg(name, W, 100, "Minimum spanning tree by Prim: repeatedly add the cheapest edge leaving the tree")
    E = [(0, 1, 4), (0, 3, 2), (3, 1, 1), (1, 2, 5), (3, 4, 7), (1, 4, 3), (4, 2, 1), (4, 5, 2), (2, 5, 6)]
    adj = {}
    for u, v, w in E:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    intree, mst, tot = {0}, set(), 0
    while len(intree) < len(_NODES):
        best = min(((w, u, v) for u in intree for v, w in adj[u] if v not in intree))
        w, u, v = best
        intree.add(v)
        mst.add((u, v))
        tot += w
    kit.graph(s, _NODES, E, hl_edges=mst, hl_role="c")
    note(s, 490, 60, [f"green: the MST, total weight {tot}", "", "Prim on a dense matrix: key[v] =", "cheapest edge from the tree to v.",
                      "Each of the n steps is an argmin", "over n keys (one block, shared-", "memory reduction) plus a key",
                      "update from the new vertex's row.", "Disconnected graph → +∞."], role="ink")
    return finish(s, 262)


def fig_all_pairs_shortest_path(name):
    s = Svg(name, W, 100, "All-pairs shortest paths: Floyd–Warshall with 0 = no edge and −1 = unreachable")
    nodes = {0: (70, 70), 1: (200, 70), 2: (200, 190), 3: (70, 190)}
    E = [(0, 1, 3), (1, 2, 2), (0, 3, 7), (2, 3, 1)]
    kit.graph(s, nodes, E, directed=True)
    INF = math.inf
    n = 4
    d = [[0 if i == j else INF for j in range(n)] for i in range(n)]
    for u, v, w in E:
        d[u][v] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    x0, y0, cw = 320, 50, 40
    for j in range(n):
        s.text(x0 + j * cw + cw / 2, y0 - 12, str(j), size="small", role="muted")
    for i in range(n):
        s.text(x0 - 10, y0 + i * cw + cw / 2, str(i), anchor="end", size="small", role="muted")
        for j in range(n):
            v = d[i][j]
            s.box(x0 + j * cw, y0 + i * cw, cw, cw, "−1" if v == INF else fmt(v), role="hl" if v == INF else "c",
                  fill="fig-panel" if v == INF else None, size="small", rx=0)
    note(s, x0 + 4 * cw + 24, y0 + 10, ["output distances", "(row = from, col = to)", "", "d(0, 3) = min(7, 3+2+1) = 6",
                                         "grey: unreachable → −1"], role="ink")
    return finish(s, note(s, 40, y0 + 4 * cw + 34, ["Input 0 off the diagonal means \"no edge\" and becomes +∞ before the k-loop; +∞ left at the",
                                                    "end becomes −1. The k-loop is the blocked three-phase kernel (pivot, pivot row/column, rest)."]))


# ----- matrix products -----------------------------------------------------------------------------------------

def fig_matrix_multiplication(name):
    return kit.gemm(name, "SGEMM: 64 × 64 block tiles, register-blocked per thread, no tensor cores", tile=(2, 1),
                    epilogue=["block: 64 × 64 C tile", "stage A and B slices (K = 16)", "  in shared memory",
                              "thread: 8 × 8 outputs in", "  registers (register blocking)", "exact FP32 FMAs"],
                    note_lines=["rtol = 2e-4 rules out TF32 tensor cores (10-bit mantissa): this is a true FP32 SGEMM."])


def fig_square_matmul(name):
    return kit.gemm(name, "Square SGEMM: the tiled kernel with M = N = K; the test sizes divide by 64", tile=(1, 2),
                    dims=("N", "N", "N"),
                    epilogue=["N = 6144, 7168, 9216 are", "multiples of 64: no partial", "tiles, no bounds checks",
                              "in the inner loop", "error ≤ γ_N Σ |A||B|"],
                    note_lines=["The same shared-memory + register-blocked kernel as Matrix Multiplication."])


def fig_symmetric_matmul(name):
    return kit.gemm(name, "Symmetric inputs, general output: AB is symmetric only if A and B commute",
                    dims=("N", "N", "N"),
                    epilogue=["A = Aᵀ and B = Bᵀ, but", "Cᵀ = BA ≠ AB in general,", "so all N² outputs are",
                              "computed with the general", "tiled SGEMM"],
                    note_lines=["Symmetry could halve the input reads, but the output has no symmetry to exploit."])


def fig_lower_trig_matmul(name):
    return kit.gemm(name, "Lower-triangular product: skip every tile that is structurally zero", tile=(2, 1), kstep=1,
                    a_mask=lambda r, c: c > r, b_mask=lambda r, c: c > r, c_mask=lambda r, c: c > r,
                    dims=("N", "N", "N"),
                    epilogue=["C_{ij} = Σ_{k=j}^{i} A_{ik} B_{kj}", "tiles above the diagonal:", "  write zeros, no work",
                              "k-loop runs only over", "  [c₀, r₀ + 64)", "≈ 1/6 of the dense FLOPs"],
                    note_lines=["Grey cells are the zero triangles. A_{ik} ≠ 0 needs k ≤ i and B_{kj} ≠ 0 needs k ≥ j."])


def fig_upper_trig_matmul(name):
    return kit.gemm(name, "Upper-triangular product: the k-range of a tile shrinks to [r₀, c₀ + 64)", tile=(1, 2), kstep=1,
                    a_mask=lambda r, c: c < r, b_mask=lambda r, c: c < r, c_mask=lambda r, c: c < r,
                    dims=("N", "N", "N"),
                    epilogue=["C_{ij} = Σ_{k=i}^{j} A_{ik} B_{kj}", "tiles below the diagonal:", "  write zeros, no work",
                              "k starts at r₀ and stops", "  at min(N, c₀ + 64)"],
                    note_lines=["Grey cells are the zero triangles; the product of upper-triangular matrices is upper triangular."])


def fig_matmul_3d(name):
    s = kit.gemm(name, "3-D tensor × matrix: fold the batch into the rows and run one big GEMM", batch=2,
                 a_label="A as (N·M) × K", dims=("N·M", "L", "K"),
                 epilogue=["C[b, i, l] = Σ_k A[b, i, k] B[k, l]", "B is shared by every batch", "and A's first two axes",
                           "are contiguous: row ρ = bM + i", "→ one (N·M) × K by K × L GEMM"],
                 note_lines=["No batched kernel is needed: reshaping is free because it changes no memory."])
    return s


def fig_matmul_4d(name):
    return kit.gemm(name, "einsum(\"bijl,lk->bijk\"): flatten the free indices b, i, j into one row index", batch=2,
                    a_label="A as (B·I·J) × L", b_label="W", dims=("B·I·J", "K", "L"),
                    epilogue=["ρ = (bI + i)J + j", "C[ρ, k] = Σ_l A[ρ, l] W[l, k]", "all free indices of A come",
                              "before the contracted one,", "so this is a plain GEMM"],
                    note_lines=["Largest test: (16·256·512) × 256 times 256 × 768."])


def fig_gemm_relu(name):
    return kit.gemm(name, "Linear layer + ReLU: an NT GEMM with bias and ReLU fused into the epilogue",
                    a_label="A", b_label="Wᵀ", c_label="C", dims=("B", "M", "N"),
                    epilogue=["Z = A Wᵀ (W stored M × N)", "epilogue in registers:", "  C = max(Z + b, 0)",
                              "Z is never written", "one pass over C"],
                    note_lines=["W's nn.Linear layout makes the B operand transposed (\"NT\"); the tile loader handles it."])


def fig_gemm_multiply_leakyrelu(name):
    return kit.gemm(name, "GEMM, elementwise multiply, LeakyReLU: all after the tile product, in registers",
                    epilogue=["G = A B (registers)", "H = G ⊙ C (load C tile)", "O = H if H ≥ 0 else α H",
                              "one read of C and one", "write of O per element"])


def fig_matmul_swish(name):
    return kit.gemm(name, "Linear layer + Swish + scale: z = x Wᵀ + b, out = s · z · σ(z)",
                    a_label="x", b_label="Wᵀ", c_label="out", dims=("B", "out", "in"),
                    epilogue=["z = x Wᵀ + b (registers)", "out = s · z / (1 + e^{−z})", "fused epilogue:",
                              "no intermediate tensor"])


def fig_matmul_swish_scaling(name):
    return kit.gemm(name, "O = scale · swish(AB): a plain GEMM with a Swish-and-scale epilogue",
                    epilogue=["G = A B (registers)", "O = scale · G · σ(G)", "σ(t) = 1 / (1 + e^{−t})", "written once"])


def fig_matrix_power(name):
    s = Svg(name, W, 100, "A^P for P = 8 by repeated squaring: 3 GEMMs instead of 7")
    x0, y0 = 60, 60
    pw = ["A", "A²", "A⁴", "A⁸"]
    for i, lab in enumerate(pw):
        bx = x0 + i * 160
        s.box(bx, y0, 90, 38, lab, role="c" if i == 3 else "a", bold=True, fill="f-c2" if i == 3 else None)
        if i:
            s.arrow(bx - 68, y0 + 19, bx - 2, y0 + 19, role="a", sw=1.2)
            s.text(bx - 35, y0 + 6, "square", size="tiny", role="a")
    return finish(s, note(s, 40, y0 + 80, ["#GEMMs = ⌊log₂P⌋ + popcount(P) − 1: P = 2, 4, 8 need 1, 2, 3 GEMMs. P = 0 returns I.",
                                           "The squaring order is chosen to match the reference's rounding (rtol 1e-4)."]))


def fig_matrix_vector(name):
    s = Svg(name, W, 100, "Matrix-vector product: one warp streams one row of A with float4 loads")
    x0, y0, cw = 110, 60, 26
    R, C = 5, 16
    for c in range(C):
        s.rect(x0 + c * cw, 26, cw - 2, 22, fill="f-b" if (c // 4) % 2 == 0 else "f-b2", stroke="s-b", sw=0.7, rx=1)
    row_label(s, x0 - 10, 37, "b", role="b", bold=True)
    for r in range(R):
        for c in range(C):
            f = ("f-a2" if (c // 4) % 2 == 0 else "f-a") if r == 2 else "fig-panel"
            s.rect(x0 + c * cw, y0 + r * 26, cw - 2, 24, fill=f, stroke="s-a" if r == 2 else "s-line", sw=0.7, rx=1)
    row_label(s, x0 - 10, y0 + 2 * 26 + 12, "warp 2", role="a", bold=True)
    for q in range(4):
        s.text(x0 + q * 4 * cw + 2 * cw - 1, y0 + 5 * 26 + 12, f"lane {q}", size="tiny", role="a")
    cx = x0 + C * cw + 60
    for r in range(R):
        s.box(cx, y0 + r * 26, 40, 24, f"c{r}", role="c", size="tiny", rx=1, fill="f-c2" if r == 2 else None)
    s.arrow(x0 + C * cw + 6, y0 + 2 * 26 + 12, cx - 6, y0 + 2 * 26 + 12, role="a", sw=1.2)
    return finish(s, note(s, 40, y0 + 5 * 26 + 40, ["Lane ℓ multiplies the float4 groups q ≡ ℓ (mod 32) of row i with the same groups of b,",
                                                    "then the warp sums the 32 partials with shuffles. Each byte of A is read once: the",
                                                    "kernel is bandwidth-bound, and b stays in L1/L2 for all rows."]))


def fig_poly_multiply_ff(name):
    s = Svg(name, W, 100, "Polynomial product = linear convolution: c_k adds every a_i b_j with i + j = k (mod p)")
    n, cw, x0, y0 = 5, 42, 110, 60
    k = 4
    for j in range(n):
        s.box(x0 + j * cw, y0 - 36, cw - 4, 28, f"b{j}", role="b", size="small", rx=2)
    for i in range(n):
        s.box(x0 - 44, y0 + i * cw + 6, 36, cw - 12, f"a{i}", role="a", size="small", rx=2)
        for j in range(n):
            on = i + j == k
            s.box(x0 + j * cw, y0 + i * cw, cw, cw, f"a{i}b{j}", role="c" if on else "line",
                  fill="f-c2" if on else "fig-paper", size="tiny", rx=0)
    s.rect(x0, y0, n * cw, n * cw, fill="fig-none", stroke="s-ink", sw=1.4)
    tx = x0 + n * cw + 40
    note(s, tx, y0, [f"green anti-diagonal: i + j = {k}", f"c_{k} = a₀b₄ + a₁b₃ + … + a₄b₀ (mod p)", "",
                     "p = 2³¹ − 1 (Mersenne): each 62-bit", "product folds with two shifts and adds,",
                     "and 2n − 1 outputs each sum ≤ n terms.", "", "Thread k sums its anti-diagonal with a",
                     "and b staged in shared memory; the", "sum is kept in 64 bits and reduced", "mod p every few terms."], role="ink")
    return finish(s, y0 + n * cw + 10)


# ----- sorting and histograms ----------------------------------------------------------------------------------

def fig_array_sort(name):
    s = Svg(name, W, 100, "Sorting signed int32 with radix sort: flip the sign bit so unsigned order = signed order")
    vals = [-5, 3, -1, 0, 7, -128]
    x0, y0 = 60, 40
    xs = [x0, x0 + 80, x0 + 220, x0 + 480]
    for h, xx in zip(["x", "bits(x)", "f(x) = bits ⊕ 0x80000000", "sorted rank"], xs):
        s.text(xx, y0, h, anchor="start", size="small", bold=True)
    keys = [((v & 0xFFFFFFFF) ^ 0x80000000) for v in vals]
    order = sorted(range(len(vals)), key=lambda i: keys[i])
    for i, v in enumerate(vals):
        yy = y0 + 28 + i * 24
        s.text(xs[0], yy, str(v).replace("-", "−"), anchor="start", size="small", role="a")
        s.text(xs[1], yy, f"0x{v & 0xFFFFFFFF:08X}", anchor="start", size="small", mono=True)
        s.text(xs[2], yy, f"0x{keys[i]:08X}", anchor="start", size="small", mono=True, role="c")
        s.text(xs[3], yy, str(order.index(i)), anchor="start", size="small", role="c")
    return finish(s, note(s, 40, y0 + 28 + len(vals) * 24 + 20,
                          ["Two's complement puts the negatives (0x80000000 … 0xFFFFFFFF) above the positives when read",
                           "as unsigned numbers; flipping bit 31 swaps the two halves, after which four 8-bit LSD passes",
                           "sort the keys exactly."]))


def fig_histogram(name):
    s = Svg(name, W, 100, "Image histogram: clamp each pixel to a bin, count per block in shared memory, merge")
    data = [0, 2, 3, 5, 1, 2, -1, 2, 3, 0, 9, 2]
    nb = 4
    b = [int(min(max(v, 0), nb - 1)) for v in data]
    x0, y0 = 60, 40
    row_label(s, x0 - 10, y0 + 14, "pixel", role="a", bold=True)
    row_label(s, x0 - 10, y0 + 56, "bin", role="d", bold=True)
    for i, (v, bi) in enumerate(zip(data, b)):
        s.box(x0 + i * 40, y0, 36, 28, str(v).replace("-", "−"), role="a", size="small", rx=2)
        s.box(x0 + i * 40, y0 + 42, 36, 28, str(bi), role="d", size="small", rx=2,
              fill="f-hl" if bi != v else None)
    hb = y0 + 110
    kit.bars(s, x0 + 100, hb, 70, [b.count(k) for k in range(nb)], 5, bw=40, gap=20, role="c",
             labels=[f"bin {k}" for k in range(nb)])
    return finish(s, note(s, 40, hb + 124, ["n_b = 4 here: values below 0 clamp to bin 0 and values above n_b − 1 to the last bin (red).",
                                            "Each block counts into n_b shared-memory counters with atomics, then adds them to",
                                            "global memory once; the counts are returned as floats and compared exactly."]))


# ----- attention -----------------------------------------------------------------------------------------------

def fig_scaled_dot_attention(name):
    return kit.attention(name, "Scaled dot-product attention for every (batch, head): full softmax, no mask", 8, 12,
                         lambda i, j: True, weight=lambda i, j: "f-c2" if i in (4, 5) else _tile(i, j),
                         legend=[("f-a2", "a", "key tiles 0 and 2"), ("f-a", "a", "key tile 1"),
                                 ("f-c2", "c", "one query block")],
                         pipeline_steps=["S = Q Kᵀ / √E (tile)", "online softmax (m, ℓ)", "O += P V (tile)"],
                         note_lines=["Grid = (query blocks, B·H): each (b, h) pair is an independent flash-attention problem",
                                     "on (S, E) matrices; E up to 256 is held in registers across the query block."])


def _tile(i, j):
    return "f-a2" if (j // 4) % 2 == 0 else "f-a"


# ----- low-precision formats -----------------------------------------------------------------------------------

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def _e4m3_values():
    vals = [f / 8 * 2 ** -6 for f in range(8)]
    for e in range(1, 16):
        for f in range(8):
            if e == 15 and f == 7:
                continue
            vals.append((1 + f / 8) * 2 ** (e - 7))
    return vals


E4M3 = _e4m3_values()


def _round_to(v, grid):
    m = min(grid, key=lambda g: (abs(abs(v) - g), g))
    return math.copysign(m, v) if m else 0.0


def _block_row(s, x0, y, vals, role, cw=56, labels=None, fills=None):  # noqa: E302
    cells(s, x0, y, labels or [fmt(v, 3) for v in vals], cw=cw - 6, gap=6, role=role, fills=fills)


def _mx_quant(name, title, elem, emax, grid, note_lines):
    s = Svg(name, W, 100, title)
    a = [1.8, -5.4, 10.2, 0.6, -2.4, 7.2, 0.1, -8.8]
    amax = max(abs(v) for v in a)
    E = math.floor(math.log2(amax)) - emax
    scale = 2.0 ** E
    codes = [_round_to(v / scale, grid) for v in a]
    x0, y0 = 150, 40
    row_label(s, x0 - 12, y0 + 14, "a (one block)", role="a", bold=True)
    _block_row(s, x0, y0, a, "a", fills=["f-a2" if abs(v) == amax else None for v in a])
    s.text(x0 + 8 * 56 + 6, y0 + 14, f"α = {fmt(amax)}", anchor="start", size="small", role="a")
    y1 = y0 + 56
    sc_txt = fmt(scale) if scale >= 1 else f"1/{int(round(1 / scale))}"
    s.box(x0 + 40, y1, 380, 30, f"E = ⌊log₂ α⌋ − {emax} = {fmt(E)}   →   scale 2^{{{fmt(E)}}} = {sc_txt},  u = {E + 127}",
          role="d", size="small")
    y2 = y1 + 56
    row_label(s, x0 - 12, y2 + 14, f"a / 2^E → {elem}", role="c", bold=True)
    _block_row(s, x0, y2, codes, "c")
    s.arrow(x0 + 220, y1 + 32, x0 + 220, y2 - 2, role="d", sw=1.2)
    return finish(s, note(s, 40, y2 + 58, note_lines))


def fig_mxfp4_quantize(name):
    return _mx_quant(name, "MXFP4 quantisation: one power-of-two scale per 32 values, E2M1 elements", "E2M1", 2, E2M1,
                     ["8 of the 32 elements of a block are shown. E2M1's largest value is 6 = 1.5·2², so e_{max} = 2;",
                      "FLOOR rounding of the scale exponent then maps the block into ±6 and each element is rounded",
                      "to the nearest of ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. Two codes per byte; the scale byte u = E + 127."])


def fig_mxfp8_quantize(name):
    return _mx_quant(name, "MXFP8 quantisation: one power-of-two scale per 32 values, E4M3 elements", "E4M3", 8, E4M3,
                     ["8 of the 32 elements of a block are shown. E4M3's largest value is 448 = 1.75·2⁸, so e_{max} = 8.",
                      "The scaled values have 3 mantissa bits, so they round finely (e.g. 10.2·2⁵ = 326.4 → 320).",
                      "One E4M3 byte per element plus one E8M0 byte (u = E + 127) per block of 32."])


def _mx_dequant(name, title, elem, codes, u, note_lines, packed):
    s = Svg(name, W, 100, title)
    x0, y0, cw = 130, 40, 52
    scale = 2.0 ** (u - 127)
    tx = x0 + 8 * cw + 8
    if packed:
        row_label(s, x0 - 12, y0 + 14, "bytes q", role="b", bold=True)
        for i in range(4):
            lo, hi = codes[2 * i], codes[2 * i + 1]
            enc = lambda v: (8 if v < 0 else 0) | E2M1.index(abs(v))
            s.box(x0 + i * 2 * cw, y0, 2 * cw - 6, 28, f"0x{enc(hi):X}{enc(lo):X}", role="b", size="small", mono=True,
                  rx=2)
        s.text(tx, y0 + 14, "byte = hi | lo nibble", anchor="start", size="small", role="b")
        y1 = y0 + 50
    else:
        y1 = y0
    row_label(s, x0 - 12, y1 + 14, f"{elem}(code)", role="c", bold=True)
    _block_row(s, x0, y1, codes, "c", cw=cw)
    s.text(tx, y1 + 44, f"× 2^{{{u} − 127}} = {fmt(scale)}", anchor="start", size="small", role="d")
    y2 = y1 + 60
    row_label(s, x0 - 12, y2 + 14, "out (FP32)", role="a", bold=True)
    _block_row(s, x0, y2, [c * scale for c in codes], "a", cw=cw)
    return finish(s, note(s, 40, y2 + 58, note_lines))


def fig_mxfp4_dequantize(name):
    return _mx_dequant(name, "MXFP4 dequantisation: decode two E2M1 codes per byte, multiply by the block's power of two",
                       "e2m1", [1, -3, 6, 0.5, -1, 4, 0, -4], 128,
                       ["Element j = 2i sits in the low nibble of byte i, j = 2i + 1 in the high nibble. A tiny lookup of",
                        "the 16 codes and an exponent add replace any float arithmetic; one thread writes 8 floats",
                        "(two float4 stores) per 4-byte load."], True)


def fig_mxfp8_dequantize(name):
    return _mx_dequant(name, "MXFP8 dequantisation: each E4M3 byte times its block's power-of-two scale", "e4m3",
                       [0.875, -2.75, 5, 0.25, -1.25, 3.75, 0.125, -4.5], 128,
                       ["E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa bits, max 448, no infinities (0x7F/0xFF are NaN).",
                        "The scale byte u = 128 means 2¹: dequantisation is exact, a table lookup and a multiply."], False)


def _block_gemm(name, title, block, scale_fmt, elem, extra):
    s = Svg(name, W, 100, title)
    x0, y0 = 60, 50
    nblk, bw = 4, 110
    roles = ["a", "b"]
    for r, (lab, role) in enumerate((("row i of A", "a"), ("row j of B", "b"))):
        y = y0 + r * 70
        row_label(s, x0 + 50, y + 14, lab, role=role, bold=True)
        for bl in range(nblk):
            bx = x0 + 60 + bl * (bw + 8)
            s.rect(bx, y, bw, 28, fill=f"f-{role}2" if bl == 1 else f"f-{role}", stroke=f"s-{role}", sw=1, rx=2)
            s.text(bx + bw / 2, y + 14, f"{block} × {elem}", size="tiny")
            s.box(bx + bw / 2 - 26, y + 34, 52, 22, f"σ{'AB'[r]}{bl}", role="d", size="tiny", rx=2)
    yb = y0 + 160
    s.box(x0 + 60, yb, 470, 34, "c_{ij} = Σ_{β} σ^{A}_{iβ} σ^{B}_{jβ} · (Σ_{ℓ∈β} x^{A}_{iℓ} x^{B}_{jℓ})", role="c", size="small")
    s.arrow(x0 + 60 + bw + 8 + bw / 2, y0 + 128, x0 + 250, yb - 2, role="c", sw=1.1)
    return finish(s, note(s, 40, yb + 64, [f"K is cut into blocks of {block}; every block has one {scale_fmt} scale per row (σ). The inner",
                                            "sum over a block runs on low-precision tensor-core MMAs (block-scaled MMA instructions",
                                            *extra]))


def fig_mxfp4_gemm(name):
    return _block_gemm(name, "MXFP4 GEMM: block-scaled dot products, C = Â B̂ᵀ in FP32", 32, "E8M0", "E2M1",
                       ["apply both scales in hardware); scales arrive in the swizzled 128 × 4 layout, so a warp's",
                        "16-byte load returns exactly the scales of the rows it computes."])


def fig_mxfp8_gemm(name):
    return _block_gemm(name, "MXFP8 GEMM: E4M3 blocks of 32 with power-of-two scales, C = Â B̂ᵀ", 32, "E8M0", "E4M3",
                       ["apply both scales in hardware); B is stored N × K, so the product is an \"NT\" GEMM and",
                        "both operands stream along K. Scales use the swizzled 128 × 4 layout."])


def fig_nvfp4_gemm(name):
    return _block_gemm(name, "NVFP4 GEMM: E2M1 blocks of 16 with E4M3 scales and one global factor per operand", 16,
                       "E4M3", "E2M1",
                       ["apply both scales); the global factors enter once at the end: c = acc / (g_A · g_B),",
                        "then the FP16 output is written. Smaller blocks (16) track local magnitudes better than MX."])


def _nv_block(s, x0, y0, g):
    a = [0.9, -2.7, 5.1, 0.3, -1.2, 3.6, 0.05, -4.4]
    amax = max(abs(v) for v in a)
    sc = _round_to(g * amax / 6, E4M3)
    codes = [_round_to(v * g / sc, E2M1) for v in a]
    row_label(s, x0 - 12, y0 + 14, "a (block of 16)", role="a", bold=True)
    _block_row(s, x0, y0, a, "a", fills=["f-a2" if abs(v) == amax else None for v in a])
    y1 = y0 + 56
    s.box(x0 + 20, y1, 420, 30, f"s = e4m3(g · α / 6) = e4m3({fmt(g * amax / 6, 3)}) = {fmt(sc, 4)}   (g = {fmt(g)})",
          role="d", size="small")
    y2 = y1 + 56
    row_label(s, x0 - 12, y2 + 14, "code = e2m1(a·g/s)", role="c", bold=True)
    _block_row(s, x0, y2, codes, "c")
    s.arrow(x0 + 220, y1 + 32, x0 + 220, y2 - 2, role="d", sw=1.2)
    return y2, codes, sc


def fig_nvfp4_quantize(name):
    s = Svg(name, W, 100, "NVFP4 quantisation: an E4M3 scale per 16 values on top of a global factor g")
    y2, codes, sc = _nv_block(s, 170, 40, 2.0)
    return finish(s, note(s, 40, y2 + 58, ["8 of the 16 values are shown. The element encode divides by the *rounded* scale e4m3(s),",
                                           "so decode(encode(x)) = code · e4m3(s) / g is consistent. Scales are written in the",
                                           "swizzled 128 × 4 layout: rows r, r + 32, r + 64, r + 96 share one 16-byte group."]))


def fig_nvfp4_dequantize(name):
    s = Svg(name, W, 100, "NVFP4 dequantisation: â = e2m1(code) · e4m3(s) / g")
    codes = [0.5, -1.5, 3, 0, -0.5, 2, 0, -2]
    sc, g = 3.25, 2.0
    x0, y0 = 170, 40
    row_label(s, x0 - 12, y0 + 14, "e2m1(code)", role="c", bold=True)
    _block_row(s, x0, y0, codes, "c")
    s.box(x0 + 20, y0 + 50, 420, 30, f"block scale e4m3(s) = {fmt(sc)},  global g = {fmt(g)}  →  × {fmt(sc / g, 4)}",
          role="d", size="small")
    y2 = y0 + 106
    row_label(s, x0 - 12, y2 + 14, "â (FP32)", role="a", bold=True)
    _block_row(s, x0, y2, [c * sc / g for c in codes], "a")
    s.arrow(x0 + 220, y0 + 82, x0 + 220, y2 - 2, role="d", sw=1.2)
    return finish(s, note(s, 40, y2 + 58, ["Two codes per byte (low nibble first); one E4M3 scale per 16 elements, read from the",
                                           "swizzled layout; g is one FP32 number per tensor. Each thread expands 16 elements",
                                           "(8 bytes plus one scale byte) into four float4 stores."]))


def fig_nvfp4_gemv(name):
    s = Svg(name, W, 100, "NVFP4 GEMV: a warp streams one quantised row and the quantised vector, block by block")
    x0, y0 = 60, 50
    for r, (lab, role) in enumerate((("x (NVFP4)", "b"), ("row i of A", "a"))):
        y = y0 + r * 60
        row_label(s, x0 + 60, y + 14, lab, role=role, bold=True)
        for bl in range(4):
            bx = x0 + 72 + bl * 120
            s.rect(bx, y, 112, 28, fill=f"f-{role}2" if bl == 2 else f"f-{role}", stroke=f"s-{role}", sw=1, rx=2)
            s.text(bx + 56, y + 14, "16 codes + s", size="tiny")
    yb = y0 + 140
    s.box(x0 + 72, yb, 470, 34, "y_i = (1 / (g_A g_x)) · Σ_{β} s^{A}_{iβ} s^{x}_{β} Σ_{ℓ∈β} e2m1(a) e2m1(x)", role="c",
          size="small")
    s.arrow(x0 + 72 + 2 * 120 + 56, y0 + 90, x0 + 300, yb - 2, role="c", sw=1.1)
    return finish(s, note(s, 40, yb + 64, ["4 bits per weight make the GEMV even more bandwidth-bound than FP16: one warp per row,",
                                           "each lane decodes whole 16-element blocks with a 16-entry table, and FP32 partial sums",
                                           "are combined with shuffles before the single FP16 store."]))
