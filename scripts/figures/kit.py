"""Reusable figure templates for the problem pages (figures/leetgpu.py, figures/tensara.py).

Every problem gets one figure that shows *what* is computed (with small, real
numbers where possible) and *how* the work is split across threads. The
templates below cover the recurring shapes: elementwise maps, activation
curves, reductions, scans, reductions along one axis, row/column
normalisation, tiled matrix products, sliding windows, attention masks and
graphs. They only use the primitives of svg.py, so the colours follow the
site theme and scripts/check_figures.py can verify every label.
"""
from __future__ import annotations

import math
import re

from . import svg as _svg

# x_s after any letter (also Greek, primes or accents) becomes a subscript; svg.rich only does this after ASCII.
_LOOSE_SUB = re.compile(r"(?<=[^\x00-\x7F])_([^\s_{}])(?=[\s,.;:)\]·/+−=²]|$)")


class Svg(_svg.Svg):
    """svg.Svg whose labels also accept single-character subscripts after non-ASCII letters (κ_j, δ_t)."""

    def text(self, x, y, s, *args, **kwargs):
        if not kwargs.get("mono") and not kwargs.get("plain"):
            s = _LOOSE_SUB.sub(r"_{\1}", str(s))
        super().text(x, y, s, *args, **kwargs)

W = 720  # every problem figure is 720 px wide


def fmt(v, nd=2) -> str:
    """Compact number: integers without a decimal point, otherwise up to nd decimals."""
    if isinstance(v, str):
        return v
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v))).replace("-", "−")
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s.replace("-", "−")


# ----- small building blocks ---------------------------------------------------------------------------------

def cells(s: Svg, x, y, labels, cw=40, ch=28, role="a", fills=None, size="small", mono=False, gap=0,
          text_roles=None):
    """A row of labelled cells; fills[i] overrides the fill class of cell i (None = default)."""
    for i, lab in enumerate(labels):
        fill = fills[i] if fills and fills[i] else None
        tr = text_roles[i] if text_roles else None
        cell_role = role if isinstance(role, str) else role[i]
        s.box(x + i * (cw + gap), y, cw, ch, lab, role=cell_role, fill=fill, size=size, mono=mono, rx=2,
              text_role=tr)


def row_label(s: Svg, x, y, text, role="ink", size="small", bold=False):
    """Label written to the left of a row (right-aligned at x)."""
    s.text(x, y, text, anchor="end", size=size, role=role, bold=bold)


def note(s: Svg, x, y, lines, role="muted", size="small", anchor="start", lh=17):
    """Lines of text; returns the y just below the last line."""
    for i, ln in enumerate(lines):
        s.text(x, y + i * lh, ln, role=role, size=size, anchor=anchor)
    return y + (len(lines) - 1) * lh + 8 if lines else y - 14


def finish(s: Svg, bottom) -> Svg:
    """Shrink or grow the figure so it ends a little below `bottom`."""
    s.height = math.ceil(bottom + 14)
    return s


def matrix(s: Svg, x, y, rows, cols, cw, ch=None, fill_fn=None, label=None, label_pos="top", role="ink",
           outline=None):
    """A grid of cells with an optional name above (label_pos='top') or to the left ('left')."""
    ch = ch or cw
    s.grid(x, y, rows, cols, cw, ch, fill_fn=fill_fn, outline=outline or f"s-{role}")
    if label:
        if label_pos == "top":
            s.text(x + cols * cw / 2, y - 12, label, role=role, size="small", bold=True)
        elif label_pos == "bottom":
            s.text(x + cols * cw / 2, y + rows * ch + 13, label, role=role, size="small", bold=True)
        else:
            s.text(x - 8, y + rows * ch / 2, label, role=role, size="small", bold=True, anchor="end")


def pipeline(s: Svg, x, y, steps, w=120, h=34, gap=26, role="d", size="small"):
    """Boxes left to right joined by arrows; steps = [label | (label, role)]."""
    for i, st in enumerate(steps):
        lab, r = (st, role) if isinstance(st, str) else st
        bx = x + i * (w + gap)
        s.box(bx, y, w, h, lab, role=r, size=size)
        if i:
            s.arrow(bx - gap + 2, y + h / 2, bx - 2, y + h / 2, role="muted", sw=1.1)
    return x + len(steps) * (w + gap) - gap


# ----- 1. elementwise maps -----------------------------------------------------------------------------------

def elementwise(name, title, inputs, output, op, height=250, note_lines=(), n_label="thread i", x0=150, cw=46,
                warp_note=True, op_role="ink"):
    """inputs: [(name, labels, role)], output: (name, labels, role); op: text drawn between the input rows."""
    s = Svg(name, W, height, title)
    n = len(output[1])
    for i in range(n):
        s.text(x0 + i * cw + (cw - 6) / 2, 22, str(i), size="small", role="muted")
    row_label(s, x0 - 14, 22, n_label, role="muted")
    y = 38
    for k, (nm, labs, role) in enumerate(inputs):
        row_label(s, x0 - 14, y + 14, nm, role=role, bold=True)
        cells(s, x0, y, labs, cw=cw - 6, gap=6, role=role)
        y += 28
        if k < len(inputs) - 1:
            for i in range(n):
                s.text(x0 + i * cw + (cw - 6) / 2, y + 11, op, role=op_role, size="small")
            y += 22
    for i in range(n):
        s.arrow(x0 + i * cw + (cw - 6) / 2, y + 2, x0 + i * cw + (cw - 6) / 2, y + 28, role="muted", sw=1)
    if len(inputs) == 1:
        s.text(x0 + n * cw + 4, y + 15, op, role=op_role, size="small", anchor="start")
    y += 30
    nm, labs, role = output
    row_label(s, x0 - 14, y + 14, nm, role=role, bold=True)
    cells(s, x0, y, labs, cw=cw - 6, gap=6, role=role)
    y += 28
    if warp_note:
        s.brace_h(x0, x0 + n * cw - 6, y + 12, "", role="muted")
        s.text(x0 + (n * cw - 6) / 2, y + 30, "consecutive threads touch consecutive addresses → coalesced",
               size="small", role="muted")
        y += 34
    return finish(s, note(s, x0 - 130, y + 22, note_lines))


# ----- 2. function plots -------------------------------------------------------------------------------------

class Plot:
    """Axes mapping data coordinates to a box; draws curves, ticks and marked points."""

    def __init__(self, s: Svg, x, y, w, h, xr, yr):
        self.s, self.x, self.y, self.w, self.h, self.xr, self.yr = s, x, y, w, h, xr, yr

    def px(self, v):
        return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def py(self, v):
        return self.y + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def axes(self, xticks=(), yticks=(), xlabel="x", ylabel="y"):
        s = self.s
        s.rect(self.x, self.y, self.w, self.h, fill="fig-panel", stroke="s-line", sw=0.6)
        if self.yr[0] < 0 < self.yr[1]:
            s.line(self.x, self.py(0), self.x + self.w, self.py(0), stroke="s-muted", sw=1)
        if self.xr[0] < 0 < self.xr[1]:
            s.line(self.px(0), self.y, self.px(0), self.y + self.h, stroke="s-muted", sw=1)
        for t in xticks:
            s.line(self.px(t), self.y + self.h, self.px(t), self.y + self.h + 4, stroke="s-muted", sw=1)
            s.text(self.px(t), self.y + self.h + 14, fmt(t), size="tiny", role="muted")
        for t in yticks:
            s.line(self.x - 4, self.py(t), self.x, self.py(t), stroke="s-muted", sw=1)
            s.text(self.x - 7, self.py(t), fmt(t), size="tiny", role="muted", anchor="end")
        s.text(self.x + self.w / 2, self.y + self.h + 30, xlabel, size="small", role="muted", italic=True)
        s.text(self.x - 42, self.y + self.h / 2, ylabel, size="small", role="muted", italic=True, rotate=-90)

    def curve(self, f, role="a", sw=2.2, dash=None, n=240):
        pts = []
        segs = []
        for k in range(n + 1):
            xv = self.xr[0] + (self.xr[1] - self.xr[0]) * k / n
            try:
                yv = f(xv)
            except (OverflowError, ValueError, ZeroDivisionError):
                yv = None
            if yv is None or math.isnan(yv) or not (self.yr[0] - 1e-9 <= yv <= self.yr[1] + 1e-9):
                if len(pts) > 1:
                    segs.append(pts)
                pts = []
                continue
            pts.append((self.px(xv), self.py(yv)))
        if len(pts) > 1:
            segs.append(pts)
        for p in segs:
            d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in p)
            self.s.path(d, stroke=f"s-{role}", sw=sw, dash=dash)

    def dot(self, xv, yv, role="hl", r=4):
        self.s.circle(self.px(xv), self.py(yv), r, fill=f"k-{role}")


def function_plot(name, title, curves, xr, yr, xticks, yticks, formula_lines, points=(), side_lines=(),
                  height=270, legend=None, xlabel="x", ylabel="f(x)"):
    """curves: [(f, role, dash)]; formula_lines: text on the right; points: [(x, y, role)]."""
    s = Svg(name, W, height, title)
    p = Plot(s, 70, 20, 300, height - 70, xr, yr)
    p.axes(xticks, yticks, xlabel, ylabel)
    for f, role, dash in curves:
        p.curve(f, role=role, dash=dash)
    for xv, yv, role in points:
        p.dot(xv, yv, role=role)
    tx = 400
    for i, ln in enumerate(formula_lines):
        s.text(tx, 36 + i * 22, ln, anchor="start", size="small", bold=(i == 0), role="ink")
    y = 36 + len(formula_lines) * 22 + 8
    if legend:
        s.legend(tx, y + 6, legend)
        y += 18 * len(legend) + 10
    note(s, tx, y + 8, side_lines)
    return finish(s, max(height - 14, y))


# ----- 3. reductions -----------------------------------------------------------------------------------------

def reduce_tree(name, title, values, combine, op_label, result_label=None, map_row=None, final_lines=(),
                height=None, value_fmt=fmt, role="a", tail=None, x0=150, cw=62):
    """A tree reduction of 8 values; map_row = (row name, labels) drawn above the tree (the per-thread transform).

    combine(a, b) gives the parent value; tail = (label, text) for a final scalar step (e.g. '÷ N').
    """
    n = len(values)
    levels = int(math.log2(n))
    top = 24
    rows_before = 2 if map_row else 1
    height = height or top + rows_before * 44 + levels * 50 + 60 + 18 * len(final_lines) + (40 if tail else 0)
    s = Svg(name, W, height, title)
    y = top
    if map_row:
        nm, labs = map_row
        row_label(s, x0 - 14, y + 14, nm, role="a", bold=True)
        cells(s, x0, y, labs, cw=cw - 10, gap=10, role="a")
        for i in range(n):
            s.arrow(x0 + i * cw + (cw - 10) / 2, y + 28, x0 + i * cw + (cw - 10) / 2, y + 44, role="muted", sw=1)
        y += 44
    nm = op_label
    row_label(s, x0 - 14, y + 14, nm, role="b", bold=True)
    cells(s, x0, y, [value_fmt(v) for v in values], cw=cw - 10, gap=10, role="b")
    level = list(values)
    pos = [x0 + i * cw + (cw - 10) / 2 for i in range(n)]
    for lv in range(levels):
        ny = y + 50
        new, npos = [], []
        for i in range(0, len(level), 2):
            v = combine(level[i], level[i + 1])
            cx = (pos[i] + pos[i + 1]) / 2
            s.line(pos[i], y + 28, cx, ny, stroke="s-muted", sw=1)
            s.line(pos[i + 1], y + 28, cx, ny, stroke="s-muted", sw=1)
            new.append(v)
            npos.append(cx)
        for v, cx in zip(new, npos):
            last = len(new) == 1 and not tail
            s.box(cx - 26, ny, 52, 28, value_fmt(v), role="c" if last else "d", size="small", bold=last,
                  fill="f-c2" if last else None)
        row_label(s, x0 - 14, ny + 14, ["warp shuffle", "shared memory", "across blocks"][min(lv, 2)]
                  if levels == 3 else f"level {lv + 1}", role="muted")
        level, pos, y = new, npos, ny
    if tail:
        lab, txt = tail
        ny = y + 50
        s.arrow(pos[0], y + 28, pos[0], ny, role="muted", sw=1)
        s.text(pos[0] + 8, y + 39, lab, anchor="start", size="small", role="muted")
        s.box(pos[0] - 45, ny, 90, 28, txt, role="c", fill="f-c2", size="small", bold=True)
        y = ny
    if result_label:
        s.text(pos[0] + 60, y + 14, result_label, anchor="start", size="small", role="c")
    return finish(s, note(s, x0 - 130, y + 52, final_lines) if final_lines else y + 30)


# ----- 4. scans ----------------------------------------------------------------------------------------------

def scan(name, title, xs, ys, op_text, x_name="x", y_name="y", flags=None, highlight=None, chunk=None,
         note_lines=(), height=None, reverse=False, cw=52, x0=150, ys_role="c", note_lh=17):
    """Input row, (optional flag row), output row; highlight = output index whose inputs are bracketed."""
    n = len(xs)
    height = height or 190 + (40 if flags else 0) + (60 if chunk else 0) + 17 * len(note_lines)
    s = Svg(name, W, height, title)
    for i in range(n):
        s.text(x0 + i * cw + (cw - 8) / 2, 20, str(i), size="small", role="muted")
    row_label(s, x0 - 14, 20, "index", role="muted")
    y = 36
    if flags is not None:
        row_label(s, x0 - 14, y + 13, "flag", role="d", bold=True)
        cells(s, x0, y, [str(f) for f in flags], cw=cw - 8, gap=8, ch=26, role="d",
              fills=["f-d2" if f else None for f in flags])
        y += 40
    row_label(s, x0 - 14, y + 14, x_name, role="a", bold=True)
    fills = None
    if highlight is not None:
        lo, hi, _ = highlight
        fills = ["f-a2" if lo <= i <= hi else None for i in range(n)]
    cells(s, x0, y, [fmt(v) for v in xs], cw=cw - 8, gap=8, role="a", fills=fills)
    xy = y
    y += 28 + 44
    row_label(s, x0 - 14, y + 14, y_name, role=ys_role, bold=True)
    tgt = None
    if highlight is not None:
        tgt = highlight[2]
    cells(s, x0, y, [fmt(v) for v in ys], cw=cw - 8, gap=8, role=ys_role,
          fills=["f-c2" if i == tgt else None for i in range(n)])
    if highlight is not None:
        lo, hi, t = highlight
        cx = x0 + t * cw + (cw - 8) / 2
        for i in range(lo, hi + 1):
            s.line(x0 + i * cw + (cw - 8) / 2, xy + 28, cx, y, stroke="s-a", sw=0.9)
    s.text(x0 + n * cw + 4, xy + 50, op_text, anchor="start", size="small", role="ink")
    if reverse:
        s.arrow(x0 + n * cw - 20, xy + 36, x0 + 4, xy + 36, role="muted", sw=1)
    y += 28
    if chunk:
        size, labels = chunk
        for c in range(0, n, size):
            s.brace_h(x0 + c * cw, x0 + (c + size) * cw - 8, y + 12, "", role="d")
            s.text(x0 + c * cw + (size * cw - 8) / 2, y + 30, labels[c // size], size="small", role="d")
        y += 44
    return finish(s, note(s, x0 - 130, y + 22, note_lines, lh=note_lh) if note_lines else y)


# ----- 5. reduce along one axis ------------------------------------------------------------------------------

def reduce_axis(name, title, grid, op_name, out_row, note_lines=(), out_role="c", highlight_col=1,
                sub=None, cw=46, height=None):
    """grid: R x I values of one outer slice x[o, :, :]; the reduction runs down each column (axis j)."""
    R, I = len(grid), len(grid[0])
    height = height or 110 + R * 30 + 60 + 17 * len(note_lines)
    s = Svg(name, W, height, title)
    x0, y0 = 190, 44
    # shadow slices for the outer axis O
    for k in (2, 1):
        s.rect(x0 + 8 * k, y0 - 8 * k, I * cw, R * 30, fill="fig-panel", stroke="s-line", sw=0.8)
    s.text(x0 + I * cw + 26, y0 - 4, "o = 0 … O−1", anchor="start", size="small", role="muted")
    for j in range(R):
        fills = ["f-a2" if i == highlight_col else None for i in range(I)]
        cells(s, x0, y0 + j * 30, [fmt(v) for v in grid[j]], cw=cw, ch=30, role="a", fills=fills)
        row_label(s, x0 - 12, y0 + j * 30 + 15, f"j = {j}", role="muted")
    s.text(x0 + 16 + I * cw / 2, y0 - 32, "i = 0 … I−1 (contiguous)", size="small", role="muted")
    s.brace_v(x0 - 70, y0, y0 + R * 30, "", role="b")
    s.text(x0 - 80, y0 + R * 30 / 2, "R", anchor="end", role="b", bold=True)
    yo = y0 + R * 30 + 40
    cx = x0 + highlight_col * cw + cw / 2
    s.arrow(cx, y0 + R * 30 + 2, cx, yo - 2, role="b", sw=1.2)
    s.text(cx + 8, y0 + R * 30 + 20, op_name, anchor="start", size="small", role="b")
    row_label(s, x0 - 12, yo + 15, "out", role=out_role, bold=True)
    cells(s, x0, yo, out_row, cw=cw, ch=30, role=out_role,
          fills=["f-c2" if i == highlight_col else None for i in range(len(out_row))])
    return finish(s, note(s, 40, yo + 60, note_lines) if note_lines else yo + 30)


# ----- 6. matrix products ------------------------------------------------------------------------------------

def gemm(name, title, a_label="A", b_label="B", c_label="C", m=6, n=6, k=6, cw=20, a_mask=None, b_mask=None,
         c_mask=None, tile=(2, 2), kstep=None, epilogue=(), note_lines=(), batch=0, dims=("M", "N", "K"),
         height=None, a_role="a", b_role="b"):
    """C (m x n) = A (m x k) · B (k x n). Masks are functions (r, c) -> bool (True = structurally zero).

    tile = (tile row, tile col) of the highlighted output tile (2 x 2 cells); kstep = index of the k slice
    (2 wide) highlighted in A and B; epilogue = lines of text in a box to the right of C.
    """
    ax, by = 40, 30
    bx = ax + k * cw + 30
    ay = by + k * cw + 30
    height = height or ay + m * cw + 40 + 17 * len(note_lines)
    s = Svg(name, W, height, title)
    tr, tc = tile
    ks = kstep if kstep is not None else 1

    def fa(r, c):
        if a_mask and a_mask(r, c):
            return "fig-panel"
        if tr * 2 <= r < tr * 2 + 2:
            return f"f-{a_role}2" if ks * 2 <= c < ks * 2 + 2 else f"f-{a_role}"
        return None

    def fb(r, c):
        if b_mask and b_mask(r, c):
            return "fig-panel"
        if tc * 2 <= c < tc * 2 + 2:
            return f"f-{b_role}2" if ks * 2 <= r < ks * 2 + 2 else f"f-{b_role}"
        return None

    def fc(r, c):
        if c_mask and c_mask(r, c):
            return "fig-panel"
        if tr * 2 <= r < tr * 2 + 2 and tc * 2 <= c < tc * 2 + 2:
            return "f-c2"
        return None

    for bi in range(batch, 0, -1):  # stacked copies behind (batched GEMM)
        off = 6 * bi
        for (x, y, rr, cc) in ((ax, ay, m, k), (bx, by, k, n), (bx, ay, m, n)):
            s.rect(x + off, y - off, cc * cw, rr * cw, fill="fig-panel", stroke="s-line", sw=0.8)
    matrix(s, ax, ay, m, k, cw, fill_fn=fa, role=a_role)
    matrix(s, bx, by, k, n, cw, fill_fn=fb, role=b_role)
    matrix(s, bx, ay, m, n, cw, fill_fn=fc, role="c")
    s.text(ax + k * cw / 2, ay + m * cw + 14, f"{a_label}  ({dims[0]} × {dims[2]})", size="small", role=a_role,
           bold=True)
    s.text(bx - 12, by + k * cw / 2, f"{b_label}  ({dims[2]} × {dims[1]})", size="small", role=b_role, bold=True,
           anchor="end")
    s.text(bx + n * cw / 2, ay + m * cw + 14, f"{c_label}  ({dims[0]} × {dims[1]})", size="small", role="c",
           bold=True)
    ex = bx + n * cw + 34
    if epilogue:
        lines = list(epilogue)
        bh = 16 + 18 * len(lines)
        byy = ay + (m * cw - bh) / 2
        s.arrow(bx + n * cw + 4, ay + m * cw / 2, ex - 2, ay + m * cw / 2, role="d", sw=1.2)
        s.rect(ex, byy, W - ex - 16, bh, fill="f-d", stroke="s-d", sw=1.2, rx=4)
        for i, ln in enumerate(lines):
            s.text(ex + 10, byy + 17 + 18 * i, ln, anchor="start", size="small", bold=(i == 0))
    if not epilogue:
        note(s, ex, by + 20, note_lines)
        return finish(s, ay + m * cw + 20)
    return finish(s, note(s, ax, ay + m * cw + 40, note_lines) if note_lines else ay + m * cw + 20)


# ----- 7. sliding windows ------------------------------------------------------------------------------------

def window_1d(name, title, xs, ys, k, stride=1, pad=0, dilation=1, out_index=1, op_text="Σ w·x", note_lines=(),
              causal=False, height=None, pad_label="0", cw=42, x_name="x", y_name="y", weights=None):
    """1-D window: padded input row (pad cells shown), taps of output out_index bracketed and joined."""
    n = len(xs)
    total = n + 2 * pad if not causal else n + pad
    height = height or 196 + (40 if weights else 0) + 17 * len(note_lines)
    s = Svg(name, W, height, title)
    x0 = 130
    left = pad
    labs = [pad_label] * left + [fmt(v) for v in xs] + ([pad_label] * pad if not causal else [])
    start = out_index * stride
    taps = [start + t * dilation for t in range(k)]
    fills = ["fig-panel" if (i < left or i >= left + n) else ("f-a2" if i in taps else None) for i in range(total)]
    y = 34
    for i in range(total):
        idx = i - left
        s.text(x0 + i * cw + (cw - 6) / 2, 20, str(idx).replace("-", "−"), size="tiny", role="muted")
    row_label(s, x0 - 12, y + 14, x_name, role="a", bold=True)
    cells(s, x0, y, labs, cw=cw - 6, gap=6, role=["line" if (i < left or i >= left + n) else "a" for i in range(total)],
          fills=fills)
    y2 = y + 28
    if weights:
        yw = y2 + 12
        row_label(s, x0 - 12, yw + 13, "w", role="b", bold=True)
        for t, tpos in enumerate(taps):
            s.box(x0 + tpos * cw, yw, cw - 6, 26, weights[t], role="b", size="small", rx=2)
        y2 = yw + 26
    yo = y2 + 58
    no = len(ys)
    ox0 = x0 + left * cw if not causal else x0 + left * cw
    ocx = ox0 + out_index * cw + (cw - 6) / 2
    for tpos in taps:
        s.line(x0 + tpos * cw + (cw - 6) / 2, y2, ocx, yo, stroke="s-a", sw=0.9)
    row_label(s, x0 - 12, yo + 14, y_name, role="c", bold=True)
    cells(s, ox0, yo, [fmt(v) for v in ys], cw=cw - 6, gap=6, role="c",
          fills=["f-c2" if i == out_index else None for i in range(no)])
    s.text(ocx + 16, (y2 + yo) / 2 + 4, op_text, anchor="start", size="small", role="ink", plate=True)
    return finish(s, note(s, 30, yo + 56, note_lines) if note_lines else yo + 30)


def window_2d(name, title, rows, cols, k, out_rc=(1, 1), stride=1, pad=0, note_lines=(), op_text="Σ w·x",
              out_rows=None, out_cols=None, cw=24, dilation=1, pad_role="line", pad_note=None, height=None,
              in_label="input", out_label="output", taps=None):
    """2-D window over a padded grid; the output grid is drawn to the right."""
    tot_r, tot_c = rows + 2 * pad, cols + 2 * pad
    out_rows = out_rows or rows
    out_cols = out_cols or cols
    height = height or max(tot_r, out_rows) * cw + 80 + 17 * len(note_lines)
    s = Svg(name, W, height, title)
    x0, y0 = 40, 34
    orow, ocol = out_rc
    r0, c0 = orow * stride, ocol * stride
    win = taps or {(r0 + a * dilation, c0 + b * dilation) for a in range(k) for b in range(k)}

    def f(r, c):
        inside = pad <= r < pad + rows and pad <= c < pad + cols
        if (r, c) in win:
            return "f-a2" if inside else "f-hl"
        return None if inside else "fig-panel"

    matrix(s, x0, y0, tot_r, tot_c, cw, fill_fn=f, role="a")
    s.text(x0, y0 - 14, in_label, size="small", role="a", bold=True, anchor="start")
    wr = min(r for r, _ in win), max(r for r, _ in win)
    wc = min(c for _, c in win), max(c for _, c in win)
    s.rect(x0 + wc[0] * cw, y0 + wr[0] * cw, (wc[1] - wc[0] + 1) * cw, (wr[1] - wr[0] + 1) * cw,
           fill="fig-none", stroke="s-hl", sw=2)
    ox = x0 + tot_c * cw + 150
    oy = y0 + (tot_r - out_rows) * cw / 2
    matrix(s, ox, oy, out_rows, out_cols, cw, fill_fn=lambda r, c: "f-c2" if (r, c) == out_rc else None,
           role="c")
    s.text(ox + out_cols * cw / 2, oy - 14, out_label, size="small", role="c", bold=True)
    s.arrow(x0 + (wc[1] + 1) * cw + 2, y0 + (wr[0] + wr[1] + 1) * cw / 2,
            ox + ocol * cw + cw / 2 - 2, oy + orow * cw + cw / 2, role="hl", sw=1.3)
    s.text((x0 + tot_c * cw + ox) / 2, y0 + tot_r * cw + 20, op_text, size="small", role="ink")
    bottom = y0 + max(tot_r * cw, oy - y0 + out_rows * cw) + 26
    if pad_note:
        s.legend(x0, bottom + 20, pad_note)
        bottom += 18 * len(pad_note) + 8
    return finish(s, note(s, 40, bottom + 20, note_lines) if note_lines else bottom)


# ----- 8. attention ------------------------------------------------------------------------------------------

def attention(name, title, n_q, n_k, visible, weight=None, note_lines=(), legend=None, pipeline_steps=None,
              cw=22, row_lab="query i", col_lab="key j", height=None, hl_row=None, x0=110):
    """Score matrix: visible(i, j) -> bool; weight(i, j) -> 'f-a' / 'f-a2' / ... overrides the colour."""
    height = height or 70 + n_q * cw + 90 + 17 * len(note_lines)
    s = Svg(name, W, height, title)
    y0 = 50

    def f(i, j):
        if not visible(i, j):
            return "fig-panel"
        if weight:
            return weight(i, j)
        return "f-c2" if i == hl_row else "f-a2"

    matrix(s, x0, y0, n_q, n_k, cw, fill_fn=f, role="ink")
    s.text(x0 + n_k * cw / 2, y0 - 26, col_lab + " →", size="small", role="muted")
    s.text(x0 - 8, y0 - 26, row_lab + " ↓", size="small", role="muted", anchor="end")
    for j in range(n_k):
        if j % 2 == 0:
            s.text(x0 + j * cw + cw / 2, y0 - 9, str(j), size="tiny", role="muted")
    for i in range(n_q):
        if i % 2 == 0:
            s.text(x0 - 8, y0 + i * cw + cw / 2, str(i), size="tiny", role="muted", anchor="end")
    rx = x0 + n_k * cw + 36
    ly = y0 + 8
    if legend:
        s.legend(rx, ly, legend)
        ly += 18 * len(legend) + 20
    if pipeline_steps:
        for i, st in enumerate(pipeline_steps):
            lab, role = (st, "d") if isinstance(st, str) else st
            bw = W - rx - 20
            s.box(rx, ly + i * 50, bw, 30, lab, role=role, size="small")
            if i:
                s.arrow(rx + bw / 2, ly + i * 50 - 18, rx + bw / 2, ly + i * 50 - 2, role="muted", sw=1)
        ly += 50 * len(pipeline_steps)
    bottom = max(y0 + n_q * cw, ly - 20)
    return finish(s, note(s, 40, bottom + 30, note_lines) if note_lines else bottom)


# ----- 9. graphs ---------------------------------------------------------------------------------------------

def graph(s: Svg, nodes, edges, directed=False, hl_edges=(), node_role=None, labels=None, r=17, weight_pos=0.5,
          hl_role="hl"):
    """nodes: {id: (x, y)}; edges: [(u, v, weight)]; hl_edges: set of (u, v)."""
    node_role = node_role or {}
    for u, v, wgt in edges:
        (x1, y1), (x2, y2) = nodes[u], nodes[v]
        d = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        hl = (u, v) in hl_edges or (not directed and (v, u) in hl_edges)
        role = hl_role if hl else "muted"
        sx, sy, ex, ey = x1 + ux * r, y1 + uy * r, x2 - ux * (r + 2), y2 - uy * (r + 2)
        if directed:
            s.arrow(sx, sy, ex, ey, role=role, sw=2 if hl else 1.1)
        else:
            s.line(sx, sy, ex, ey, stroke=f"s-{role}", sw=2.4 if hl else 1.1)
        if wgt != "":
            mx, my = x1 + (x2 - x1) * weight_pos, y1 + (y2 - y1) * weight_pos
            s.text(mx, my, str(wgt), size="small", role=role, bold=hl, plate=True)
    for nid, (x, y) in nodes.items():
        role = node_role.get(nid, "a")
        s.box(x - r, y - r, 2 * r, 2 * r, labels.get(nid, str(nid)) if labels else str(nid), role=role, rx=r,
              size="small", bold=True, sw=1.4)


# ----- 10. normalisation -------------------------------------------------------------------------------------

def normalize(name, title, rows, cols, group, hl, stat_lines, out_lines, note_lines=(), row_names=None,
              col_names=None, cw=34, ch=26, x0=150):
    """A rows x cols grid whose cells belong to statistic groups: group(r, c) -> id; hl = id highlighted.

    The highlighted group feeds a statistics box (stat_lines) and then the output formula (out_lines).
    """
    s = Svg(name, W, 100, title)
    y0 = 40
    matrix(s, x0, y0, rows, cols, cw, ch, fill_fn=lambda r, c: "f-a2" if group(r, c) == hl else None, role="a")
    if col_names:
        s.text(x0 + cols * cw / 2, y0 - 14, col_names, size="small", role="muted")
    if row_names:
        s.text(x0 - 10, y0 + rows * ch / 2, row_names, size="small", role="muted", anchor="end")
    bx = x0 + cols * cw + 50
    bw = W - bx - 20
    sh = 14 + 18 * len(stat_lines)
    s.arrow(x0 + cols * cw + 4, y0 + 20, bx - 2, y0 + 20, role="a", sw=1.2)
    s.rect(bx, y0, bw, sh, fill="f-d", stroke="s-d", sw=1.2, rx=4)
    for i, ln in enumerate(stat_lines):
        s.text(bx + 10, y0 + 16 + 18 * i, ln, anchor="start", size="small", bold=(i == 0))
    oy = y0 + sh + 34
    oh = 14 + 18 * len(out_lines)
    s.arrow(bx + bw / 2, y0 + sh + 2, bx + bw / 2, oy - 2, role="d", sw=1.2)
    s.rect(bx, oy, bw, oh, fill="f-c", stroke="s-c", sw=1.2, rx=4)
    for i, ln in enumerate(out_lines):
        s.text(bx + 10, oy + 16 + 18 * i, ln, anchor="start", size="small", bold=(i == 0))
    bottom = max(y0 + rows * ch, oy + oh)
    return finish(s, note(s, 40, bottom + 30, note_lines) if note_lines else bottom)


# ----- 11. bars ----------------------------------------------------------------------------------------------

def bars(s: Svg, x, y, h, values, vmax, bw=26, gap=8, fills=None, labels=None, value_labels=True, role="a",
         vfmt=None):
    """Vertical bars standing on the baseline y + h; labels under the bars, values over them."""
    vfmt = vfmt or fmt
    prev = None
    for i, v in enumerate(values):
        bh = max(1.0, h * v / vmax)
        bx = x + i * (bw + gap)
        f = fills[i] if fills else f"f-{role}2"
        s.rect(bx, y + h - bh, bw, bh, fill=f, stroke=f"s-{role}", sw=1)
        if value_labels:
            ly = y + h - bh - 9
            if prev is not None and abs(ly - prev) < 7 and ly >= prev:
                ly = prev  # snap nearly level labels onto one line (they sit above their bars)
            s.text(bx + bw / 2, ly, vfmt(v), size="tiny", role="ink")
            prev = ly
        if labels:
            s.text(bx + bw / 2, y + h + 12, labels[i], size="tiny", role="muted")
    s.line(x - 4, y + h, x + len(values) * (bw + gap) - gap + 4, y + h, stroke="s-ink", sw=1)


# ----- 12. layer diagrams ------------------------------------------------------------------------------------

def layer_flow(name, title, rows, note_lines=(), bw=108, bh=38, gap=22, x0=24, residuals=(), breaks=()):
    """Rows of boxes connected left to right; each row continues below the previous one.

    rows = [[(label, role), ...], ...]; residuals = [(row, from_col, to_col, label)] drawn as arcs over a row;
    breaks = indices of rows that are not joined to the next row.
    """
    s = Svg(name, W, 100, title)
    y = 46
    ys = []
    for r, row in enumerate(rows):
        for i, (lab, role) in enumerate(row):
            bx = x0 + i * (bw + gap)
            s.box(bx, y, bw, bh, lab, role=role, size="small")
            if i:
                s.arrow(bx - gap + 2, y + bh / 2, bx - 2, y + bh / 2, role="muted", sw=1.1)
        ys.append(y)
        if r < len(rows) - 1 and r not in breaks:
            last = x0 + (len(row) - 1) * (bw + gap) + bw / 2
            s.path(f"M{last:.1f},{y + bh + 2:.1f} L{last:.1f},{y + bh + 12:.1f} L{x0 + bw / 2:.1f},{y + bh + 12:.1f} "
                   f"L{x0 + bw / 2:.1f},{y + bh + 64:.1f}", stroke="s-muted", sw=1.1, arrow="muted")
        y += bh + 66
    for r, c0, c1, lab in residuals:
        yy = ys[r] - 16
        xa = x0 + c0 * (bw + gap) + bw / 2
        xb = x0 + c1 * (bw + gap) + bw / 2
        s.path(f"M{xa:.1f},{ys[r]:.1f} L{xa:.1f},{yy:.1f} L{xb:.1f},{yy:.1f} L{xb:.1f},{ys[r] - 2:.1f}",
               stroke="s-hl", sw=1.2, dash="4 3", arrow="hl")
        s.text((xa + xb) / 2, yy - 10, lab, size="tiny", role="hl")
    bottom = ys[-1] + bh
    return finish(s, note(s, 40, bottom + 30, note_lines) if note_lines else bottom)
