"""Figures for tutorials/03-parallel-reduction.md."""
from .svg import Svg


def fig_tree(name):
    s = Svg(name, 720, 305, "Block reduction with sequential addressing (8 values shown)")
    x0, cw, rh = 120, 64, 62
    vals = [3, 1, 4, 1, 5, 9, 2, 6]
    s.text(x0 - 12, 36, "cache[ ]", anchor="end", size="small", mono=True)
    for i, v in enumerate(vals):
        s.box(x0 + i * cw, 22, cw - 10, 28, str(v), role="a", size="small")
        s.text(x0 + i * cw + (cw - 10) / 2, 12, f"tid {i}", size="small", role="muted")
    level = vals[:]
    stride = 4
    y = 22
    while stride >= 1:
        ny = y + rh
        new = [level[i] + level[i + stride] for i in range(stride)]
        s.text(x0 - 12, ny + 14, f"stride {stride}", anchor="end", size="small")
        for i in range(stride):
            s.arrow(x0 + i * cw + 20, y + 28, x0 + i * cw + 20, ny, role="c", sw=1)
            s.arrow(x0 + (i + stride) * cw + 22, y + 28, x0 + i * cw + 34, ny, role="b", sw=1)
            s.box(x0 + i * cw, ny, cw - 10, 28, str(new[i]), role="c", size="small")
        for i in range(stride, len(vals)):
            s.rect(x0 + i * cw, ny, cw - 10, 28, fill="fig-panel", stroke="s-line", sw=0.6, rx=4)
        s.line(x0 - 4, ny + 36, x0 + 8 * cw - 6, ny + 36, stroke="s-hl", sw=1, dash="4 3")
        level = new
        y = ny
        stride //= 2
    s.text(x0 + 8 * cw - 6, 254, "__syncthreads() after every level", anchor="end", size="small", role="hl")
    s.text(360, 276, "Active threads stay contiguous (0 … stride−1): whole warps retire together,", size="small",
           role="muted")
    s.text(360, 293, "and the reads of cache[tid + stride] are conflict-free.", size="small", role="muted")
    return s


def fig_shuffle(name):
    s = Svg(name, 720, 340, "Warp reduction with __shfl_down_sync (16 lanes shown)")
    x0, cw, rh = 110, 36, 56
    lanes = 16
    s.text(x0 - 10, 30, "lane", anchor="end", size="small", role="muted")
    for l in range(lanes):
        s.text(x0 + l * cw + 14, 30, str(l), size="small", role="muted")
    offsets = [8, 4, 2, 1]
    for step in range(len(offsets) + 1):
        y = 44 + step * rh
        valid = lanes >> step
        for l in range(lanes):
            s.rect(x0 + l * cw, y, 28, 20, fill="f-c2" if l < valid else "fig-panel", stroke="s-line", sw=0.6, rx=3)
        if step < len(offsets):
            off = offsets[step]
            s.text(x0 - 10, y + rh / 2 + 8, f"δ = {off}", anchor="end", size="small")
            for l in range(off):
                s.arrow(x0 + (l + off) * cw + 14, y + 20, x0 + l * cw + 16, y + rh - 2, role="b", sw=0.9)
    s.text(x0 + 14, 44 + 4 * rh + 34, "lane 0 holds the sum", anchor="start", size="small", role="c")
    s.text(470, 44 + 4 * rh + 34, "no shared memory, no __syncthreads()", size="small", role="muted")
    return s


def fig_two_level(name):
    s = Svg(name, 720, 250, "A full reduction: per-thread sums, block reductions, then across blocks")
    # input
    s.rect(20, 30, 680, 22, fill="f-a", stroke="s-a", sw=1)
    for i in range(1, 32):
        s.line(20 + i * 21.25, 30, 20 + i * 21.25, 52, stroke="s-line", sw=0.4)
    s.text(360, 18, "input x[0 … n−1]  (grid-stride loop: each thread sums many elements)", size="small")
    blocks = 4
    for b in range(blocks):
        bx = 40 + b * 170
        s.box(bx, 90, 130, 36, f"block {b}: registers", role="c", size="small")
        for k in range(3):
            s.line(bx + 20 + k * 45, 52, bx + 30 + k * 35, 90, stroke="s-a", sw=0.6)
        s.box(bx + 10, 150, 110, 30, "blockReduce", role="d", size="small")
        s.arrow(bx + 65, 126, bx + 65, 150, role="d", sw=1)
        s.arrow(bx + 65, 180, 360, 210, role="b", sw=1)
    s.box(300, 210, 120, 30, "total", role="b", fill="f-b2", size="small", bold=True)
    s.text(440, 218, "atomicAdd, or a second", anchor="start", size="small", role="b")
    s.text(440, 234, "kernel with one block", anchor="start", size="small", role="b")
    return s
