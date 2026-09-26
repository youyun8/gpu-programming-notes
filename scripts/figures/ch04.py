"""Figures for tutorials/04-tiled-matmul.md."""
from .svg import Svg


def fig_block_tiling(name):
    s = Svg(name, 700, 420, "Block tiling of C = AB: one output tile, its A row panel and B column panel")
    kx, ky = 40, 190          # A origin
    bx, by = 210, 30          # B origin
    kw, mh, nw = 128, 192, 192
    t = 48                    # tile size in px (4 tiles per side)
    sl = 32                   # K slice width in px (4 slices)
    ti, tj, ss = 1, 2, 1      # highlighted tile row/col and slice
    cx, cy = bx, ky
    # A
    s.grid(kx, ky, 4, 4, sl, t, stroke="s-line", fill_fn=None)
    s.rect(kx, cy + ti * t, kw, t, fill="f-a", stroke="s-a", sw=1.4)
    s.rect(kx + ss * sl, cy + ti * t, sl, t, fill="f-a2", stroke="s-a", sw=1.6)
    s.rect(kx, ky, kw, mh, fill="fig-none", stroke="s-ink", sw=1.4)
    # B
    s.rect(bx, by, nw, kw, fill="fig-paper", stroke="s-ink", sw=1.4)
    for i in range(1, 4):
        s.line(bx, by + i * sl, bx + nw, by + i * sl, stroke="s-line", sw=0.8)
    s.rect(bx + tj * t, by, t, kw, fill="f-b", stroke="s-b", sw=1.4)
    s.rect(bx + tj * t, by + ss * sl, t, sl, fill="f-b2", stroke="s-b", sw=1.6)
    # C
    s.grid(cx, cy, 4, 4, t, fill_fn=lambda r, c: "f-c2" if (r, c) == (ti, tj) else None)
    s.rect(cx + tj * t, cy + ti * t, t, t, fill="fig-none", stroke="s-c", sw=2.2)
    # projections
    s.line(kx + kw, cy + ti * t, cx + tj * t, cy + ti * t, stroke="s-a", sw=1, dash="4 3")
    s.line(kx + kw, cy + ti * t + t, cx + tj * t, cy + ti * t + t, stroke="s-a", sw=1, dash="4 3")
    s.line(bx + tj * t, by + kw, bx + tj * t, cy + ti * t, stroke="s-b", sw=1, dash="4 3")
    s.line(bx + tj * t + t, by + kw, bx + tj * t + t, cy + ti * t, stroke="s-b", sw=1, dash="4 3")
    # labels
    s.text(kx + kw / 2, ky + mh + 18, "A  (M × K)", bold=True)
    s.text(bx + nw + 12, by + kw / 2, "B  (K × N)", anchor="start", bold=True)
    s.text(cx + nw / 2, cy + mh + 18, "C  (M × N)", bold=True)
    s.brace_h(kx + ss * sl, kx + ss * sl + sl, ky - 10, "T", role="a", up=True)
    s.brace_v(bx - 10, by + ss * sl, by + ss * sl + sl, "T", role="b")
    s.brace_v(cx + nw + 10, cy + ti * t, cy + ti * t + t, "T", role="c", left=False)
    s.text(kx + ss * sl + sl / 2, cy + ti * t + t / 2, "s", role="a", italic=True, bold=True)
    s.text(bx + tj * t + t / 2, by + ss * sl + sl / 2, "s", role="b", italic=True, bold=True)
    s.text(cx + tj * t + t / 2, cy + ti * t + t / 2, "C_{IJ}", role="c", bold=True)
    # notes
    x = 460
    s.text(x, 250, "One block ↔ one T × T tile of C.", anchor="start", size="small")
    s.text(x, 272, "Phase s stages the A slice and the", anchor="start", size="small")
    s.text(x, 290, "B slice (dark) in shared memory,", anchor="start", size="small")
    s.text(x, 308, "then every thread does T FMAs.", anchor="start", size="small")
    s.text(x, 334, "⌈K / T⌉ phases walk the panels", anchor="start", size="small")
    s.text(x, 352, "from left to right / top to bottom.", anchor="start", size="small")
    return s


def fig_register_tile(name):
    s = Svg(name, 700, 356, "Register tiling: one thread's outer product per k step")
    # B row in shared memory
    cw = 5.5
    bx0, by0 = 250, 34
    s.text(bx0 - 8, by0 + 7, "b_tile[kk][0..63]", anchor="end", mono=True, size="small")
    tx = 5
    picks_b = [tx + 16 * j for j in range(4)]
    for c in range(64):
        s.rect(bx0 + c * cw, by0, cw, 14, fill="f-b2" if c in picks_b else "fig-paper",
               stroke="s-line", sw=0.5)
    s.rect(bx0, by0, 64 * cw, 14, fill="fig-none", stroke="s-b", sw=1.2)
    # A column (transposed tile row) in shared memory
    ch = 3.6
    ax0, ay0 = 120, 110
    ty = 3
    picks_a = [ty + 16 * i for i in range(4)]
    for r in range(64):
        s.rect(ax0, ay0 + r * ch, 14, ch, fill="f-a2" if r in picks_a else "fig-paper", stroke="s-line", sw=0.4)
    s.rect(ax0, ay0, 14, 64 * ch, fill="fig-none", stroke="s-a", sw=1.2)
    s.text(ax0 + 7, ay0 - 22, "a_tile[kk][0..63]", mono=True, size="small")
    s.text(ax0 + 7, ay0 - 8, "(A stored transposed)", size="small", role="muted")
    # fragments
    fx, fy = 300, 110
    cell = 40
    for j in range(4):
        s.box(fx + j * cell, fy - 50, cell, 30, f"b{j}", role="b", mono=True, size="small")
        s.arrow(bx0 + (picks_b[j] + 0.5) * cw, by0 + 14, fx + j * cell + cell / 2, fy - 50, role="b", sw=1)
    for i in range(4):
        s.box(fx - 70, fy + i * cell + 5, 30, cell - 10, f"a{i}", role="a", mono=True, size="small")
        s.arrow(ax0 + 14, ay0 + (picks_a[i] + 0.5) * ch, fx - 70, fy + i * cell + cell / 2, role="a", sw=1)
    s.grid(fx, fy, 4, 4, cell, fill_fn=lambda r, c: "f-c")
    for i in range(4):
        for j in range(4):
            s.text(fx + j * cell + cell / 2, fy + i * cell + cell / 2, f"c{i}{j}", mono=True, size="small",
                   role="c")
    s.text(fx - 20, fy + 2 * cell, "×", size="big", bold=True)
    s.text(fx + 2 * cell, fy + 4 * cell + 18, "acc[4][4]: 16 registers", size="small", role="c")
    # notes
    x = 480
    s.text(x, 130, "Per k step, thread (tx, ty):", anchor="start", size="small", bold=True)
    s.text(x, 152, "• 4 loads of A, 4 loads of B", anchor="start", size="small")
    s.text(x, 172, "• 16 FMAs: acc[i][j] += a_i b_j", anchor="start", size="small")
    s.text(x, 192, "• FMAs / load = 16 / 8 = 2", anchor="start", size="small")
    s.text(x, 222, "Stride-16 ownership:", anchor="start", size="small", bold=True)
    s.text(x, 242, "16 lanes read 16 consecutive", anchor="start", size="small")
    s.text(x, 260, "words → no bank conflicts,", anchor="start", size="small")
    s.text(x, 278, "coalesced stores of C.", anchor="start", size="small")
    s.text(fx + 2 * cell, 20, "tx = 5 picks columns 5, 21, 37, 53", size="small", role="b")
    return s


def fig_ladder(name):
    s = Svg(name, 700, 262, "Typical fraction of cuBLAS FP32 reached by each rung of the ladder")
    rows = [("Naive", 1, 5), ("Shared-memory tiling", 10, 20), ("4 × 4 register tiling", 40, 60),
            ("8 × 8, float4, double buffer, warp tiling", 80, 95)]
    x0, x1, y0, rh = 300, 620, 24, 44
    for p in range(0, 101, 20):
        x = x0 + (x1 - x0) * p / 100
        s.line(x, y0 - 6, x, y0 + rh * len(rows), stroke="s-line", sw=0.7, dash="3 3")
        s.text(x, y0 + rh * len(rows) + 14, f"{p} %", size="small", role="muted")
    roles = ["hl", "b", "a", "c"]
    for i, (label, lo, hi) in enumerate(rows):
        y = y0 + i * rh + 8
        s.text(x0 - 12, y + 12, label, anchor="end", size="small")
        xa = x0 + (x1 - x0) * lo / 100
        xb = x0 + (x1 - x0) * hi / 100
        s.rect(x0, y + 4, xb - x0, 16, fill=f"f-{roles[i]}", stroke="", sw=0)
        s.rect(xa, y, xb - xa, 24, fill=f"f-{roles[i]}2", stroke=f"s-{roles[i]}", sw=1.2, rx=3)
        s.text(xb + 6, y + 12, f"{lo}–{hi} %", anchor="start", size="small", role=roles[i], plate=True)
    s.text((x0 + x1) / 2, 246, "fraction of cuBLAS FP32 throughput (typical range)", size="small", role="muted")
    return s
