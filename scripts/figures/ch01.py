"""Figures for tutorials/01-execution-model.md."""
from .svg import Svg


def fig_hierarchy(name):
    s = Svg(name, 720, 315, "Grid, blocks, warps and threads, and where they run")
    # grid of blocks
    s.text(120, 22, "software: one kernel launch", size="small", bold=True)
    gx, gy, bw, bh = 20, 40, 60, 40
    for r in range(3):
        for c in range(3):
            hl = (r, c) == (1, 1)
            s.box(gx + c * (bw + 6), gy + r * (bh + 6), bw, bh, f"({c},{r})", role="c" if hl else "ink",
                  fill="f-c2" if hl else None, size="small", mono=True)
    s.text(gx + 96, gy + 150, "grid of blocks (blockIdx)", size="small")
    # block expanded into warps
    wx, wy = 250, 40
    s.rect(wx - 8, wy - 8, 200, 150, fill="f-c", stroke="s-c", sw=1.2, rx=6)
    s.line(gx + 2 * bw + 6, gy + bh + 6, wx - 8, wy - 8, stroke="s-c", sw=0.8, dash="3 3")
    s.line(gx + 2 * bw + 6, gy + 2 * bh + 6, wx - 8, wy + 142, stroke="s-c", sw=0.8, dash="3 3")
    for w in range(4):
        y = wy + w * 32
        s.text(wx + 4, y + 11, f"warp {w}", anchor="start", size="small")
        for t in range(32):
            s.rect(wx + 56 + t * 4, y + 4, 4, 14, fill="f-a2" if (w == 1 and t == 5) else "f-a", stroke="s-a",
                   sw=0.3)
    s.text(wx + 92, wy + 158, "block: up to 1024 threads", size="small")
    s.text(wx + 92, wy + 174, "warp = 32 consecutive threads", size="small")
    # hardware
    hx = 500
    s.text(hx + 100, 22, "hardware", size="small", bold=True)
    s.rect(hx, 36, 200, 200, fill="fig-panel", stroke="s-ink", sw=1.2, rx=6)
    s.text(hx + 100, 50, "GPU (A100: 108 SMs)", size="small", role="muted")
    for i in range(4):
        x = hx + 12 + (i % 2) * 94
        y = 64 + (i // 2) * 84
        s.box(x, y, 82, 72, "", role="ink")
        s.text(x + 41, y + 12, f"SM {i}", size="small", bold=True)
        for q in range(4):
            s.rect(x + 6 + q * 18, y + 28, 15, 34, fill="f-d", stroke="s-d", sw=0.8, rx=2)
        if i == 1:
            s.rect(x + 2, y + 2, 78, 68, fill="fig-none", stroke="s-c", sw=2, rx=4)
    s.text(hx + 100, 252, "4 warp schedulers per SM (purple)", size="small")
    s.arrow(wx + 192, wy + 60, hx + 110, 80, role="c", dash="5 3")
    s.text(360, 280, "Each block runs on one SM (green arrow); blocks start in any order and never migrate;", size="small", role="muted")
    s.text(360, 298, "each warp is issued by one scheduler, one instruction for all 32 lanes.", size="small",
           role="muted")
    return s


def fig_indexing(name):
    s = Svg(name, 720, 300, "2-D indexing and how warps are formed from a block")
    x0, y0, cell = 30, 50, 9
    bx_n, by_n, bs = 4, 3, 8  # blocks of 8 x 8 threads
    for by in range(by_n):
        for bx in range(bx_n):
            hl = (bx, by) == (2, 1)
            s.rect(x0 + bx * bs * cell, y0 + by * bs * cell, bs * cell, bs * cell, fill="f-c" if hl else "fig-paper",
                   stroke="s-line", sw=0.6)
    for i in range(1, bx_n * bs):
        s.line(x0 + i * cell, y0, x0 + i * cell, y0 + by_n * bs * cell, stroke="s-line", sw=0.3)
    for i in range(1, by_n * bs):
        s.line(x0, y0 + i * cell, x0 + bx_n * bs * cell, y0 + i * cell, stroke="s-line", sw=0.3)
    for i in range(1, bx_n):
        s.line(x0 + i * bs * cell, y0, x0 + i * bs * cell, y0 + by_n * bs * cell, stroke="s-ink", sw=1)
    for i in range(1, by_n):
        s.line(x0, y0 + i * bs * cell, x0 + bx_n * bs * cell, y0 + i * bs * cell, stroke="s-ink", sw=1)
    s.rect(x0, y0, bx_n * bs * cell, by_n * bs * cell, fill="fig-none", stroke="s-ink", sw=1.4)
    tx, ty = 5, 3
    cx = x0 + (2 * bs + tx) * cell
    cy = y0 + (1 * bs + ty) * cell
    s.rect(cx, cy, cell, cell, fill="f-hl2", stroke="s-hl", sw=1)
    s.line(x0, cy + cell / 2, cx, cy + cell / 2, stroke="s-hl", sw=0.8, dash="3 2")
    s.line(cx + cell / 2, y0, cx + cell / 2, cy, stroke="s-hl", sw=0.8, dash="3 2")
    s.text(x0 + 2.5 * bs * cell, y0 - 12, "blockIdx = (2, 1), threadIdx = (5, 3), blockDim = (8, 8)", size="small")
    s.text(x0 + bx_n * bs * cell / 2, y0 + by_n * bs * cell + 18, "col = 2·8 + 5 = 21,  row = 1·8 + 3 = 11",
           size="small", role="hl")
    # warp formation
    wx = 440
    s.text(wx + 120, 22, "warps of a 16 × 16 block vs a 32 × 8 block", size="small", bold=True)
    c2 = 7
    colors = ["f-a", "f-b", "f-c", "f-d", "f-a2", "f-b2", "f-c2", "f-d2"]
    for r in range(16):
        for c in range(16):
            w = (r * 16 + c) // 32
            s.rect(wx + c * c2, 50 + r * c2, c2, c2, fill=colors[w], stroke="s-line", sw=0.2)
    s.text(wx + 56, 50 + 16 * c2 + 14, "16 × 16: a warp = 2 rows", size="small")
    x2 = wx + 140
    c3 = 3.5
    for r in range(8):
        for c in range(32):
            w = (r * 32 + c) // 32
            s.rect(x2 + c * c3, 50 + r * c3 * 2, c3, c3 * 2, fill=colors[w], stroke="s-line", sw=0.2)
    s.text(x2 + 56, 50 + 16 * c3 + 14, "32 × 8: a warp = 1 row", size="small")
    s.text(wx + 120, 200, "τ = t_x + B_x t_y,   warp = ⌊τ / 32⌋,   lane = τ mod 32", size="small")
    return s


def fig_latency_hiding(name):
    s = Svg(name, 720, 290, "Latency hiding: the scheduler issues from whichever warp is ready")
    x0, unit = 110, 14
    lat = 12  # memory latency in units
    end = 42  # right edge of the plot, in units

    def warp_row(y, label, start):
        s.text(x0 - 10, y + 9, label, anchor="end", size="small")
        t = start
        while t < end:
            s.rect(x0 + t * unit, y, 2 * unit, 18, fill="f-c2", stroke="s-c", sw=0.8, rx=2)
            t += 2
            s.rect(x0 + t * unit, y + 5, min(lat, end - t) * unit, 8, fill="f-b", stroke="s-b", sw=0.6, rx=2)
            t += lat
    s.text(x0, 24, "one warp alone: the SM idles while its load is in flight", anchor="start", size="small",
           bold=True)
    warp_row(36, "warp 0", 0)
    s.text(x0, 90, "several warps: their compute fills the gaps", anchor="start", size="small", bold=True)
    for w in range(6):
        warp_row(102 + w * 22, f"warp {w}", 2 * w)
    s.legend(x0 + 385, 24, [("f-c2", "c", "issuing")])
    s.legend(x0 + 470, 24, [("f-b", "b", "waiting for memory")])
    s.text(360, 252, "Little's law: bytes in flight = bandwidth × latency, so enough warps", size="small",
           role="muted")
    s.text(360, 270, "(or enough independent loads per warp) must be resident.", size="small", role="muted")
    return s


def fig_occupancy(name):
    s = Svg(name, 720, 250, "Occupancy example: B = 256, r = 64, s = 32 KB on an A100")
    x0, unit, rh = 230, 48, 34
    rows = [("threads: ⌊2048 / 256⌋", 8), ("registers: ⌊65536 / (64·256)⌋", 4), ("shared memory: ⌊164 / 32⌋", 5),
            ("block slots: 32", 32)]
    for i, (label, v) in enumerate(rows):
        y = 30 + i * rh
        s.text(x0 - 10, y + 12, label, anchor="end", size="small")
        shown = min(v, 9)
        for b in range(shown):
            lim = b < 4
            s.rect(x0 + b * unit + 2, y, unit - 4, 24, fill="f-c2" if lim else "f-c", stroke="s-c", sw=0.8, rx=3)
        if v > 9:
            s.text(x0 + 9 * unit + 4, y + 12, "… 32", anchor="start", size="small")
        else:
            s.text(x0 + shown * unit + 8, y + 12, str(v), anchor="start", size="small", bold=True,
                   role="hl" if v == 4 else "ink")
    s.line(x0 + 4 * unit, 22, x0 + 4 * unit, 30 + 4 * rh, stroke="s-hl", sw=1.6, dash="5 3")
    s.text(360, 30 + 4 * rh + 18, "min = 4 blocks = 1024 threads → 50 % occupancy (registers are the limit)",
           size="small", role="hl")
    s.text(360, 230, "Each resource allows some number of resident blocks per SM; the smallest wins.", size="small",
           role="muted")
    return s
