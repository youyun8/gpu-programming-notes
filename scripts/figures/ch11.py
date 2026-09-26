"""Figures for tutorials/11-scan.md."""
from .svg import Svg


def fig_kogge_stone(name):
    s = Svg(name, 720, 350, "Kogge-Stone inclusive scan of 8 values: log2(8) = 3 steps")
    x0, cw, rh = 150, 64, 72
    vals = [3, 1, 4, 1, 5, 9, 2, 6]
    for i in range(8):
        s.text(x0 + i * cw + 26, 22, f"lane {i}", size="small", role="muted")
    rows = [("input", vals)]
    cur = vals[:]
    for d in (1, 2, 4):
        cur = [cur[i] + (cur[i - d] if i >= d else 0) for i in range(8)]
        rows.append((f"d = {d}", cur[:]))
    for r, (label, vs) in enumerate(rows):
        y = 36 + r * rh
        s.text(x0 - 14, y + 16, label, anchor="end", size="small", mono=r > 0)
        for i, v in enumerate(vs):
            active = r > 0 and i >= [1, 2, 4][r - 1]
            s.rect(x0 + i * cw, y, 52, 32, fill="f-c2" if active else ("f-a" if r == 0 else "fig-panel"),
                   stroke="s-ink", sw=1, rx=4)
            s.text(x0 + i * cw + 26, y + 16, str(v), size="small", mono=True)
        if r + 1 < len(rows):
            d = [1, 2, 4][r]
            for i in range(d, 8):
                s.arrow(x0 + (i - d) * cw + 30, y + 32, x0 + i * cw + 22, y + rh - 2, role="b", sw=0.9)
    s.text(360, 334, "Step d: lane l adds the value of lane l − d (__shfl_up_sync). Work n log2 n, depth log2 n.",
           size="small", role="muted")
    return s


def fig_hierarchy(name):
    s = Svg(name, 720, 340, "Scanning one 2048-item tile: registers, warps, block")
    s.text(20, 30, "1. each thread scans its 8 consecutive items sequentially (registers)", anchor="start",
           size="small", bold=True)
    x0, y0 = 20, 44
    for t in range(3):
        for j in range(8):
            s.rect(x0 + (t * 8 + j) * 20, y0, 18, 24, fill=["f-a", "f-b", "f-c", "f-d"][t], stroke="s-line",
                   sw=0.6, rx=2)
        s.text(x0 + t * 160 + 80, y0 + 40, f"thread {t}: total T{t}", size="small")
    s.text(x0 + 3 * 160 + 20, y0 + 12, "… up to thread 255", anchor="start", size="small", role="muted")
    s.text(20, 130, "2. warp scan of the thread totals (__shfl_up_sync); lane 31 holds the warp total",
           anchor="start", size="small", bold=True)
    for w in range(8):
        s.box(20 + w * 84, 146, 76, 30, f"warp {w}", role="c", size="small")
    s.text(20, 210, "3. warp 0 scans the 8 warp totals; every thread adds the totals of earlier warps",
           anchor="start", size="small", bold=True)
    s.box(20, 226, 660, 30, "warp_totals[0 … 7] → inclusive scan → carry for each warp", role="d", size="small")
    s.text(20, 288, "4. each item = its register prefix + its thread's exclusive prefix (+ the tile's carry-in)",
           anchor="start", size="small", bold=True)
    s.text(20, 314, "Only steps 2–3 synchronize; the sequential step does 7 of every 8 additions without any.",
           anchor="start", size="small", role="muted")
    return s


def fig_lookback(name):
    s = Svg(name, 720, 300, "Decoupled look-back: tile 5 reads its predecessors' published status")
    x0, cw = 40, 104
    status = [("P", "0–0: 12"), ("P", "0–1: 30"), ("A", "sum 7"), ("A", "sum 9"), ("X", "not ready"),
              ("·", "scanning")]
    roles = {"P": "c", "A": "b", "X": "hl", "·": "a"}
    for t, (flag, text) in enumerate(status):
        s.box(x0 + t * cw, 60, cw - 12, 56, f"tile {t}\n{flag}: {text}" if flag != "·" else f"tile {t}\n{text}",
              role=roles[flag], size="small")
    s.text(x0 + 5 * cw + 46, 40, "current tile", size="small", role="a")
    steps = [(4, "waits (X)"), (3, "+9 (A)"), (2, "+7 (A)"), (1, "+30 (P): stop")]
    for k, (p, label) in enumerate(steps):
        y = 150 + k * 26
        s.arrow(x0 + 5 * cw + 30, y, x0 + p * cw + 46, y, role="ink", sw=1)
        s.text(x0 + 5 * cw + 36, y, label, anchor="start", size="small")
    s.text(360, 268, "X: nothing published. A: the tile's own sum. P: the inclusive prefix up to that tile.",
           size="small", role="muted")
    s.text(360, 286, "Tile 5's exclusive prefix = 9 + 7 + 30 = 46; it then publishes P: 46 + its own sum.",
           size="small", role="muted")
    return s
