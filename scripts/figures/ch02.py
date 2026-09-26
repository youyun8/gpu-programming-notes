"""Figures for tutorials/02-memory-hierarchy.md."""
from .svg import Svg


def fig_memory_levels(name):
    s = Svg(name, 720, 292, "Memory hierarchy of an A100: each level is larger and slower")
    levels = [("registers", "per thread", "~1 cycle", "256 KB / SM", "d"),
              ("shared memory / L1", "per block / per SM", "~20–30 cycles", "192 KB / SM", "c"),
              ("L2 cache", "whole GPU", "~200 cycles", "40 MB", "a"),
              ("global memory (HBM)", "whole GPU", "~400–800 cycles", "40–80 GB", "b")]
    cx, top, rh = 250, 20, 56
    for i, (label, scope, lat, size, role) in enumerate(levels):
        w = 150 + i * 90
        y = top + i * rh
        s.box(cx - w / 2, y, w, rh - 8, "", role=role)
        s.text(cx, y + 15, label, size="small", bold=True)
        s.text(cx, y + 33, scope, size="small", role="muted")
        s.text(470, y + 16, lat, anchor="start", size="small")
        s.text(600, y + 16, size, anchor="start", size="small")
    s.text(470, 12, "latency", anchor="start", size="small", bold=True)
    s.text(600, 12, "size", anchor="start", size="small", bold=True)
    s.arrow(700, 30, 700, 230, role="muted")
    s.text(360, 258, "Bandwidth drops with every level: DRAM delivers ~1.5 TB/s,", size="small", role="muted")
    s.text(360, 276, "an order of magnitude less than the SMs can consume.", size="small", role="muted")
    return s


def fig_coalescing(name):
    s = Svg(name, 720, 335, "How a warp's addresses become 32-byte sectors")
    x0, lw = 160, 13  # lane width in px (32 lanes = 416)
    cases = [("x[i]  (stride 1)", 1, "c", "4 sectors, 100 %"),
             ("x[2i] (stride 2)", 2, "b", "8 sectors, 50 %"),
             ("x[32i] (stride 32)", 32, "hl", "32 sectors*, 12.5 %")]
    y = 30
    for label, stride, role, note in cases:
        s.text(x0 - 10, y + 10, label, anchor="end", size="small", mono=True)
        # lanes
        for l in range(32):
            s.rect(x0 + l * lw, y, lw, 18, fill="f-ink", stroke="s-line", sw=0.4)
        if stride == 1:
            s.text(x0 + 16 * lw, y - 10, "lanes 0 … 31", size="small", role="muted")
        # memory: show sectors touched (32 bytes = 8 floats each), scaled to at most 32 sectors
        ym = y + 48
        nsec = min(32, (32 * stride + 7) // 8)
        total_sec = 32
        sw_px = 32 * lw / total_sec
        for sct in range(total_sec):
            used = sct < nsec if stride <= 2 else True
            s.rect(x0 + sct * sw_px, ym, sw_px, 16, fill=f"f-{role}2" if used else "fig-paper", stroke="s-line",
                   sw=0.4)
        for l in range(0, 32, 1 if stride == 32 else 4):
            addr_float = l * stride
            sct = addr_float // 8
            if stride == 32:
                sct = l
            tx = x0 + (sct + (addr_float % 8) / 8) * sw_px + 2
            s.line(x0 + l * lw + lw / 2, y + 18, tx, ym, stroke=f"s-{role}", sw=0.6)
        s.text(x0 + 32 * lw + 8, ym + 8, note, anchor="start", size="small", role=role)
        y += 98
    s.text(x0 - 10, 86, "sectors", anchor="end", size="small", role="muted")
    s.text(360, 306, "Upper row: the 32 lanes. Lower row: 32-byte sectors (8 floats) fetched.", size="small",
           role="muted")
    s.text(360, 322, "Efficiency = useful / fetched bytes. *Each sector 128 B apart.", size="small", role="muted")
    return s


def fig_bank_conflicts(name):
    s = Svg(name, 720, 318, "Reading a column of a 32 × 32 tile: without and with padding")
    cell = 7
    for idx, (pad, title) in enumerate([(0, "float tile[32][32]"), (1, "float tile[32][33]")]):
        x0 = 40 + idx * 350
        y0 = 50
        s.text(x0 + 16 * cell, y0 - 26, title, mono=True, size="small", bold=True)
        # draw the 32 rows as they sit in the 32 banks (row r starts at bank (r * (32 + pad)) % 32)
        col = 3
        for r in range(32):
            for b in range(32):
                s.rect(x0 + b * cell, y0 + r * cell, cell, cell, fill="fig-paper", stroke="s-line", sw=0.2)
            bank = (r * (32 + pad) + col) % 32
            s.rect(x0 + bank * cell, y0 + r * cell, cell, cell, fill="f-hl2" if pad == 0 else "f-c2",
                   stroke="s-hl" if pad == 0 else "s-c", sw=0.6)
        s.rect(x0, y0, 32 * cell, 32 * cell, fill="fig-none", stroke="s-ink", sw=1)
        s.text(x0 + 16 * cell, y0 - 10, "bank 0 → 31", size="small", role="muted")
        s.text(x0 - 6, y0 + 16 * cell, "row", anchor="end", size="small", role="muted")
        verdict = "all 32 rows in bank 3: 32-way conflict" if pad == 0 else \
            "row r in bank (r + 3) mod 32: no conflict"
        s.text(x0 + 16 * cell, y0 + 32 * cell + 18, verdict, size="small", role="hl" if pad == 0 else "c")
    return s


def fig_transpose(name):
    s = Svg(name, 720, 280, "Transpose through a padded shared-memory tile")
    # input matrix with a highlighted tile
    ix, iy, c = 30, 60, 14
    s.grid(ix, iy, 8, 8, c, fill_fn=lambda r, cc: "f-a" if 2 <= r < 4 and 4 <= cc < 6 else None, sw=0.4)
    s.text(ix + 56, iy - 30, "in (R × C)", size="small", bold=True)
    s.text(ix + 56, iy - 14, "tile (by, bx)", size="small", role="a")
    # shared tile
    sx, sy = 230, 40
    for r in range(8):
        for cc in range(9):
            s.rect(sx + cc * 16, sy + r * 16, 16, 16, fill="f-c" if cc < 8 else "fig-panel", stroke="s-line", sw=0.4)
    s.rect(sx, sy, 8 * 16, 8 * 16, fill="fig-none", stroke="s-c", sw=1.4)
    s.text(sx + 72, sy - 14, "tile[32][32 + 1] (shared)", size="small", bold=True)
    s.text(sx + 9 * 16 + 12, sy + 64, "pad", size="small", role="muted", rotate=-90)
    s.arrow(ix + 8 * c + 10, iy + 42, sx - 6, sy + 30, role="a")
    s.text(ix + 8 * c + 14, iy + 72, "coalesced", size="small", role="a", anchor="start")
    s.text(ix + 8 * c + 14, iy + 88, "row reads", size="small", role="a", anchor="start")
    s.arrow(sx + 64, sy + 8, sx + 64, sy + 120, role="c", dash="4 3")
    s.text(sx + 70, sy + 150, "read down a column:", size="small", role="c")
    s.text(sx + 70, sy + 166, "conflict-free thanks to the pad", size="small", role="c")
    # output matrix
    ox, oy = 520, 60
    s.grid(ox, oy, 8, 8, c, fill_fn=lambda r, cc: "f-b" if 4 <= r < 6 and 2 <= cc < 4 else None, sw=0.4)
    s.text(ox + 56, oy - 30, "out (C × R)", size="small", bold=True)
    s.text(ox + 56, oy - 14, "tile (bx, by)", size="small", role="b")
    s.arrow(sx + 8 * 16 + 30, sy + 90, ox - 6, oy + 70, role="b")
    s.text(ox - 72, oy + 110, "coalesced row writes", size="small", role="b")
    s.text(360, 262, "Both global accesses walk along rows; the transposition happens in shared memory.",
           size="small", role="muted")
    return s
