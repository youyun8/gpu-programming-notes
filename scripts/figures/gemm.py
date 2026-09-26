"""Figures for the GEMM technique pages (tutorials/gemm/)."""
from .svg import Svg


# ----------------------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------------------
def fig_hierarchy(name):
    s = Svg(name, 740, 290, "The GEMM tiling hierarchy: grid, block tile, warp tile, thread tile")
    y0 = 50
    # Panel 1: C as a grid of block tiles
    x1, n1, c1 = 20, 8, 17.5
    s.grid(x1, y0, n1, n1, c1, fill_fn=lambda r, c: "f-c2" if (r, c) == (2, 5) else None, sw=0.5)
    s.text(x1 + 70, y0 - 16, "global memory / L2", size="small", role="muted")
    s.text(x1 + 70, y0 + 160, "C: a grid of", size="small")
    s.text(x1 + 70, y0 + 176, "128 × 128 block tiles", size="small")
    # Panel 2: block tile with 2 x 4 warps
    x2 = 205
    ww, wh = 35, 70
    for r in range(2):
        for c in range(4):
            hl = (r, c) == (1, 2)
            s.rect(x2 + c * ww, y0 + r * wh, ww, wh, fill="f-c2" if hl else "f-c", stroke="s-c", sw=0.8)
            s.text(x2 + c * ww + ww / 2, y0 + r * wh + wh / 2, f"w{4 * r + c}", size="small", mono=True)
    s.rect(x2, y0, 140, 140, fill="fig-none", stroke="s-ink", sw=1.4)
    s.text(x2 + 70, y0 - 16, "one block (shared memory)", size="small", role="muted")
    s.text(x2 + 70, y0 + 160, "8 warps in a 2 × 4 grid", size="small")
    s.text(x2 + 70, y0 + 176, "block tile 128 × 128", size="small")
    # zoom lines 1 -> 2
    s.line(x1 + 6 * c1, y0 + 2 * c1, x2, y0, stroke="s-muted", sw=0.8, dash="3 3")
    s.line(x1 + 6 * c1, y0 + 3 * c1, x2, y0 + 140, stroke="s-muted", sw=0.8, dash="3 3")
    # Panel 3: warp tile 64 x 32 (2 x 2 sub-tiles, 8 x 4 lanes)
    x3 = 420
    sw_, sh_ = 35, 70
    for r in range(2):
        for c in range(2):
            s.rect(x3 + c * sw_, y0 + r * sh_, sw_, sh_, fill="f-a" if (r + c) % 2 == 0 else "f-b",
                   stroke="s-line", sw=0.8)
    for r in range(8):
        for c in range(4):
            lane = 4 * r + c
            s.rect(x3 + c * 8.75, y0 + r * 8.75, 8.75, 8.75, fill="f-d2" if lane == 9 else "fig-none",
                   stroke="s-muted", sw=0.4)
    s.rect(x3, y0, 70, 140, fill="fig-none", stroke="s-c", sw=1.6)
    s.text(x3 + 35, y0 - 16, "one warp", size="small", role="muted")
    s.text(x3 + 35, y0 + 160, "warp tile 64 × 32:", size="small")
    s.text(x3 + 35, y0 + 176, "2 × 2 sub-tiles 32 × 16", size="small")
    s.text(x3 + 35, y0 + 192, "8 × 4 lanes each", size="small")
    s.line(x2 + 3 * ww, y0 + wh, x3, y0, stroke="s-muted", sw=0.8, dash="3 3")
    s.line(x2 + 3 * ww, y0 + 2 * wh, x3, y0 + 140, stroke="s-muted", sw=0.8, dash="3 3")
    # Panel 4: one lane's 8 x 8 accumulators = 2 x 2 patches of 4 x 4
    x4, c4 = 585, 14
    s.grid(x4, y0 + 14, 8, 8, c4, fill_fn=lambda r, c: ("f-a" if (r // 4 + c // 4) % 2 == 0 else "f-b"), sw=0.4)
    s.line(x4 + 4 * c4, y0 + 14, x4 + 4 * c4, y0 + 14 + 8 * c4, stroke="s-ink", sw=1.2)
    s.line(x4, y0 + 14 + 4 * c4, x4 + 8 * c4, y0 + 14 + 4 * c4, stroke="s-ink", sw=1.2)
    s.rect(x4, y0 + 14, 8 * c4, 8 * c4, fill="fig-none", stroke="s-d", sw=1.8)
    s.text(x4 + 56, y0 - 16, "one lane (registers)", size="small", role="muted")
    s.text(x4 + 56, y0 + 160, "8 × 8 accumulators:", size="small")
    s.text(x4 + 56, y0 + 176, "a 4 × 4 patch per sub-tile", size="small")
    s.text(x4 + 56, y0 + 192, "(tensor cores: fragments)", size="small", role="muted")
    s.line(x3 + 8.75, y0 + 2 * 8.75, x4, y0 + 14, stroke="s-d", sw=0.8, dash="3 3")
    s.line(x3 + 8.75, y0 + 3 * 8.75, x4, y0 + 14 + 8 * c4, stroke="s-d", sw=0.8, dash="3 3")
    s.text(370, 270, "Each level re-uses what the level above staged: global → shared → registers.",
           size="small", role="muted")
    return s


# ----------------------------------------------------------------------------------------
# 01: vectorized accesses
# ----------------------------------------------------------------------------------------
def fig_thread_map(name):
    s = Svg(name, 720, 330, "Output ownership in 01-vectorized.cu: split 4-wide patches")
    x0, y0, cell = 40, 40, 2.0  # 128 x 128 outputs at 2 px
    size = 128 * cell
    s.rect(x0, y0, size, size, fill="f-c", stroke="s-ink", sw=1.2)
    # 16 x 16 thread grid, lines every 4 outputs within each half
    for i in range(1, 32):
        s.line(x0 + i * 4 * cell, y0, x0 + i * 4 * cell, y0 + size, stroke="s-line", sw=0.3)
        s.line(x0, y0 + i * 4 * cell, x0 + size, y0 + i * 4 * cell, stroke="s-line", sw=0.3)
    s.line(x0 + 64 * cell, y0, x0 + 64 * cell, y0 + size, stroke="s-ink", sw=1)
    s.line(x0, y0 + 64 * cell, x0 + size, y0 + 64 * cell, stroke="s-ink", sw=1)
    tx, ty = 2, 1
    for hr in range(2):
        for hc in range(2):
            s.rect(x0 + (64 * hc + 4 * tx) * cell, y0 + (64 * hr + 4 * ty) * cell, 4 * cell, 4 * cell,
                   fill="f-hl2", stroke="s-hl", sw=1)
    # warp 0 footprint: tx 0..15, ty 0..1
    for hr in range(2):
        for hc in range(2):
            s.rect(x0 + 64 * hc * cell, y0 + 64 * hr * cell, 64 * cell, 8 * cell, fill="fig-none", stroke="s-a",
                   sw=1.4, extra=' stroke-dasharray="4 2"')
    s.brace_h(x0, x0 + 64 * cell, y0 - 12, "64", up=True)
    s.brace_h(x0 + 64 * cell, x0 + size, y0 - 12, "64", up=True)
    s.text(x0 + size / 2, y0 + size + 18, "block tile 128 × 128 (C)", size="small")
    x = 330
    s.legend(x, 70, [("f-hl2", "hl", "thread (tx = 2, ty = 1): four 4 × 4 patches"),
                     ("fig-none", "a", "warp 0 (tx = 0..15, ty = 0..1)")])
    notes = ["16 threads × 4 columns = 64 per half of the tile.",
             "Thread (tx, ty) owns rows 4ty..4ty+3 and 64+4ty..",
             "and columns 4tx..4tx+3 and 64+4tx..",
             "",
             "Fragment loads per k step: a_s[kk][4ty], a_s[kk][64+4ty],",
             "b_s[kk][4tx], b_s[kk][64+4tx]: four LDS.128.",
             "Lanes tx = 0..7 read 8 consecutive float4 = 128 bytes:",
             "one conflict-free shared-memory wavefront.",
             "",
             "Adjacent 4 × 4 patches (4tx..4tx+3 and 4tx+4..) would",
             "put lanes 32 bytes apart: 2 wavefronts per load."]
    for i, t in enumerate(notes):
        if t:
            s.text(x, 125 + i * 19, t, anchor="start", size="small", plain=True)
    return s


def fig_lds128(name):
    s = Svg(name, 720, 300, "128-bit shared loads: 8 lanes per wavefront; contiguous vs 32-byte stride")
    x0, bw = 150, 14
    def banks(y, title, lanes_of_bank, note, role):
        s.text(x0 - 10, y + 11, title, anchor="end", size="small", bold=True)
        for bnk in range(32):
            lane = lanes_of_bank(bnk)
            fill = f"f-{role}2" if lane is not None else "fig-paper"
            s.rect(x0 + bnk * bw, y, bw, 22, fill=fill, stroke="s-line", sw=0.6)
            if lane is not None:
                s.text(x0 + bnk * bw + bw / 2, y + 11, str(lane), size="small", mono=True)
        s.text(x0 + 32 * bw + 10, y + 11, note, anchor="start", size="small", role=role)
    for bnk in range(0, 32, 4):
        s.text(x0 + bnk * bw + 2 * bw, 36, f"b{bnk}–{bnk + 3}", size="small", role="muted")
    s.text(x0 + 16 * bw, 16, "32 banks × 4 bytes = one 128-byte wavefront (cell number = lane)", size="small")
    # contiguous: lane l reads bytes 16l..16l+15 -> banks 4l..4l+3
    banks(52, "16-byte stride", lambda b: b // 4, "1 wavefront", "c")
    # stride 32B: lanes 0..3 in first 128B, lanes 4..7 wrap to next line (same banks)
    banks(96, "32-byte stride", lambda b: b // 8 if b % 8 < 4 else None, "lanes 0–3", "hl")
    banks(124, "", lambda b: 4 + b // 8 if b % 8 < 4 else None, "lanes 4–7", "hl")
    s.text(x0 - 10, 135, "(second pass)", anchor="end", size="small", role="muted")
    s.text(x0 + 32 * bw + 10, 150, "2 wavefronts", anchor="start", size="small", role="hl")
    lines = ["A 128-bit load is served 8 lanes at a time (8 × 16 B = 128 B, one pass over the 32 banks).",
             "Consecutive lanes on consecutive 16-byte chunks fill each pass exactly: no conflicts.",
             "With a 32-byte stride (lane l owning columns 8l..8l+7), 8 lanes span 256 bytes and each",
             "pass needs two wavefronts: a 2-way conflict. Hence the split ownership 4tx and 64 + 4tx."]
    for i, t in enumerate(lines):
        s.text(20, 190 + i * 22, t, anchor="start", size="small")
    return s


# ----------------------------------------------------------------------------------------
# 02: double buffering
# ----------------------------------------------------------------------------------------
def fig_double_buffer(name):
    s = Svg(name, 720, 300, "Timeline of single vs double buffering for one warp")
    x0, unit = 150, 26
    def row(y, label, segs):
        s.text(x0 - 10, y + 12, label, anchor="end", size="small", bold=True)
        for (start, length, role, text) in segs:
            s.box(x0 + start * unit, y, length * unit, 24, text, role=role, size="small", rx=3)
    # single buffering: load (4) | sync | compute (3) | sync, repeated
    segs = []
    t = 0
    for i in range(3):
        segs.append((t, 3, "b", f"load {i}"))
        t += 3
        segs.append((t, 0.5, "hl", ""))
        t += 0.5
        segs.append((t, 3, "c", f"math {i}"))
        t += 3
        segs.append((t, 0.5, "hl", ""))
        t += 0.5
    row(50, "single buffer", segs[:11])
    # double buffering: loads issued before math, overlap
    segs2 = [(0, 3, "b", "load 0"), (3, 0.5, "hl", "")]
    t = 3.5
    for i in range(4):
        segs2.append((t, 3, "c", f"math {i}"))
        t += 3
        segs2.append((t, 0.5, "hl", ""))
        t += 0.5
    row(110, "double buffer", segs2)
    for i in range(3):
        st = 3.5 + i * 3.5
        s.box(x0 + st * unit, 140, 3 * unit, 20, f"load {i + 1}", role="b", size="small",
              rx=3, dash="4 2")
    s.line(x0, 40, x0 + 21 * unit, 40, stroke="s-muted", sw=1, arrow="muted")
    s.text(x0 + 21 * unit, 26, "time", anchor="end", size="small", role="muted")
    s.legend(x0, 200, [("f-b", "b", "global → registers → shared (latency ~ hundreds of cycles)"),
                       ("f-c", "c", "FMAs on the shared tiles"),
                       ("f-hl", "hl", "__syncthreads()")])
    s.text(x0, 265, "With one buffer the warp waits for every load; with two, slice s + 1 is fetched",
           anchor="start", size="small")
    s.text(x0, 283, "while slice s is computed, and one barrier per slice suffices.", anchor="start", size="small")
    return s


def fig_buffer_rotation(name):
    s = Svg(name, 720, 222, "Who reads and writes which buffer at each step (double buffering)")
    x0, cw = 170, 120
    s.text(x0 - 10, 40, "step", anchor="end", size="small", bold=True)
    for sidx in range(4):
        s.text(x0 + sidx * cw + cw / 2, 40, f"s = {sidx}", size="small", bold=True)
    for bidx in range(2):
        y = 60 + bidx * 60
        s.text(x0 - 10, y + 20, f"buffer {bidx}", anchor="end", size="small", bold=True)
        for sidx in range(4):
            compute = sidx % 2 == bidx
            s.box(x0 + sidx * cw + 6, y, cw - 12, 40,
                  f"read slice {sidx}" if compute else f"write slice {sidx + 1}",
                  role="c" if compute else "b", size="small")
    for sidx in range(4):
        x = x0 + (sidx + 1) * cw
        s.line(x, 52, x, 172, stroke="s-hl", sw=2)
    s.text(x0 + 2 * cw, 190, "Red: the one __syncthreads() per step. A buffer is rewritten only after", size="small")
    s.text(x0 + 2 * cw, 208, "the barrier that ends its last read.", size="small")
    return s


# ----------------------------------------------------------------------------------------
# 03: async copies
# ----------------------------------------------------------------------------------------
def fig_copy_paths(name):
    s = Svg(name, 720, 300, "Three ways from global to shared memory")
    cols = [("Ampere and older", 20), ("cp.async (sm_80+)", 260), ("TMA (sm_90+)", 500)]
    for title, x in cols:
        s.text(x + 100, 22, title, bold=True, size="small")
    # classic
    x = 20
    s.box(x + 15, 40, 170, 34, "global (L2 / DRAM)", role="muted", size="small")
    s.box(x + 15, 110, 170, 34, "registers", role="d", size="small")
    s.box(x + 15, 180, 170, 34, "shared memory", role="c", size="small")
    s.arrow(x + 100, 74, x + 100, 110, role="ink")
    s.arrow(x + 100, 144, x + 100, 180, role="ink")
    s.text(x + 108, 92, "LDG (via L1)", anchor="start", size="small")
    s.text(x + 108, 162, "STS", anchor="start", size="small")
    s.text(x + 100, 236, "2 instructions per chunk,", size="small")
    s.text(x + 100, 254, "registers held while in flight", size="small")
    # cp.async
    x = 260
    s.box(x + 15, 40, 170, 34, "global (L2 / DRAM)", role="muted", size="small")
    s.box(x + 15, 110, 170, 34, "L1 (.ca) or bypass (.cg)", role="line", size="small")
    s.box(x + 15, 180, 170, 34, "shared memory", role="c", size="small")
    s.arrow(x + 100, 74, x + 100, 110, role="a")
    s.arrow(x + 100, 144, x + 100, 180, role="a")
    s.text(x + 108, 92, "cp.async", anchor="start", size="small", role="a")
    s.text(x + 100, 236, "per-thread 4/8/16-byte copies,", size="small")
    s.text(x + 100, 254, "no registers; wait per group", size="small")
    # TMA
    x = 500
    s.box(x + 15, 40, 170, 34, "global (L2 / DRAM)", role="muted", size="small")
    s.box(x + 15, 110, 170, 34, "TMA unit (per SM)", role="b", size="small")
    s.box(x + 15, 180, 170, 34, "shared memory", role="c", size="small")
    s.arrow(x + 100, 74, x + 100, 110, role="b")
    s.arrow(x + 100, 144, x + 100, 180, role="b")
    s.text(x + 108, 92, "whole tile", anchor="start", size="small", role="b")
    s.text(x + 100, 236, "one thread issues a whole", size="small")
    s.text(x + 100, 254, "2-D tile; an mbarrier counts it", size="small")
    return s


def fig_pipeline(name):
    s = Svg(name, 720, 280, "A 3-stage cp.async pipeline")
    x0, cw = 110, 92
    stages = 3
    s.text(x0 - 10, 30, "step s", anchor="end", size="small", bold=True)
    for st in range(6):
        s.text(x0 + st * cw + cw / 2, 30, str(st), size="small", bold=True)
    for stage in range(stages):
        y = 46 + stage * 46
        s.text(x0 - 10, y + 17, f"stage {stage}", anchor="end", size="small", bold=True)
        for st in range(6):
            # at step st: compute slice st in stage st % 3; slices st+1, st+2 in flight
            if st % stages == stage:
                s.box(x0 + st * cw + 4, y, cw - 8, 34, f"math {st}", role="c", size="small")
            else:
                sl = st + ((stage - st) % stages)
                s.box(x0 + st * cw + 4, y, cw - 8, 34, f"copy {sl}", role="a", size="small", dash="4 2")
    s.text(x0, 200, "red lines: __pipeline_wait_prior(1) + __syncthreads()", anchor="start", size="small",
           role="hl")
    for st in range(6):
        s.line(x0 + st * cw + 2, 44, x0 + st * cw + 2, 184, stroke="s-hl", sw=1.6)
    s.text(20, 230, "Before step s: wait until at most kStages − 2 = 1 group is pending (slice s has landed), barrier,",
           anchor="start", size="small")
    s.text(20, 250, "then issue slice s + 2 into the stage slice s − 1 just vacated. Two slices are always in flight.",
           anchor="start", size="small")
    return s


# ----------------------------------------------------------------------------------------
# 04: warp tiling
# ----------------------------------------------------------------------------------------
def fig_warp_tile(name):
    s = Svg(name, 720, 440, "Warp tiling in 04-warp-tiling.cu")
    x0, y0, sc = 56, 40, 1.7   # block tile 128 x 128 at 1.7 px
    bs = 128 * sc
    zx, zy, zs = 312, 30, 5.6  # zoom of warp 6: 32 wide -> 179, 64 tall -> 358
    # zoom lines first: the warp tiles drawn on top hide them
    s.line(x0 + 3 * 32 * sc, y0 + 64 * sc, zx, zy, stroke="s-hl", sw=0.8, dash="3 3")
    s.line(x0 + 3 * 32 * sc, y0 + bs, zx, zy + 64 * zs, stroke="s-hl", sw=0.8, dash="3 3")
    colors = ["f-a", "f-b", "f-c", "f-d", "f-b", "f-c", "f-d", "f-a"]
    for w in range(8):
        wm, wn = w // 4, w % 4
        s.rect(x0 + wn * 32 * sc, y0 + wm * 64 * sc, 32 * sc, 64 * sc, fill=colors[w], stroke="s-line", sw=0.8)
        s.text(x0 + wn * 32 * sc + 16 * sc, y0 + wm * 64 * sc + 32 * sc, f"w{w}", size="small", mono=True)
    s.rect(x0, y0, bs, bs, fill="fig-none", stroke="s-ink", sw=1.4)
    s.rect(x0 + 2 * 32 * sc, y0 + 64 * sc, 32 * sc, 64 * sc, fill="fig-none", stroke="s-hl", sw=2)
    s.brace_h(x0, x0 + 32 * sc, y0 - 10, "32", up=True)
    s.brace_v(x0 - 10, y0, y0 + 64 * sc, "64")
    # zoom of warp 6
    for im in range(2):
        for jn in range(2):
            sx, sy = zx + jn * 16 * zs, zy + im * 32 * zs
            s.rect(sx, sy, 16 * zs, 32 * zs, fill="f-c" if (im + jn) % 2 == 0 else "fig-panel", stroke="s-ink",
                   sw=1)
            for lm in range(8):
                for ln in range(4):
                    lane = 4 * lm + ln
                    s.rect(sx + ln * 4 * zs, sy + lm * 4 * zs, 4 * zs, 4 * zs,
                           fill="f-hl2" if lane == 9 else "fig-none", stroke="s-line", sw=0.4)
                    if im == 0 and jn == 0:
                        s.text(sx + ln * 4 * zs + 2 * zs, sy + lm * 4 * zs + 2 * zs, str(lane), size="tiny",
                               mono=True)
    s.rect(zx, zy, 32 * zs, 64 * zs, fill="fig-none", stroke="s-hl", sw=2)
    s.text(zx + 16 * zs, zy + 64 * zs + 18, "warp tile of w6: 64 × 32", size="small")
    s.text(x0 + bs / 2, y0 + bs + 18, "block tile 128 × 128", size="small", plate=True)
    s.text(x0 + bs / 2, y0 + bs + 38, "8 warps (w0–w7) in a 2 × 4 grid", size="small", plate=True)
    notes = ["2 × 2 sub-tiles of 32 × 16.", "In each, the 32 lanes form an", "8 × 4 grid of 4 × 4 patches",
             "(numbers: lane ids).", "", "Lane 9 (red) owns 4 patches:", "rows 4·⌊9/4⌋ + {0, 32} + i,",
             "cols 4·(9 mod 4) + {0, 16} + j,", "i, j = 0 … 3.", "", "Per k step: 2 float4 of A and", "2 float4 of B per lane."]
    for i, t in enumerate(notes):
        if t:
            s.text(508, 50 + i * 21, t, anchor="start", size="small")
    return s


# ----------------------------------------------------------------------------------------
# 05: tile swizzle
# ----------------------------------------------------------------------------------------
def _grouped(pid, tiles_m, tiles_n, group_m):
    if group_m <= 0:
        return pid // tiles_n, pid % tiles_n
    per = group_m * tiles_n
    first = (pid // per) * group_m
    rows = min(tiles_m - first, group_m)
    within = pid % per
    return first + within % rows, within // rows


def fig_tile_order(name):
    s = Svg(name, 720, 380, "Launch order of output tiles: row-major vs grouped (group_m = 4)")
    tm = tn = 8
    wave = 12
    cell = 30
    for idx, (group, title) in enumerate([(0, "row-major"), (4, "grouped, group_m = 4")]):
        x0 = 50 + idx * 350
        y0 = 60
        order = {}
        for pid in range(tm * tn):
            order[_grouped(pid, tm, tn, group)] = pid
        rows_used = {r for (r, c), p in order.items() if p < wave}
        cols_used = {c for (r, c), p in order.items() if p < wave}
        for r in range(tm):
            for c in range(tn):
                p = order[(r, c)]
                fill = "f-c2" if p < wave else "fig-paper"
                s.rect(x0 + c * cell, y0 + r * cell, cell, cell, fill=fill, stroke="s-line", sw=0.6)
                s.text(x0 + c * cell + cell / 2, y0 + r * cell + cell / 2, str(p), size="small", mono=True,
                       role="ink" if p < wave else "muted")
        s.rect(x0, y0, tn * cell, tm * cell, fill="fig-none", stroke="s-ink", sw=1.2)
        # A panels (rows) on the left, B panels (cols) on top
        for r in range(tm):
            s.rect(x0 - 14, y0 + r * cell + 2, 9, cell - 4, fill="f-a2" if r in rows_used else "fig-paper",
                   stroke="s-a", sw=0.6)
        for c in range(tn):
            s.rect(x0 + c * cell + 2, y0 - 14, cell - 4, 9, fill="f-b2" if c in cols_used else "fig-paper",
                   stroke="s-b", sw=0.6)
        s.text(x0 + tn * cell / 2, y0 - 32, title, bold=True, size="small", plain=True)
        s.text(x0 + tn * cell / 2, y0 + tm * cell + 20,
               f"first {wave} tiles: {len(rows_used)} A panels + {len(cols_used)} B panels", size="small")
    s.text(360, 348, "Green: the first wave of 12 blocks. A panel = a 128-row strip of A (blue) or a 128-column",
           size="small", role="muted")
    s.text(360, 366, "strip of B (orange). Fewer distinct panels per wave → more L2 hits, less DRAM traffic.",
           size="small", role="muted")
    return s


# ----------------------------------------------------------------------------------------
# 06: split-K and Stream-K
# ----------------------------------------------------------------------------------------
def fig_split_k(name):
    s = Svg(name, 720, 310, "Split-K: one output tile computed by S blocks over disjoint K ranges")
    ax, ay = 30, 110
    kw, th = 320, 50
    S = 4
    roles = ["a", "b", "c", "d"]
    s.rect(ax, ay, kw, th, fill="fig-paper", stroke="s-ink", sw=1.2)
    for z in range(S):
        s.rect(ax + z * kw / S, ay, kw / S, th, fill=f"f-{roles[z]}", stroke="s-line", sw=0.8)
        s.text(ax + z * kw / S + kw / S / 2, ay + th / 2, f"K_{z}", size="small")
    s.text(ax + kw / 2, ay - 14, "A row panel (128 × K), split into S = 4 ranges", size="small")
    s.brace_h(ax, ax + kw, ay + th + 12, "K", size="small")
    # partial tiles: a stack of cards, each label in the strip that stays visible
    px, py, step, card = 395, 40, 26, 100
    for z in range(S):
        s.box(px + z * step, py + z * step, card, card, "", role=roles[z])
        s.text(px + z * step + 8, py + z * step + 13, f"z = {z}", anchor="start", size="small")
    s.text(px + 90, py + 3 * step + card + 18, "S partial 128 × 128 tiles", size="small")
    s.arrow(ax + kw + 10, ay + th / 2, px - 10, py + 70, role="muted")
    s.arrow(px + 3 * step + card + 6, py + 90, px + 3 * step + card + 40, py + 90, role="ink")
    s.box(px + 3 * step + card + 44, py + 60, 80, 60, "C tile", role="c", fill="f-c2", size="small")
    s.text(360, 268, "grid = (tiles_n, tiles_m, S): S times more blocks for the same tiles; each does K / S",
           size="small")
    s.text(360, 288, "of the work, and the S partial tiles are summed (atomicAdd, or a workspace + a reduction).",
           size="small")
    return s


def fig_wave_quantization(name):
    s = Svg(name, 720, 322, "9 equal tiles on 4 SMs: data-parallel vs Stream-K")
    x0, unit, rh = 120, 40, 26
    roles = ["a", "b", "c", "d", "hl", "a", "b", "c", "d"]
    # data parallel: tile t on SM t%4 in wave t//4, each tile takes 4 units
    s.text(x0 + 6 * unit, 22, "data-parallel: one block per tile", size="small", bold=True)
    for sm in range(4):
        y = 36 + sm * rh
        s.text(x0 - 10, y + rh / 2 - 2, f"SM {sm}", anchor="end", size="small")
        for t in range(9):
            if t % 4 == sm:
                w = t // 4
                s.box(x0 + w * 4 * unit, y, 4 * unit - 2, rh - 4, f"tile {t}", role=roles[t], size="small", rx=2)
    s.line(x0 + 12 * unit, 30, x0 + 12 * unit, 36 + 4 * rh, stroke="s-hl", sw=1.4, dash="4 3")
    s.text(x0 + 12 * unit + 4, 30, "12 units", anchor="start", size="small", role="hl")
    # stream-k: 36 iteration units split 9 per SM
    y1 = 36 + 4 * rh + 40
    s.text(x0 + 6 * unit, y1 - 14, "Stream-K: 36 MAC iterations, 9 per SM", size="small", bold=True)
    for sm in range(4):
        y = y1 + sm * rh
        s.text(x0 - 10, y + rh / 2 - 2, f"SM {sm}", anchor="end", size="small")
        start = sm * 9
        it = start
        while it < start + 9:
            t = it // 4
            seg_end = min(start + 9, (t + 1) * 4)
            owner = seg_end == (t + 1) * 4
            s.box(x0 + (it - start) * unit, y, (seg_end - it) * unit - 2, rh - 4, f"t{t}", role=roles[t],
                  size="small", rx=2, dash=None if owner else "3 2")
            it = seg_end
    s.line(x0 + 9 * unit, y1 - 4, x0 + 9 * unit, y1 + 4 * rh, stroke="s-c", sw=1.4, dash="4 3")
    s.text(x0 + 9 * unit + 4, y1 + 2 * rh, "9 units (+ fix-up)", anchor="start", size="small", role="c")
    s.text(x0, y1 + 4 * rh + 14, "dashed: a partial tile, passed on through the workspace",
           anchor="start", size="small", role="muted")
    return s


def fig_stream_k_ranges(name):
    s = Svg(name, 720, 250, "Stream-K ranges, contributors and owners")
    x0, unit = 40, 16
    iters, per_tile, blocks = 40, 8, 3
    roles = ["a", "b", "c", "d", "hl"]
    y = 50
    s.text(x0, y - 20, "MAC iterations of the whole GEMM (5 tiles × 8 K-slices)", anchor="start", size="small",
           bold=True)
    for t in range(iters // per_tile):
        s.box(x0 + t * per_tile * unit, y, per_tile * unit, 28, f"tile {t}", role=roles[t], size="small", rx=0)
    bounds = [g * iters // blocks for g in range(blocks + 1)]
    y2 = 110
    for g in range(blocks):
        a, b = bounds[g], bounds[g + 1]
        s.box(x0 + a * unit + 1, y2, (b - a) * unit - 2, 26, f"block {g}: iterations {a}–{b - 1}", role="ink",
              size="small")
        s.line(x0 + a * unit, y + 30, x0 + a * unit, y2 + 30, stroke="s-ink", sw=1, dash="2 2")
    s.line(x0 + iters * unit, y + 30, x0 + iters * unit, y2 + 30, stroke="s-ink", sw=1, dash="2 2")
    notes = [
        "block 0: tile 0 whole → store; tile 1 up to iteration 12 → contributor (workspace[0], flag[0]).",
        "block 1: finishes tile 1 → owner: waits for flag[0], adds workspace[0], stores; tile 2 whole → store;",
        "            tile 3 up to iteration 25 → contributor.   block 2: owner of tile 3, then tile 4 whole.",
    ]
    for i, t in enumerate(notes):
        s.text(x0, 170 + i * 22, t, anchor="start", size="small")
    return s


# ----------------------------------------------------------------------------------------
# 07: tensor cores
# ----------------------------------------------------------------------------------------
def fig_mma_layout(name):
    s = Svg(name, 740, 430, "Register layout of mma.sync.m16n8k16 (FP16 A/B, FP32 C/D)")
    cell = 19
    lane_hl = 5  # g = 1, t = 1
    g, t = lane_hl // 4, lane_hl % 4
    y0 = 66
    # A 16 x 16: quadrants = registers a0..a3
    ax = 24
    shades = ["f-a", "f-a2", "f-b", "f-b2"]
    for r in range(16):
        for c in range(16):
            reg = (r >= 8) + 2 * (c >= 8)
            s.rect(ax + c * cell, y0 + r * cell, cell, cell, fill=shades[reg], stroke="s-line", sw=0.4)
    for q, (qx, qy) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        # lane 5's pair of halves in this register: row g (+8), columns 2t, 2t+1 (+8)
        rx, ry = ax + (8 * qx + 2 * t) * cell, y0 + (8 * qy + g) * cell
        s.rect(rx, ry, 2 * cell, cell, fill="f-hl2", stroke="s-hl", sw=1.2)
        s.text(rx + cell, ry + cell / 2, f"a{q}", size="small", mono=True)
        s.text(ax + qx * 8 * cell + 4 * cell, y0 + qy * 8 * cell + 5.5 * cell, f"register a{q}", size="small",
               plate=True)
    s.rect(ax, y0, 16 * cell, 16 * cell, fill="fig-none", stroke="s-ink", sw=1.2)
    s.line(ax + 8 * cell, y0, ax + 8 * cell, y0 + 16 * cell, stroke="s-ink", sw=1)
    s.line(ax, y0 + 8 * cell, ax + 16 * cell, y0 + 8 * cell, stroke="s-ink", sw=1)
    s.text(ax + 8 * cell, y0 - 36, "A: 16 × 16 (m × k)", size="small", bold=True)
    s.text(ax + 8 * cell, y0 - 16, "k →", size="small", role="muted")
    # B 16 x 8
    bx = 380
    for r in range(16):
        for c in range(8):
            s.rect(bx + c * cell, y0 + r * cell, cell, cell, fill="f-b" if r >= 8 else "f-b2", stroke="s-line", sw=0.4)
    for q in range(2):
        rx, ry = bx + g * cell, y0 + (8 * q + 2 * t) * cell
        s.rect(rx, ry, cell, 2 * cell, fill="f-hl2", stroke="s-hl", sw=1.2)
        s.text(rx + cell / 2, ry + cell, f"b{q}", size="tiny", mono=True)
    s.rect(bx, y0, 8 * cell, 16 * cell, fill="fig-none", stroke="s-ink", sw=1.2)
    s.text(bx + 4 * cell, y0 - 36, "B: 16 × 8 (k × n)", size="small", bold=True)
    s.text(bx + 4 * cell, y0 - 16, "n →", size="small", role="muted")
    # C 16 x 8
    cx = 556
    for r in range(16):
        for c in range(8):
            s.rect(cx + c * cell, y0 + r * cell, cell, cell, fill="f-c", stroke="s-line", sw=0.4)
    for h in range(2):
        for j in range(2):
            rx, ry = cx + (2 * t + j) * cell, y0 + (g + 8 * h) * cell
            s.rect(rx, ry, cell, cell, fill="f-hl2", stroke="s-hl", sw=1.2)
            s.text(rx + cell / 2, ry + cell / 2, f"d{2 * h + j}", size="tiny", mono=True)
    s.rect(cx, y0, 8 * cell, 16 * cell, fill="fig-none", stroke="s-ink", sw=1.2)
    s.text(cx + 4 * cell, y0 - 36, "C / D: 16 × 8, FP32", size="small", bold=True)
    s.text(cx + 4 * cell, y0 - 16, "n →", size="small", role="muted")
    s.text(370, 390, f"Red: what lane {lane_hl} holds (g = lane / 4 = {g}, t = lane % 4 = {t}).", size="small")
    s.text(370, 410, "Each 32-bit a/b register packs two FP16 values, the lower index in the low half.",
           size="small")
    return s


def fig_ldmatrix(name):
    s = Svg(name, 720, 380, "ldmatrix.x4: 32 row addresses in, four 8 × 8 fragments out")
    cell = 18
    x0, y0 = 60, 60
    roles = ["a", "b", "c", "d"]
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (col block, row block) for A m16k16: q=0 rows0-7 k0-7 ...
    for q, (kb, rb) in enumerate(pos):
        for r in range(8):
            for c in range(8):
                s.rect(x0 + (kb * 8 + c) * cell, y0 + (rb * 8 + r) * cell, cell, cell, fill=f"f-{roles[q]}",
                       stroke="s-line", sw=0.3)
        s.text(x0 + (kb * 8 + 4) * cell, y0 + (rb * 8 + 4) * cell, f"matrix {q}", size="small", bold=True,
               plate=True)
        for r in range(8):
            lane = 8 * q + r
            if kb == 0:
                s.text(x0 - 6, y0 + (rb * 8 + r) * cell + cell / 2, f"{lane}", anchor="end", size="small",
                       mono=True, role=roles[q])
            else:
                s.text(x0 + 16 * cell + 6, y0 + (rb * 8 + r) * cell + cell / 2, f"{lane}", anchor="start",
                       size="small", mono=True, role=roles[q])
    s.rect(x0, y0, 16 * cell, 16 * cell, fill="fig-none", stroke="s-ink", sw=1.2)
    s.text(x0 + 8 * cell, y0 - 30, "16 × 16 A tile in shared memory", size="small", bold=True)
    s.text(x0 + 8 * cell, y0 - 14, "number = lane giving the row address", size="small", role="muted")
    notes = ["Lanes 8q … 8q+7 point at the 8 rows",
             "(16 bytes each) of matrix q. Register q of",
             "lane l receives row l / 4, halves 2(l % 4)",
             "and 2(l % 4) + 1 of matrix q.",
             "",
             "In this order the four registers are exactly",
             "a0–a3 of mma.m16n8k16: one instruction",
             "loads a whole A fragment.",
             "",
             ".trans delivers the transpose: on B stored",
             "k-major, one x4 gives b0, b1 of two n8 tiles."]
    for i, t in enumerate(notes):
        if t:
            s.text(405, 80 + i * 22, t, anchor="start", size="small")
    return s


def fig_smem_swizzle(name):
    s = Svg(name, 720, 330, "XOR swizzle of 16-byte chunks for the B slice (32 × 128 halves)")
    cw, rh = 30, 22
    for idx, (title, swz) in enumerate([("plain: chunk c stays at c", False), ("swizzled: chunk c → c XOR (row % 8)", True)]):
        x0 = 70 + idx * 340
        y0 = 70
        s.text(x0 + 4 * cw, y0 - 36, title, size="small", bold=True)
        for c in range(8):
            s.text(x0 + c * cw + cw / 2, y0 - 12, f"g{c}", size="small", role="muted")
        for r in range(8):
            s.text(x0 - 8, y0 + r * rh + rh / 2, f"row {r}", anchor="end", size="small")
            for pc in range(8):
                s.rect(x0 + pc * cw, y0 + r * rh, cw, rh, fill="fig-paper", stroke="s-line", sw=0.5)
            logical = 0  # the chunk column ldmatrix reads
            phys = logical ^ r if swz else logical
            s.rect(x0 + phys * cw, y0 + r * rh, cw, rh, fill="f-c2" if swz else "f-hl2", stroke="s-ink", sw=0.8)
            s.text(x0 + phys * cw + cw / 2, y0 + r * rh + rh / 2, "c0", size="small", mono=True)
        verdict = "8 rows in 8 bank groups: 1 wavefront" if swz else "8 rows in bank group 0: 8-way conflict"
        s.text(x0 + 4 * cw, y0 + 8 * rh + 20, verdict, size="small", role="c" if swz else "hl")
    s.text(360, 296, "g0–g7: the 4-bank groups of a 128-byte line; a 256-byte row starts in g0.", size="small",
           role="muted")
    s.text(360, 314, "Shown: where chunk 0 of rows 0–7 lives, the 8 rows one ldmatrix matrix reads.", size="small",
           role="muted")
    return s


def fig_hopper(name):
    s = Svg(name, 720, 270, "Hopper: TMA producer, wgmma consumers, mbarrier handshakes")
    s.box(30, 40, 150, 50, "TMA\n(1 thread issues)", role="b", size="small")
    stages = 4
    for i in range(stages):
        s.box(226 + i * 94, 40, 86, 50, f"stage {i}\nA + B tiles", role="c" if i != 1 else "a", size="small")
    s.arrow(180, 65, 224, 65, role="b")
    s.box(240, 150, 310, 50, "warpgroup = 4 warps: wgmma.mma_async\n"
          "B (and A) read straight from shared memory", role="d", size="small")
    for i in range(stages):
        s.arrow(269 + i * 94, 90, 300 + i * 50, 148, role="d", sw=1)
    s.box(590, 150, 110, 50, "accumulators\nin registers", role="d", fill="f-d2", size="small")
    s.arrow(550, 175, 588, 175, role="d")
    s.text(430, 20, "per stage: a \"full\" mbarrier (TMA bytes arrived) and an \"empty\" one (stage consumed)",
           size="small", role="muted")
    s.text(360, 235, "Producer and consumers sync per stage with mbarriers (transaction counts), not __syncthreads();",
           size="small")
    s.text(360, 255, "the MMA is asynchronous and wide (m64 nN k16 per warpgroup), so issue slots stay free.",
           size="small")
    return s
