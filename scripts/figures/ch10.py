"""Figures for tutorials/10-warp-primitives.md (8 lanes shown instead of 32)."""
from .svg import Svg

LANES = 8


def _lanes(s, x0, y, cw, values, fill_fn=None, label=None, role="ink", mono=True):
    if label:
        s.text(x0 - 12, y + 14, label, anchor="end", size="small")
    for i, v in enumerate(values):
        fill = fill_fn(i) if fill_fn else "fig-paper"
        s.rect(x0 + i * cw, y, cw - 6, 28, fill=fill, stroke=f"s-{role}", sw=1, rx=4)
        s.text(x0 + i * cw + (cw - 6) / 2, y + 14, str(v), size="small", mono=mono)


def fig_shuffles(name):
    s = Svg(name, 720, 380, "Where each lane reads from in the four shuffle variants (8 lanes)")
    x0, cw = 250, 56
    vals = [f"v{i}" for i in range(LANES)]
    rows = [("__shfl_sync(m, v, 2)", [2] * LANES),
            ("__shfl_up_sync(m, v, 1)", [max(i - 1, 0) if i >= 1 else i for i in range(LANES)]),
            ("__shfl_down_sync(m, v, 2)", [i + 2 if i + 2 < LANES else i for i in range(LANES)]),
            ("__shfl_xor_sync(m, v, 1)", [i ^ 1 for i in range(LANES)])]
    for i in range(LANES):
        s.text(x0 + i * cw + (cw - 6) / 2, 24, f"lane {i}", size="small", role="muted")
    for r, (label, src) in enumerate(rows):
        y = 44 + r * 80
        _lanes(s, x0, y, cw, vals, fill_fn=lambda i: "f-a", label="", role="a")
        s.text(x0 - 12, y + 14, "before", anchor="end", size="small", role="muted")
        got = [f"v{src[i]}" for i in range(LANES)]
        changed = [src[i] != i for i in range(LANES)]
        _lanes(s, x0, y + 42, cw, got, fill_fn=lambda i: "f-c2" if changed[i] else "fig-panel", role="c")
        s.text(x0 - 12, y + 56, label, anchor="end", size="small", mono=True)
        for i in range(LANES):
            if changed[i]:
                s.arrow(x0 + src[i] * cw + (cw - 6) / 2, y + 29, x0 + i * cw + (cw - 6) / 2, y + 41, role="c",
                        sw=0.9)
    s.text(360, 366, "Grey: the lane keeps its own value (the source lane is out of range).", size="small",
           role="muted")
    return s


def fig_ballot_compaction(name):
    s = Svg(name, 720, 384, "Stream compaction in one warp: ballot, popc, one atomic")
    x0, cw = 200, 60
    vals = ["−1.2", "0.7", "0.3", "−0.5", "−2.0", "1.1", "0.9", "−0.1"]
    keep = [v[0] != "−" for v in vals]
    _lanes(s, x0, 36, cw, vals, fill_fn=lambda i: "f-c" if keep[i] else "fig-paper", label="in[i]", role="a")
    _lanes(s, x0, 88, cw, [int(k) for k in keep], fill_fn=lambda i: "f-c2" if keep[i] else "fig-panel",
           label="keep", role="c")
    votes = "".join("1" if k else "0" for k in reversed(keep))
    s.text(x0 - 12, 142, "__ballot_sync", anchor="end", size="small", mono=True)
    s.text(x0, 142, f"votes = 0b{votes}  (bit l = lane l),  __popc(votes) = {sum(keep)}", anchor="start",
           size="small", mono=True)
    before = [sum(keep[:i]) for i in range(LANES)]
    _lanes(s, x0, 196, cw, before, fill_fn=lambda i: "f-d" if keep[i] else "fig-panel",
           label="popc(lower bits)", role="d")
    s.text(x0 - 12, 168, "lane 0", anchor="end", size="small", mono=True)
    s.text(x0, 168, "base = atomicAdd(count, 4), say 100; __shfl_sync to all", anchor="start",
           size="small", mono=True)
    # output slots
    out_x, ow = x0 + 40, 80
    for j in range(4):
        s.rect(out_x + j * ow, 290, ow - 8, 28, fill="f-c", stroke="s-c", sw=1, rx=4)
        s.text(out_x + j * ow + (ow - 8) / 2, 304, str([v for v, k in zip(vals, keep) if k][j]), size="small",
               mono=True)
        s.text(out_x + j * ow + (ow - 8) / 2, 332, f"out[{100 + j}]", size="small", mono=True, role="muted")
    s.text(x0 - 12, 304, "out", anchor="end", size="small")
    k = 0
    for i in range(LANES):
        if keep[i]:
            s.arrow(x0 + i * cw + (cw - 6) / 2, 225, out_x + k * ow + (ow - 8) / 2, 288, role="c", sw=0.9)
            k += 1
    s.text(360, 362, "Each surviving lane writes to base + (number of surviving lanes before it).",
           size="small", role="muted")
    return s


def fig_match_any(name):
    s = Svg(name, 720, 250, "__match_any_sync groups lanes with equal keys; the lowest lane of each group adds")
    x0, cw = 200, 60
    keys = [7, 3, 7, 7, 12, 3, 7, 5]
    groups = {7: "f-a2", 3: "f-b2", 12: "f-c2", 5: "f-d2"}
    _lanes(s, x0, 40, cw, keys, fill_fn=lambda i: groups[keys[i]], label="key", role="ink")
    masks = []
    for i in range(LANES):
        m = sum(1 << j for j in range(LANES) if keys[j] == keys[i])
        masks.append(m)
    _lanes(s, x0, 96, cw, [f"{bin(m).count('1')}" for m in masks], fill_fn=lambda i: groups[keys[i]],
           label="popc(peers)", role="ink")
    leaders = [i == (masks[i] & -masks[i]).bit_length() - 1 for i in range(LANES)]
    _lanes(s, x0, 152, cw, ["+" + str(bin(masks[i]).count("1")) if leaders[i] else "–" for i in range(LANES)],
           fill_fn=lambda i: groups[keys[i]] if leaders[i] else "fig-panel", label="leader adds", role="ink")
    s.text(360, 216, "4 atomics instead of 8: hist[7] += 4, hist[3] += 2, hist[12] += 1, hist[5] += 1.",
           size="small")
    return s


def fig_cg_tiles(name):
    s = Svg(name, 720, 282, "Cooperative groups: a thread block partitioned into tiles")
    x0, y0, w = 40, 50, 640
    s.rect(x0, y0, w, 44, fill="f-a", stroke="s-a", sw=1.2, rx=6)
    s.text(x0 + w / 2, y0 + 22, "cg::thread_block block = cg::this_thread_block();   // 256 threads", size="small",
           mono=True)
    for t in range(8):
        s.box(x0 + t * w / 8 + 3, y0 + 64, w / 8 - 6, 40, f"warp {t}", role="c", size="small")
    s.text(x0 + w / 2, y0 + 124, "cg::tiled_partition<32>(block): 8 tiles, meta_group_rank() = 0 … 7",
           size="small", mono=True)
    for t in range(16):
        s.box(x0 + t * w / 16 + 2, y0 + 142, w / 16 - 4, 32, f"{t}", role="d", size="small")
    s.text(x0 + w / 2, y0 + 192, "cg::tiled_partition<16>(block): 16 tiles of 16 threads;", size="small",
           mono=True)
    s.text(x0 + w / 2, y0 + 210, "tile.sync(), tile.shfl_*(), cg::reduce() act on 16 lanes", size="small",
           mono=True)
    return s
