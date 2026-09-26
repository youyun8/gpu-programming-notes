"""Figures for tutorials/06-aiter-asm-gemm.md."""
from .svg import Svg


def fig_decomposition(name):
    s = Svg(name, 720, 360, "AITER's 128 × 64 tile: four waves split N; A is shared via LDS, B goes straight to AGPRs")
    # C tile: M = 128 rows (tall), N = 64 cols
    cx, cy, sc = 330, 60, 1.8
    roles = ["a", "b", "c", "d"]
    for w in range(4):
        s.rect(cx + w * 16 * sc, cy, 16 * sc, 128 * sc, fill=f"f-{roles[w]}", stroke=f"s-{roles[w]}", sw=1)
        s.text(cx + w * 16 * sc + 8 * sc, cy + 64 * sc, f"wave {w}", size="small", rotate=-90)
    s.rect(cx, cy, 64 * sc, 128 * sc, fill="fig-none", stroke="s-ink", sw=1.4)
    s.text(cx + 32 * sc, cy - 26, "C tile 128 (M) × 64 (N)", size="small", bold=True)
    s.text(cx + 32 * sc, cy - 10, "each wave: 16 (N) × 128 (M)", size="small")
    # A in LDS
    ax = 60
    s.rect(ax, cy, 64 * sc * 0.9, 128 * sc, fill="f-ink", stroke="s-ink", sw=1.2)
    s.text(ax + 29 * sc, cy + 64 * sc - 10, "A slice", size="small", bold=True)
    s.text(ax + 29 * sc, cy + 64 * sc + 8, "128 × 64 (K)", size="small")
    s.text(ax + 29 * sc, cy - 10, "LDS (buffer_load … lds)", size="small", role="muted")
    for w in range(4):
        s.arrow(ax + 64 * sc * 0.9, cy + 40 + w * 45, cx - 4, cy + 40 + w * 45, role=roles[w], sw=1)
    s.text(ax + 29 * sc, cy + 128 * sc + 16, "needed by all 4 waves", size="small", role="muted")
    # B strips into AGPRs
    bx = 520
    for w in range(4):
        y = cy + w * 58
        s.box(bx, y, 170, 44, f"B strip {w}: 16 cols × 64 k\n→ AGPRs of wave {w}", role=roles[w], size="small")
        s.arrow(bx - 4, y + 22, cx + w * 16 * sc + 8 * sc, cy + 128 * sc * 0.15 + w * 10, role=roles[w], sw=0.8,
                dash="4 3")
    s.text(bx + 85, cy + 4 * 58 + 12, "private per wave: no LDS", size="small", role="muted")
    s.text(360, 345, "Pre-shuffled weights make each lane's 16-byte buffer_load exactly its MFMA B fragment.",
           size="small", role="muted")
    return s


def fig_interleave(name):
    s = Svg(name, 720, 250, "Interleaving: memory instructions issue in the shadow of MFMAs")
    x0, unit = 110, 22
    # issue stream
    seq = ["M", "L", "S", "M", "R", "R", "M", "L", "S", "M", "R", "R", "M", "L", "S", "M", "R", "R"]
    roles = {"M": "d", "L": "b", "R": "a", "S": "muted"}
    names = {"M": "mfma", "L": "buffer_load", "R": "ds_read", "S": "s_add m0"}
    s.text(x0 - 10, 52, "issue", anchor="end", size="small", bold=True)
    t = 0
    mf_starts = []
    for op in seq:
        s.rect(x0 + t * unit, 40, unit - 3, 24, fill=f"f-{roles[op]}2" if op == "M" else f"f-{roles[op]}",
               stroke=f"s-{roles[op]}", sw=0.8, rx=2)
        s.text(x0 + t * unit + unit / 2 - 1, 52, op, size="small", mono=True)
        if op == "M":
            mf_starts.append(t)
        t += 1
    # MFMA pipe busy
    s.text(x0 - 10, 102, "matrix core", anchor="end", size="small", bold=True)
    busy_until = 0
    for st in mf_starts:
        start = max(st, busy_until)
        s.rect(x0 + start * unit, 90, 3 * unit - 3, 24, fill="f-d2", stroke="s-d", sw=0.8, rx=2)
        busy_until = start + 3
    s.text(x0 - 10, 149, "legend", anchor="end", size="small", role="muted")
    for i, (k, v) in enumerate(names.items()):
        s.rect(x0 + i * 140, 140, 18, 18, fill=f"f-{roles[k]}2" if k == "M" else f"f-{roles[k]}",
               stroke=f"s-{roles[k]}", sw=0.8, rx=2)
        s.text(x0 + i * 140 + 9, 149, k, size="small", mono=True)
        s.text(x0 + i * 140 + 24, 149, v, anchor="start", size="small", mono=True)
    s.text(360, 196, "An MFMA keeps the matrix core busy for several cycles after it issues. Placing 1–3 loads,",
           size="small")
    s.text(360, 214, "LDS reads or scalar updates between consecutive MFMAs hides their issue cost entirely:",
           size="small")
    s.text(360, 232, "the matrix core never waits. (Illustrative timing, not cycle-accurate.)", size="small",
           role="muted")
    return s


def fig_split_k(name):
    s = Svg(name, 720, 250, "Split-K epilogue with a per-tile semaphore")
    for z in range(3):
        s.box(30, 30 + z * 56, 170, 42, f"workgroup z = {z}\npartial 128 × 64 tile", role=["a", "b", "c"][z],
              size="small")
        s.arrow(200, 51 + z * 56, 330, 110, role=["a", "b", "c"][z], sw=1)
    s.box(330, 85, 150, 50, "C (fp32 or bf16)\nglobal_atomic_add", role="d", size="small")
    s.box(530, 85, 170, 50, "semaphore[tile]\narrival counter", role="hl", size="small")
    s.arrow(480, 110, 528, 110, role="hl", dash="4 3")
    s.text(360, 214, "The last workgroup to arrive does the final phase and resets the counter;", size="small")
    s.text(360, 232, "fp32 atomics make the summation order, and so the last bits, non-deterministic.", size="small",
           role="muted")
    return s
