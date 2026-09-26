"""Figures for tutorials/13-softmax-attention.md."""
from .svg import Svg


def fig_passes(name):
    s = Svg(name, 720, 260, "Memory passes over one row: three-pass vs online softmax")
    x0, unit = 250, 70
    rows = [("three-pass", ["read: max", "read: sum exp", "read + write"], ["a", "a", "c"]),
            ("online (two-pass)", ["read: max and sum", "read + write"], ["d", "c"]),
            ("online, row in registers", ["read once, write once"], ["c"])]
    for r, (label, parts, roles) in enumerate(rows):
        y = 40 + r * 64
        s.text(x0 - 14, y + 18, label, anchor="end", size="small", bold=True)
        x = x0
        for part, role in zip(parts, roles):
            w = 2.6 * unit if part.startswith("read once") else (
                2.2 * unit if part.startswith(("read + write", "read: max and")) else 1.9 * unit)
            s.box(x, y, w - 8, 36, part, role=role, size="small")
            x += w
    s.text(360, 222, "Rows that fit in L1/L2 make the re-reads cheap; long rows and big batches make them",
           size="small", role="muted")
    s.text(360, 240, "DRAM traffic. Keeping (max, sum) as one state removes a whole pass.", size="small",
           role="muted")
    return s


def fig_online(name):
    s = Svg(name, 720, 300, "Online softmax: when a larger maximum arrives, rescale the running sum")
    x0, cw = 70, 70
    xs = [1.0, 3.0, 2.0, 5.0, 4.0, 0.5]
    m, z = float("-inf"), 0.0
    import math
    for i, v in enumerate(xs):
        s.box(x0 + i * cw * 1.5, 40, cw, 32, f"x = {v:g}", role="a", size="small")
        if v > m:
            z = z * (math.exp(m - v) if m != float("-inf") else 0.0) + 1.0
            m = v
            role = "hl"
            note = "new max: z·e^{m−x} + 1"
        else:
            z += math.exp(v - m)
            role = "c"
            note = "z += e^{x−m}"
        s.box(x0 + i * cw * 1.5 - 10, 110, cw + 20, 48, f"m = {m:g}\nz = {z:.3f}", role=role, size="small")
        s.arrow(x0 + i * cw * 1.5 + cw / 2, 72, x0 + i * cw * 1.5 + cw / 2, 108, role=role, sw=1)
        if i:
            s.arrow(x0 + (i - 1) * cw * 1.5 + cw + 10, 134, x0 + i * cw * 1.5 - 12, 134, role="muted", sw=1)
    s.text(360, 196, "Red: the maximum grew, so the old sum is multiplied by exp(m_old − m_new)", size="small",
           plain=True)
    s.text(360, 216, "before adding 1. Green: add exp(x − m). After the last element,", size="small", plain=True)
    s.text(360, 236, "softmax(x) = exp(x − m) / z; two partial states merge with the same rule.", size="small",
           plain=True)
    s.text(360, 268, f"Here: m = {m:g}, z = {z:.3f}.", size="small", role="muted")
    return s


def fig_flash(name):
    s = Svg(name, 720, 360, "FlashAttention: stream K/V tiles past a block of queries; S never leaves the chip")
    # Q block
    qx, qy = 30, 110
    s.rect(qx, 40, 60, 240, fill="fig-panel", stroke="s-line", sw=1)
    s.rect(qx, qy, 60, 40, fill="f-a2", stroke="s-a", sw=1.6)
    s.text(qx + 30, 26, "Q (N × d)", size="small", bold=True)
    s.text(qx + 30, qy + 20, "block", size="small")
    # K^T tiles
    kx, ky = 130, 40
    s.text(kx + 150, 26, "Kᵀ (d × N): tiles of 32 keys", size="small", bold=True)
    for t in range(5):
        s.rect(kx + t * 60, ky, 56, 50, fill="f-b2" if t == 2 else "f-b", stroke="s-b", sw=1.2)
        s.text(kx + t * 60 + 28, ky + 25, f"K{t}", size="small")
    # S tile
    s.rect(kx + 2 * 60, qy, 56, 40, fill="f-hl2", stroke="s-hl", sw=1.6)
    s.text(kx + 2 * 60 + 28, qy + 20, "S", size="small", bold=True)
    s.line(kx + 2 * 60 + 28, ky + 50, kx + 2 * 60 + 28, qy, stroke="s-b", sw=1, dash="3 3")
    s.line(qx + 60, qy + 20, kx + 2 * 60, qy + 20, stroke="s-a", sw=1, dash="3 3")
    s.text(kx + 150, qy + 60, "S = Q Kᵀ tile (registers)", size="small")
    s.text(kx + 150, qy + 78, "→ online softmax → P", size="small")
    # V tiles
    vx, vy = 130, 210
    s.text(kx + 150, vy + 86, "V (N × d): the matching tile of 32 rows", size="small", bold=True)
    for t in range(5):
        s.rect(vx + t * 60, vy, 56, 60, fill="f-d2" if t == 2 else "f-d", stroke="s-d", sw=1.2)
        s.text(vx + t * 60 + 28, vy + 30, f"V{t}", size="small")
    # output
    ox = 520
    s.rect(ox, qy, 60, 40, fill="f-c2", stroke="s-c", sw=1.6)
    s.text(ox + 30, 26, "O", size="small", bold=True)
    s.text(ox + 30, qy + 20, "acc", size="small")
    s.arrow(kx + 2 * 60 + 58, qy + 20, ox - 4, qy + 20, role="c", sw=1.2)
    s.text(ox + 30, qy + 64, "acc ← acc · exp(m_old − m_new)", size="small", plain=True)
    s.text(ox + 30, qy + 82, "+ P V_t", size="small")
    s.text(ox + 30, qy + 108, "at the end: O = acc / l", size="small")
    s.text(360, 334, "HBM traffic: Q, K, V read (K, V once per query block), O written; no N × N matrix.",
           size="small", role="muted")
    return s
