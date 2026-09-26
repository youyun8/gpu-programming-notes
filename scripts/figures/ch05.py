"""Figures for tutorials/05-amd-cdna3-mfma.md."""
from .svg import Svg


def fig_mfma_layout(name):
    s = Svg(name, 720, 390, "Operand layout of v_mfma_f32_16x16x16_bf16 across the 64 lanes of a wave")
    cell = 15
    roles = ["f-a", "f-b", "f-c", "f-d"]
    # A: rows x k, lane = row + 16 * (k // 4)
    ax, ay = 40, 70
    for r in range(16):
        for k in range(16):
            s.rect(ax + k * cell, ay + r * cell, cell, cell, fill=roles[k // 4], stroke="s-line", sw=0.3)
        for g in range(4):
            s.text(ax + (4 * g + 2) * cell, ay + r * cell + cell / 2, str(r + 16 * g), size="tiny", mono=True)
    s.rect(ax, ay, 16 * cell, 16 * cell, fill="fig-none", stroke="s-ink", sw=1.2)
    s.text(ax + 8 * cell, ay - 36, "A (16 × 16)", size="small", bold=True)
    s.text(ax + 8 * cell, ay - 20, "number = lane holding 4 k values", size="small", role="muted")
    s.text(ax + 8 * cell, ay - 6, "k →", size="small", role="muted")
    # B^T: cols x k
    bx = 290
    for c in range(16):
        for k in range(16):
            s.rect(bx + k * cell, ay + c * cell, cell, cell, fill=roles[k // 4], stroke="s-line", sw=0.3)
        for g in range(4):
            s.text(bx + (4 * g + 2) * cell, ay + c * cell + cell / 2, str(c + 16 * g), size="tiny", mono=True)
    s.rect(bx, ay, 16 * cell, 16 * cell, fill="fig-none", stroke="s-ink", sw=1.2)
    s.text(bx + 8 * cell, ay - 36, "Bᵀ (row j = column j of B)", size="small", bold=True)
    s.text(bx + 8 * cell, ay - 20, "same layout as A", size="small", role="muted")
    s.text(bx + 8 * cell, ay - 6, "k →", size="small", role="muted")
    # D: rows x cols, lane = col + 16 * (row // 4)
    dx = 540
    dc = 7.5
    for r in range(16):
        for c in range(16):
            s.rect(dx + c * dc, ay + r * cell, dc, cell, fill=roles[r // 4], stroke="s-line", sw=0.3)
    s.rect(dx, ay, 16 * dc, 16 * cell, fill="fig-none", stroke="s-ink", sw=1.2)
    s.text(dx + 8 * dc, ay - 36, "D (16 × 16, fp32)", size="small", bold=True)
    s.text(dx + 8 * dc, ay - 20, "column l % 16", size="small", role="muted")
    for g in range(4):
        s.text(dx + 16 * dc + 6, ay + (4 * g + 2) * cell, f"l {16 * g}–{16 * g + 15}", anchor="start",
               size="tiny", role="muted")
    s.text(360, 340, "Lane l supplies A[l % 16][4(l/16) … 4(l/16)+3] and B[4(l/16) … +3][l % 16]: 4 consecutive k",
           size="small")
    s.text(360, 358, "per lane = one 8-byte read when A and B are both K-contiguous (the TN layout).", size="small")
    return s


def fig_mi300x(name):
    s = Svg(name, 720, 300, "MI300X: 8 XCDs with private L2s; workgroups are dealt round-robin")
    for x in range(8):
        bx = 20 + x * 86
        s.rect(bx, 40, 80, 150, fill="fig-panel", stroke="s-ink", sw=1, rx=4)
        s.text(bx + 40, 54, f"XCD {x}", size="small", bold=True)
        for c in range(38):
            cx = bx + 6 + (c % 6) * 11.5
            cy = 66 + (c // 6) * 11
            s.rect(cx, cy, 9, 8, fill="f-c", stroke="s-c", sw=0.4, rx=1)
        s.box(bx + 6, 148, 68, 32, "L2 4 MiB", role="a", size="small")
        s.text(bx + 40, 26, f"WG {x}, {x + 8}, …", size="tiny", role="b")
    s.box(20, 205, 680, 30, "Infinity Cache 256 MiB (shared)", role="d", size="small")
    s.box(20, 245, 680, 26, "HBM3 192 GB, ~5.3 TB/s", role="b", size="small")
    s.text(360, 290, "38 CUs per XCD, 304 in total. Workgroup i runs on XCD i mod 8, so neighbouring tiles land on "
           "different L2s.", size="small", role="muted")
    return s
