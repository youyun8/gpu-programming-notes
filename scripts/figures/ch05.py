"""Figures for tutorials/05-amd-cdna3-mfma.md."""
from .svg import Svg


def fig_mfma_layout(name):
    s = Svg(name, 720, 440, "Operand layout of v_mfma_f32_16x16x16_bf16 across the 64 lanes of a wave")
    ch = 19                     # row height
    gw = 44                     # width of one 4-element k group
    fills = ["f-a", "f-b", "f-c", "f-d"]

    def operand(x0, y0, title, subtitle, row_label):
        s.text(x0 + 2 * gw, y0 - 44, title, size="small", bold=True)
        s.text(x0 + 2 * gw, y0 - 26, subtitle, size="small", role="muted")
        for g in range(4):
            s.text(x0 + g * gw + gw / 2, y0 - 9, f"k {4 * g}–{4 * g + 3}", size="tiny", role="muted")
        for r in range(16):
            for g in range(4):
                s.rect(x0 + g * gw, y0 + r * ch, gw, ch, fill=fills[g], stroke="s-line", sw=0.6)
                s.text(x0 + g * gw + gw / 2, y0 + r * ch + ch / 2, str(r + 16 * g), size="tiny", mono=True)
            if r % 5 == 0 or r == 15:
                s.text(x0 - 6, y0 + r * ch + ch / 2, f"{row_label}{r}", anchor="end", size="tiny", role="muted")
        s.rect(x0, y0, 4 * gw, 16 * ch, fill="fig-none", stroke="s-ink", sw=1.2)

    y0 = 70
    operand(58, y0, "A (16 × 16, bf16)", "cell = lane holding 4 k values", "row ")
    operand(310, y0, "Bᵀ (row j = column j of B)", "same layout, by output column", "col ")
    # D: row groups of 4 rows, each group one row of lanes
    dx, dw = 556, 150
    s.text(dx + dw / 2, y0 - 44, "D (16 × 16, fp32)", size="small", bold=True)
    s.text(dx + dw / 2, y0 - 26, "column = lane % 16", size="small", role="muted")
    for g in range(4):
        s.rect(dx, y0 + 4 * g * ch, dw, 4 * ch, fill=fills[g], stroke="s-line", sw=0.6)
        s.text(dx + dw / 2, y0 + 4 * g * ch + 2 * ch - 9, f"rows {4 * g}–{4 * g + 3}", size="tiny")
        s.text(dx + dw / 2, y0 + 4 * g * ch + 2 * ch + 9, f"lanes {16 * g}–{16 * g + 15}", size="tiny", mono=True)
    s.rect(dx, y0, dw, 16 * ch, fill="fig-none", stroke="s-ink", sw=1.2)
    s.text(360, 392, "Lane l supplies A[l % 16][4(l/16) … 4(l/16)+3] and B[4(l/16) … +3][l % 16]:", size="small")
    s.text(360, 412, "4 consecutive k per lane, one 8-byte read when A and B are K-contiguous (TN layout).",
           size="small")
    return s


def fig_mi300x(name):
    s = Svg(name, 720, 322, "MI300X: 8 XCDs with private L2s; workgroups are dealt round-robin")
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
    s.text(360, 290, "38 CUs per XCD, 304 in total. Workgroup i runs on XCD i mod 8,", size="small", role="muted")
    s.text(360, 308, "so neighbouring tiles land on different L2s.", size="small", role="muted")
    return s
