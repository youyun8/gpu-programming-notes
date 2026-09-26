"""Figures for tutorials/12-convolution-stencils.md."""
from .svg import Svg


def fig_halo_1d(name):
    s = Svg(name, 720, 250, "1-D convolution: a block's input tile is its output tile plus a halo of R per side")
    x0, cw = 60, 20
    R, T = 3, 24
    s.text(x0 - 10, 64, "in", anchor="end", size="small", bold=True)
    for i in range(T + 2 * R):
        halo = i < R or i >= T + R
        s.rect(x0 + i * cw, 50, cw, 28, fill="f-b2" if halo else "f-a", stroke="s-line", sw=0.6)
    s.brace_h(x0, x0 + R * cw, 40, "R", role="b", up=True)
    s.brace_h(x0 + (T + R) * cw, x0 + (T + 2 * R) * cw, 40, "R", role="b", up=True)
    s.brace_h(x0 + R * cw, x0 + (T + R) * cw, 40, "block's own elements (blockDim.x)", role="a", up=True)
    s.text(x0 - 10, 164, "out", anchor="end", size="small", bold=True)
    for i in range(T):
        s.rect(x0 + (R + i) * cw, 150, cw, 28, fill="f-c2" if i == 10 else "f-c", stroke="s-line", sw=0.6)
    o = R + 10
    s.rect(x0 + (o - R) * cw, 50, (2 * R + 1) * cw, 28, fill="fig-none", stroke="s-hl", sw=2)
    for k in range(-R, R + 1):
        s.line(x0 + (o + k) * cw + cw / 2, 80, x0 + o * cw + cw / 2, 148, stroke="s-hl", sw=0.8)
    s.text(360, 214, "Each output reads 2R + 1 inputs; each input is read by 2R + 1 outputs of the block.",
           size="small")
    s.text(360, 232, "Orange: halo elements, loaded by this block and by its neighbour.", size="small", role="muted")
    return s


def fig_halo_2d(name):
    s = Svg(name, 720, 330, "2-D tiling: a 16 × 16 output tile needs a (16 + 2R)² input tile")
    x0, y0, c = 60, 30, 11
    R, T = 3, 16
    side = T + 2 * R
    for r in range(side):
        for q in range(side):
            halo = r < R or q < R or r >= T + R or q >= T + R
            s.rect(x0 + q * c, y0 + r * c, c, c, fill="f-b2" if halo else "f-a", stroke="s-line", sw=0.3)
    s.rect(x0 + R * c, y0 + R * c, T * c, T * c, fill="fig-none", stroke="s-a", sw=1.8)
    oy, ox = R + 5, R + 9
    s.rect(x0 + (ox - R) * c, y0 + (oy - R) * c, (2 * R + 1) * c, (2 * R + 1) * c, fill="fig-none", stroke="s-hl",
           sw=2)
    s.rect(x0 + ox * c, y0 + oy * c, c, c, fill="f-hl2", stroke="s-hl", sw=1)
    s.text(x0 + side * c / 2, y0 + side * c + 18, "shared tile[(16 + 2R)][(16 + 2R) + 1]", size="small",
           mono=True)
    x = 340
    notes = [("Blue", "inputs at the block's 16 × 16 output positions"),
             ("Orange", "halo: R rows and columns on each side"),
             ("Red", "the (2R + 1)² window of one output"),
             ("", ""),
             ("Loads per block", "(16 + 2R)² = 484 for R = 3, i.e. 1.9 per output"),
             ("Naive", "(2R + 1)² = 49 loads per output, served by L1/L2"),
             ("Filter", "__constant__, one tap for all lanes (broadcast)")]
    for i, (a, b) in enumerate(notes):
        if a:
            s.text(x, 40 + i * 38, a + ":", anchor="start", size="small", bold=True)
            s.text(x, 40 + i * 38 + 17, b, anchor="start", size="small")
    return s


def fig_stencil_25d(name):
    s = Svg(name, 720, 340, "2.5-D blocking: a block marches along z through its column of the domain")
    px, w, h, dx = 150, 240, 44, 60     # plane: parallelogram of width w, depth h, slant dx
    planes = [("z + 1", "register: above", "d", 70), ("z", "shared memory: plane + halo", "c", 140),
              ("z − 1", "register: below", "b", 210)]
    cx = px + dx / 2 + w / 2
    for zlabel, what, role, py in planes:
        s.path(f"M{px},{py + h / 2} L{px + w},{py + h / 2} L{px + w + dx},{py - h / 2} L{px + dx},{py - h / 2} Z",
               stroke=f"s-{role}", fill=f"f-{role}", sw=1.4)
        s.text(px - 10, py, zlabel, anchor="end", size="small", bold=True, role=role)
        s.text(px + w + dx + 16, py, what, anchor="start", size="small")
    s.line(cx, 70, cx, 210, stroke="s-hl", sw=1.2, dash="4 3")
    for py in (70, 210):
        s.circle(cx, py, 5, fill="k-hl")
    s.circle(cx, 140, 5, fill="k-hl")
    for ddx, ddy in ((-28, 0), (28, 0), (-10, -12), (10, 12)):
        s.circle(cx + ddx, 140 + ddy, 4, fill="k-c")
    s.text(360, 262, "Per step: load plane z + 1 once (one value per thread), put plane z in shared memory,",
           size="small")
    s.text(360, 280, "read the 4 in-plane neighbours (green) from it and the 2 out-of-plane ones from",
           size="small")
    s.text(360, 298, "registers, then shift below ← current ← above. Global loads per point: ~1 instead of 7.",
           size="small")
    return s
