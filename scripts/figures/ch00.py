"""Figures for tutorials/00-getting-started.md."""
import math

from .svg import Svg


def fig_roofline(name):
    s = Svg(name, 720, 400, "Roofline of an A100 (FP32 and FP16 tensor cores) with kernels from this site")
    x0, x1, y0, y1 = 90, 690, 350, 30          # plot box (y0 bottom)
    imin, imax = 1 / 16, 1024                  # flop/byte
    pmin, pmax = 0.05, 500                     # TFLOP/s
    beta, f32, f16 = 1.555, 19.5, 312.0        # TB/s, TFLOP/s

    def px(i):
        return x0 + (x1 - x0) * (math.log(i) - math.log(imin)) / (math.log(imax) - math.log(imin))

    def py(p):
        return y0 - (y0 - y1) * (math.log(p) - math.log(pmin)) / (math.log(pmax) - math.log(pmin))

    # grid and axes
    for e in range(-4, 11):
        i = 2.0 ** e
        s.line(px(i), y0, px(i), y1, stroke="s-line", sw=0.4, dash="2 3")
        if e % 2 == 0:
            label = f"1/{int(1 / i)}" if i < 1 else str(int(i))
            s.text(px(i), y0 + 14, label, size="small", role="muted")
    for p in (0.1, 1, 10, 100):
        s.line(x0, py(p), x1, py(p), stroke="s-line", sw=0.4, dash="2 3")
        s.text(x0 - 8, py(p), f"{p:g}", anchor="end", size="small", role="muted")
    s.rect(x0, y1, x1 - x0, y0 - y1, fill="fig-none", stroke="s-ink", sw=1)
    s.text((x0 + x1) / 2, y0 + 34, "arithmetic intensity I (flop / byte, log scale)", size="small")
    s.text(24, (y0 + y1) / 2, "TFLOP/s (log scale)", size="small", rotate=-90)

    # roofs
    ridge32, ridge16 = f32 / beta, f16 / beta
    s.path(f"M{px(imin):.1f},{py(beta * imin):.1f} L{px(ridge16):.1f},{py(f16):.1f} L{x1},{py(f16):.1f}",
           stroke="s-d", sw=1.4, dash="6 4")
    s.path(f"M{px(imin):.1f},{py(beta * imin):.1f} L{px(ridge32):.1f},{py(f32):.1f} L{x1},{py(f32):.1f}",
           stroke="s-a", sw=2.4)
    s.text(x1 - 6, py(f32) - 10, "FP32 peak ≈ 19.5 TFLOP/s", anchor="end", size="small", role="a")
    s.text(x1 - 6, py(f16) - 10, "FP16 tensor-core peak ≈ 312 TFLOP/s", anchor="end", size="small", role="d")
    s.text(px(0.35), py(beta * 0.35) - 16, "memory roof: I · β (β ≈ 1.55 TB/s)", anchor="start", size="small",
           role="a", rotate=-28)
    s.line(px(ridge32), py(f32), px(ridge32), y0, stroke="s-a", sw=0.8, dash="3 3")
    s.text(px(ridge32) + 4, y0 - 10, "I* ≈ 13", anchor="start", size="small", role="a")

    # kernels (intensity as seen from DRAM)
    pts = [(1 / 12, "vector add", "hl", "end"), (1 / 8, "y = 2x", "hl", "start"),
           (1 / 4, "naive matmul (no reuse)", "b", "start"), (8, "32 × 32 tiling", "b", "start"),
           (32, "128 × 128 tile", "c", "start")]
    for i, label, role, anchor in pts:
        p = min(beta * i, f32)  # the bound the intensity implies
        s.circle(px(i), py(p), 4.5, fill=f"k-{role}")
        dy = 12 if anchor == "start" else 30  # "end": below the others
        s.text(px(i) + 8, py(p) + dy, label, anchor="start", size="small", role=role)
    s.text(px(1 / 12) + 2, y1 + 16, "memory-bound", anchor="start", size="small", role="muted")
    s.text(px(200), y1 + 72, "compute-bound", size="small", role="muted")
    return s


def fig_nvcc(name):
    s = Svg(name, 720, 230, "What nvcc does with a .cu file")
    s.box(20, 90, 100, 44, "hello.cu", role="ink", mono=True)
    s.box(170, 30, 140, 40, "host code", role="muted", size="small")
    s.box(170, 150, 140, 40, "device code", role="c", size="small")
    s.arrow(120, 105, 168, 55)
    s.arrow(120, 120, 168, 168)
    s.box(360, 30, 150, 40, "g++ / clang → .o", role="muted", size="small")
    s.box(360, 150, 150, 40, "PTX (compute_XX)", role="a", size="small")
    s.box(560, 150, 140, 40, "ptxas → SASS (sm_XX)", role="b", size="small")
    s.arrow(310, 50, 358, 50)
    s.arrow(310, 170, 358, 170)
    s.arrow(510, 170, 558, 170)
    s.box(560, 30, 140, 40, "executable", role="ink", size="small", bold=True)
    s.arrow(510, 50, 558, 50)
    s.path("M630,150 L630,110 L650,110 L650,72", stroke="s-b", arrow="b")
    s.path("M435,150 L435,110 L610,110 L610,72", stroke="s-a", arrow="a", dash="4 3")
    s.text(530, 100, "fatbin: SASS + PTX", size="small", role="muted")
    s.text(360, 214, "PTX is portable (JIT-compiled by the driver for newer GPUs); SASS runs on exactly one architecture.",
           size="small", role="muted")
    return s
