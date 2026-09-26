"""Figures for tutorials/00-getting-started.md."""
import math

from .svg import Svg


def fig_roofline(name):
    s = Svg(name, 720, 400, "Roofline of an A100 (FP32 and FP16 tensor cores) with kernels from this site")
    x0, x1, y0, y1 = 90, 690, 350, 30          # plot box (y0 bottom)
    imin, imax = 1 / 16, 1024                  # flop/byte
    pmin, pmax = 0.05, 2000                    # TFLOP/s
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
    for p in (0.1, 1, 10, 100, 1000):
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
    s.text(x1 - 6, py(f16) - 12, "FP16 tensor-core peak ≈ 312 TFLOP/s", anchor="end", size="small", role="d")
    s.text(px(imin) + 12, py(60), "memory roof: I · β  (β ≈ 1.55 TB/s)", anchor="start", size="small", role="a")
    s.text(px(imin) + 12, py(35), "(the sloped blue line)", anchor="start", size="small", role="a")
    s.line(px(ridge32), py(f32), px(ridge32), y0, stroke="s-a", sw=0.8, dash="3 3")
    s.text(px(ridge32) + 4, y0 - 10, "I* ≈ 13", anchor="start", size="small", role="a")

    # kernels (intensity as seen from DRAM)
    pts = [(1 / 12, "", "hl"), (1 / 8, "vector add (1/12), y = 2x (1/8)", "hl"),
           (1 / 4, "naive matmul (no reuse)", "b"), (8, "32 × 32 tiling", "b"), (32, "128 × 128 tile", "c")]
    for i, label, role in pts:
        p = min(beta * i, f32)  # the bound the intensity implies
        s.circle(px(i), py(p), 4.5, fill=f"k-{role}")
        if label and i == 8:   # next to the ridge line: the plate hides the dashed line
            s.text(px(i) + 10, py(p) + 16, label, anchor="start", size="small", role=role, plate=True)
        elif label:
            s.text(px(i) + 10, py(p) + 14, label, anchor="start", size="small", role=role)
    s.text(px(1 / 12) + 2, y1 + 16, "memory-bound", anchor="start", size="small", role="muted")
    s.text(px(250), py(60), "compute-bound", size="small", role="muted")
    return s


def fig_nvcc(name):
    s = Svg(name, 720, 248, "What nvcc does with a .cu file")
    s.box(15, 90, 100, 44, "hello.cu", role="ink", mono=True)
    s.box(155, 30, 130, 40, "host code", role="muted", size="small")
    s.box(155, 150, 130, 40, "device code", role="c", size="small")
    s.arrow(115, 105, 153, 55)
    s.arrow(115, 120, 153, 168)
    s.box(325, 30, 170, 40, "g++ / clang → .o", role="muted", size="small")
    s.box(325, 150, 170, 40, "PTX (compute_XX)", role="a", size="small")
    s.box(535, 150, 170, 40, "ptxas → SASS (sm_XX)", role="b", size="small")
    s.arrow(285, 50, 323, 50)
    s.arrow(285, 170, 323, 170)
    s.arrow(495, 170, 533, 170)
    s.box(535, 30, 170, 40, "executable", role="ink", size="small", bold=True)
    s.arrow(495, 50, 533, 50)
    s.path("M640,150 L640,72", stroke="s-b", arrow="b")
    s.path("M410,150 L410,112 L600,112 L600,72", stroke="s-a", arrow="a", dash="4 3")
    s.text(505, 98, "fatbin: SASS + PTX", size="small", role="muted", plate=True)
    s.text(360, 214, "PTX is portable (the driver JIT-compiles it for newer GPUs);", size="small", role="muted")
    s.text(360, 232, "SASS runs on exactly one architecture.", size="small", role="muted")
    return s
