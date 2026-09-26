"""Figures for tutorials/09-profiling.md."""
import math

from .svg import Svg


def fig_workflow(name):
    s = Svg(name, 720, 330, "The profiling loop: from the whole program down to one kernel, then back")
    steps = [("1. Time it", "CUDA events;\nkeep a baseline", "muted"),
             ("2. Timeline", "Nsight Systems:\nwhere is time?", "a"),
             ("3. One kernel", "Nsight Compute:\nspeed of light", "b"),
             ("4. Diagnose", "limiter, stall,\nsource line", "d"),
             ("5. Fix one thing", "then measure\nagain", "c")]
    w, h, gap, y = 124, 84, 14, 60
    x0 = (720 - (5 * w + 4 * gap)) / 2
    for i, (title, body, role) in enumerate(steps):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, h, fill="fig-paper" if role == "muted" else f"f-{role}", stroke=f"s-{role}", sw=1.4, rx=6)
        s.text(x + w / 2, y + 18, title, size="small", bold=True)
        for j, ln in enumerate(body.split("\n")):
            s.text(x + w / 2, y + 44 + j * 17, ln, size="small")
        if i < 4:
            s.arrow(x + w + 2, y + h / 2, x + w + gap - 2, y + h / 2, role="muted", sw=1.2)
    # loop back
    xl, xr = x0 + w / 2, x0 + 4 * (w + gap) + w / 2
    s.path(f"M{xr},{y + h + 2} L{xr},{y + h + 36} L{xl},{y + h + 36} L{xl},{y + h + 4}", stroke="s-c", sw=1.2,
           arrow="c")
    s.text((xl + xr) / 2, y + h + 50, "repeat until the kernel sits near a roof, or the time is spent elsewhere",
           size="small", role="c")
    notes = [("Timeline questions", "Is the GPU idle? Copies serialised? Launch gaps? Which kernel dominates?"),
             ("Kernel questions", "Memory- or compute-bound? Which stall? Which source line?")]
    for i, (a, b) in enumerate(notes):
        s.text(40, 238 + i * 40, a + ":", anchor="start", size="small", bold=True)
        s.text(40, 256 + i * 40, b, anchor="start", size="small")
    return s


def fig_timeline(name):
    s = Svg(name, 720, 300, "An Nsight Systems timeline (sketch): the GPU rows show where the time goes")
    x0, x1 = 150, 690
    rows = [("CPU thread", 40), ("CUDA API", 80), ("NVTX", 120), ("GPU memcpy", 160), ("GPU kernels", 200)]
    for label, y in rows:
        s.text(x0 - 12, y + 12, label, anchor="end", size="small", bold=True)
        s.line(x0, y + 26, x1, y + 26, stroke="s-line", sw=0.6)
    # CPU work
    s.box(x0, 40, 130, 24, "setup", role="muted", size="small")
    s.box(x0 + 136, 40, 100, 24, "launch loop", role="muted", size="small")
    s.box(x0 + 420, 40, 110, 24, "post-process", role="muted", size="small")
    # API calls
    s.box(x0 + 10, 80, 110, 24, "cudaMemcpy", role="b", size="small")
    for k in range(6):
        s.rect(x0 + 140 + k * 15, 80, 11, 24, fill="f-a", stroke="s-a", sw=1)
    s.box(x0 + 240, 80, 170, 24, "cudaDeviceSynchronize", role="b", size="small")
    # NVTX
    s.box(x0 + 136, 120, 280, 24, "range: transposes", role="d", size="small")
    # memcpy
    s.box(x0 + 10, 160, 110, 24, "HtoD 256 MB", role="b", size="small")
    s.box(x0 + 420, 160, 110, 24, "DtoH 256 MB", role="b", size="small")
    # kernels
    ks = [(150, 90, "naive", "hl"), (244, 70, "shared", "d"), (318, 60, "padded", "c")]
    for x, w, lab, role in ks:
        s.box(x0 + x - 10, 200, w, 24, lab, role=role, size="small")
    s.brace_h(x0 + 10, x0 + 136, 234, role="hl")
    s.text(x0 + 73, 251, "GPU idle: copy + setup", size="small", role="hl")
    s.text(x0 + 330, 251, "kernel widths are what Nsight Compute explains", size="small", role="muted")
    s.text(360, 284, "Read it top-down: an idle GPU row under a busy CPU row is a host-side problem.",
           size="small", role="muted")
    return s


def fig_roofline(name):
    s = Svg(name, 720, 392, "The example kernels on a roofline (A100-class numbers, log-log)")
    x0, y0, w, h = 90, 30, 560, 290           # plot area; y grows downwards

    def px(i):  # arithmetic intensity (flop/B) -> x
        return x0 + (math.log10(i) + 2) / 5 * w   # 0.01 … 1000

    def py(p):  # TFLOP/s -> y
        return y0 + h - (math.log10(p) + 3) / 5 * h  # 0.001 … 100

    s.rect(x0, y0, w, h, fill="fig-none", stroke="s-line", sw=1)
    for e in range(-2, 4):
        s.text(px(10 ** e), y0 + h + 14, f"{10 ** e:g}", size="small", role="muted")
    for e in range(-3, 3):
        s.text(x0 - 8, py(10 ** e), f"{10 ** e:g}", anchor="end", size="small", role="muted")
    s.text(x0 + w / 2, y0 + h + 34, "arithmetic intensity (flop per DRAM byte)", size="small")
    s.text(24, y0 + h / 2, "TFLOP/s (FP32)", size="small", rotate=-90)
    bw, peak = 1.5, 19.5                      # TB/s, TFLOP/s
    ridge = peak / bw
    s.line(px(0.01), py(bw * 0.01), px(ridge), py(peak), stroke="s-b", sw=2)
    s.line(px(ridge), py(peak), px(1000), py(peak), stroke="s-a", sw=2)
    s.text(px(0.15), py(bw * 0.15) - 14, "memory roof: 1.5 TB/s", size="small", role="b", rotate=-math.degrees(math.atan(h / 5 / (w / 5))))
    s.text(px(150), py(peak) - 12, "FP32 roof: 19.5 TFLOP/s", size="small", role="a")
    s.line(px(ridge), py(peak), px(ridge), y0 + h, stroke="s-line", sw=1, dash="4 3")
    s.text(px(ridge) + 6, y0 + h - 14, "ridge 13", anchor="start", size="small", role="muted")
    # copies/transposes: 0 flop; drawn at a nominal intensity with their bandwidth in TB/s
    s.circle(px(0.02), py(1.5 * 0.02 * 0.9), 6, fill="k-c")
    s.text(px(0.02) + 12, py(1.5 * 0.02 * 0.9) + 14, "copies, transposes", anchor="start", size="small",
           role="c")
    s.text(px(0.02) + 12, py(1.5 * 0.02 * 0.9) + 31, "judge them by GB/s,", anchor="start", size="small",
           role="muted")
    s.text(px(0.02) + 12, py(1.5 * 0.02 * 0.9) + 48, "not by TFLOP/s", anchor="start", size="small",
           role="muted")
    s.circle(px(64), py(15.0), 6, fill="k-d")
    s.text(px(64), py(15.0) + 22, "polynomial: 64 flop/B", size="small", role="d")
    s.circle(px(0.5), py(0.02), 6, fill="k-hl")
    s.text(px(0.5) + 12, py(0.02) - 8, "latency-bound:", anchor="start", size="small", role="hl")
    s.text(px(0.5) + 12, py(0.02) + 9, "far below both roofs", anchor="start", size="small", role="hl")
    s.text(360, 376, "The distance from a point to the roof above it is the headroom; the roof says which unit.",
           size="small", role="muted")
    return s


def fig_stalls(name):
    s = Svg(name, 720, 330, "Where warps wait: stall reasons for three kernel types (illustrative)")
    x0, bw = 180, 480
    kernels = [("copyCoalesced", [("long scoreboard", 0.72, "b"), ("selected", 0.06, "c"),
                                  ("other", 0.22, "muted")]),
               ("transposeShared", [("MIO throttle", 0.38, "hl"), ("short sb.", 0.26, "d"),
                                    ("long sb.", 0.2, "b"), ("other", 0.16, "muted")]),
               ("polynomial", [("wait", 0.45, "a"), ("math pipe", 0.18, "d"),
                               ("selected", 0.25, "c"), ("other", 0.12, "muted")])]
    for r, (kname, parts) in enumerate(kernels):
        y = 44 + r * 72
        s.text(x0 - 12, y + 14, kname, anchor="end", size="small", mono=True)
        x = x0
        for label, frac, role in parts:
            w = frac * bw
            fill = "fig-panel" if role == "muted" else f"f-{role}"
            s.rect(x, y, w, 28, fill=fill, stroke="s-muted" if role == "muted" else f"s-{role}", sw=1)
            if w > 60:
                s.text(x + w / 2, y + 14, label, size="small")
            x += w
        if r == 0:
            s.text(x0, y + 44, "share of stall cycles, by reason", anchor="start", size="tiny", role="muted")
    notes = [("long scoreboard (sb.)", "waiting for global/local memory (L1TEX): add bytes in flight or reuse"),
             ("MIO throttle, short sb.", "shared-memory queue full or bank conflicts: pad or swizzle"),
             ("wait", "fixed-latency dependency: more independent work per thread (ILP)")]
    for i, (a, b) in enumerate(notes):
        s.text(40, 262 + i * 22, a + ":", anchor="start", size="small", bold=True)
        s.text(236, 262 + i * 22, b, anchor="start", size="small")
    return s
