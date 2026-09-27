"""Figures for tutorials/README.md (the tutorial map)."""
from .svg import Svg


def fig_learning_path(name):
    s = Svg(name, 1214, 500, "Seven learning paths: foundations lead to parallel patterns, matrix multiplication, Triton, AMD, and Kimi K3 cases")
    parts = [
        ("Part I", "Foundations", "a", [("00", "Getting started"), ("01", "Execution model"),
                                        ("02", "Memory hierarchy"), ("03", "Reduction"),
                                        ("09", "Profiling")]),
        ("Part II", "Parallel Patterns", "hl", [("10", "Warp primitives"), ("11", "Scan"),
                                               ("12", "Stencils, conv."), ("13", "Softmax, attention")]),
        ("Part III", "Matrix Multiply", "c", [("1", "Foundations"), ("2–4", "Loads, pipelines"),
                                              ("5–7", "Warps, scheduling"), ("8", "Tensor cores"),
                                              ("9", "Production GEMM")]),
        ("Part IV", "Triton", "d", [("14", "First kernel → prod.")]),
        ("Part V", "AMD", "b", [("05", "CDNA3, MFMA"), ("06", "AITER assembly"),
                                 ("07", "hipBLASLt")]),
        ("Part VI", "Kimi K3 Cases", "hl", [("15", "Triton in SGLang"),
                                             ("16", "FlyDSL in AITER")]),
        ("Part VII", "Publishing", "ink", [("08", "Deploying the site")]),
    ]
    col_w, gap, x0, bh, step = 160, 12, 10, 52, 70
    for i, (part, title, role, chapters) in enumerate(parts):
        x = x0 + i * (col_w + gap)
        s.rect(x, 12, col_w, 448, fill=f"f-{role}", stroke=f"s-{role}", sw=1.2, rx=8)
        s.text(x + col_w / 2, 30, part, size="small", bold=True, role=role)
        s.text(x + col_w / 2, 48, title, size="small", bold=True)
        for j, (num, label) in enumerate(chapters):
            y = 70 + j * step
            s.box(x + 10, y, col_w - 20, bh, f"{num}\n{label}", role=role, fill="fig-paper", size="small")
            if j + 1 < len(chapters):
                s.arrow(x + col_w / 2, y + bh, x + col_w / 2, y + step - 2, role=role, sw=1.2)
    notes = {1: ["after Part I"], 2: ["after Part I"], 3: ["standalone"],
             4: ["after Matrix 1, 8"], 5: ["after Parts IV–V"], 6: ["independent"]}
    for i, lines in notes.items():
        x = x0 + i * (col_w + gap) + col_w / 2
        y = 70 + len(parts[i][3]) * step + 4
        for k, ln in enumerate(lines):
            s.text(x, y + k * 19, ln, size="small", role="muted" if k < 2 else "ink", bold=k == 2)
    return s
