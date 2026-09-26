"""Figures for tutorials/README.md (the tutorial map)."""
from .svg import Svg


def fig_learning_path(name):
    s = Svg(name, 740, 460, "The tutorials: four parts, read top to bottom")
    parts = [
        ("Part I · CUDA Foundations", "a", [("00", "Getting started"), ("01", "Execution model"),
                                            ("02", "Memory hierarchy"), ("03", "Parallel reduction")]),
        ("Part II · Matrix Multiplication", "c", [("04", "Tiled matmul"), ("04.1–04.3", "Loads, pipelines"),
                                                  ("04.4–04.6", "Warps, order, split-K"),
                                                  ("04.7", "Tensor cores")]),
        ("Part III · AMD GPUs", "b", [("05", "CDNA3 and MFMA"), ("06", "AITER asm GEMM"),
                                      ("07", "hipBLASLt, TensileLite")]),
        ("Part IV · Publishing", "d", [("08", "Deploying this site")]),
    ]
    col_w, gap, x0, y0, bh, step = 170, 12, 16, 60, 58, 76
    for i, (title, role, chapters) in enumerate(parts):
        x = x0 + i * (col_w + gap)
        s.rect(x, 20, col_w, 430, fill=f"f-{role}", stroke=f"s-{role}", sw=1.2, rx=8)
        words = title.split(" · ")
        s.text(x + col_w / 2, 36, words[0], size="small", bold=True, role=role)
        s.text(x + col_w / 2, 54, words[1], size="small", bold=True)
        for j, (num, label) in enumerate(chapters):
            y = y0 + 20 + j * step
            s.box(x + 10, y, col_w - 20, bh, f"{num}\n{label}", role=role, fill="fig-paper", size="small")
            if j + 1 < len(chapters):
                s.arrow(x + col_w / 2, y + bh, x + col_w / 2, y + step - 2, role=role, sw=1.2)
    notes = ["then Part II", "then Part III", "", ""]
    for i, note in enumerate(notes):
        if note:
            s.text(x0 + i * (col_w + gap) + col_w / 2, 420, f"→ {note}", size="small", bold=True,
                   role=parts[i][1])
    s.text(x0 + 3 * (col_w + gap) + col_w / 2, 200, "read any time", size="small", role="muted")
    s.text(x0 + 3 * (col_w + gap) + col_w / 2, 370, "Each chapter has:", size="small", bold=True)
    s.text(x0 + 3 * (col_w + gap) + col_w / 2, 390, "goals, sections,", size="small")
    s.text(x0 + 3 * (col_w + gap) + col_w / 2, 408, "takeaways, exercises", size="small")
    return s
