"""Figures for tutorials/14-triton.md."""
from .svg import Svg


def fig_model(name):
    s = Svg(name, 720, 330 + 14, "CUDA programs threads; Triton programs blocks and lets the compiler map them to threads")
    # CUDA side
    x0, y0 = 40, 50
    s.text(x0 + 140, 28, "CUDA: one thread, scalar code", size="small", bold=True)
    for i in range(8):
        s.box(x0 + i * 35, y0, 30, 30, f"t{i}", role="a", size="tiny")
    s.text(x0 + 140, y0 + 52, "i = blockIdx.x * blockDim.x + threadIdx.x", size="small", mono=True)
    s.text(x0 + 140, y0 + 72, "if (i < n) out[i] = x[i] + y[i];", size="small", mono=True)
    s.text(x0 + 140, y0 + 100, "you choose: thread ↔ element map,", size="small", role="muted")
    s.text(x0 + 140, y0 + 118, "shared memory, barriers, vector widths", size="small", role="muted")
    # Triton side
    x1 = 366
    s.text(x1 + 169, 28, "Triton: one program, block-level code", size="small", bold=True)
    s.rect(x1, y0, 338, 30, fill="f-c", stroke="s-c", sw=1.4, rx=4)
    s.text(x1 + 169, y0 + 15, "offs = pid * BLOCK + tl.arange(0, BLOCK)", size="small", mono=True)
    s.text(x1 + 169, y0 + 52, "x = tl.load(x_ptr + offs, mask=m)", size="small", mono=True)
    s.text(x1 + 169, y0 + 72, "tl.store(out_ptr + offs, x + y, mask=m)", size="small", mono=True)
    s.text(x1 + 169, y0 + 100, "the compiler chooses: layout of the block", size="small", role="muted")
    s.text(x1 + 169, y0 + 118, "over threads, vector loads, shared memory", size="small", role="muted")
    s.line(353, 36, 353, 186, stroke="s-line", sw=1, dash="4 3")
    # Mapping strip
    y2 = 222
    s.text(360, y2 - 18, "one Triton program of BLOCK = 1024 elements, num_warps = 4 (128 threads)",
           size="small")
    for w in range(4):
        s.box(60 + w * 150, y2, 144, 40, f"warp {w}\nelements {256 * w}–{256 * w + 255}", role="b", size="small")
    s.text(360, y2 + 62, "each thread gets 8 elements, e.g. two 16-byte (float4) chunks: coalesced and vectorised",
           size="small", role="muted")
    s.text(360, y2 + 82, "The program id is the block index; there is no threadIdx in the language.",
           size="small", role="muted")
    return s


def fig_pointer_block(name):
    s = Svg(name, 720, 362, "A 2-D block of pointers: broadcast a column of row offsets against a row of columns")
    x0, y0, c = 300, 76, 38
    rows, cols = 4, 5
    s.text(x0 - 150, y0 - 28, "rows[:, None] * S", size="small", mono=True)
    for r in range(rows):
        s.box(x0 - 176, y0 + r * c, 52, c - 6, f"{r}·S", role="a", size="small")
    s.text(x0 + cols * c / 2 - 3, y0 - 50, "cols[None, :]", size="small", mono=True)
    for q in range(cols):
        s.box(x0 + q * c, y0 - 34, c - 6, 24, str(q), role="d", size="small")
    for r in range(rows):
        for q in range(cols):
            s.rect(x0 + q * c, y0 + r * c, c - 6, c - 6, fill="f-c", stroke="s-c", sw=0.8)
            s.text(x0 + q * c + (c - 6) / 2, y0 + r * c + (c - 6) / 2, f"{r},{q}", size="tiny")
    s.text(x0 - 97, y0 + rows * c / 2 - 3, "+", size="big", bold=True)
    s.text(x0 + cols * c + 16, y0 + rows * c / 2 - 3, "element (r, q) = base + r·S + q", anchor="start",
           size="small")
    notes = [("ptrs = base + rows[:, None] * S + cols[None, :]", True),
             ("mask = (rows[:, None] < M) & (cols[None, :] < N)", True),
             ("tile = tl.load(ptrs, mask=mask, other=0.0)  # masked lanes read 0", True),
             ("ptrs += BLOCK_K * stride  # move the whole tile along K", True)]
    for i, (ln, mono) in enumerate(notes):
        s.text(60, 250 + i * 22, ln, anchor="start", size="small", mono=mono)
    s.text(360, 344, "S is the row stride in elements; Triton scales by the element size itself.",
           size="small", role="muted")
    return s


def fig_compiler(name):
    s = Svg(name, 720, 300, "From a @triton.jit function to machine code")
    stages = [("Python AST", "@triton.jit\nfunction", "muted"),
              ("Triton IR", "block ops,\nno layouts", "a"),
              ("TritonGPU IR", "layouts, smem,\npipelining", "b"),
              ("LLVM IR", "per-thread\ncode", "d"),
              ("PTX / AMDGCN", "→ cubin / hsaco\n(ptxas, LLVM)", "c")]
    w, h, gap, y = 118, 78, 21, 50
    x0 = (720 - (5 * w + 4 * gap)) / 2
    for i, (title, body, role) in enumerate(stages):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, h, fill="fig-paper" if role == "muted" else f"f-{role}", stroke=f"s-{role}", sw=1.4, rx=6)
        s.text(x + w / 2, y + 17, title, size="small", bold=True)
        for j, ln in enumerate(body.split("\n")):
            s.text(x + w / 2, y + 42 + j * 17, ln, size="small")
        if i < 4:
            s.arrow(x + w + 2, y + h / 2, x + w + gap - 2, y + h / 2, role="muted", sw=1.2)
    notes = [("Specialisation", "one binary per set of constexpr values, dtypes and pointer alignments"),
             ("Cache", "compiled kernels are cached on disk (~/.triton/cache)"),
             ("Inspect", "kernel.asm['ttgir'], ['ptx'] and ['cubin'] of the compiled handle"),
             ("Interpreter", "TRITON_INTERPRET=1 skips all of this and runs the ops in NumPy")]
    for i, (a, b) in enumerate(notes):
        s.text(40, 170 + i * 30, a + ":", anchor="start", size="small", bold=True)
        s.text(160, 170 + i * 30, b, anchor="start", size="small")
    return s
