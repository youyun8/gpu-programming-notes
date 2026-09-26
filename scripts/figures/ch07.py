"""Figures for tutorials/07-hipblaslt-tensilelite.md."""
from .svg import Svg


def fig_macro_tile(name):
    s = Svg(name, 720, 330, "MatrixInstruction [32, 32, 1, 2,  1,  4, 1,  2, 2] as nested tiles")
    sc = 1.0
    x0, y0 = 40, 50
    # macro tile 256 (M) x 128 (N)
    s.rect(x0, y0, 128 * sc * 1.6, 256 * sc, fill="fig-panel", stroke="s-ink", sw=1.4)
    W = 128 * 1.6
    roles = ["a", "b", "c", "d"]
    for wm in range(2):
        for wn in range(2):
            wx = x0 + wn * W / 2
            wy = y0 + wm * 128
            s.rect(wx + 2, wy + 2, W / 2 - 4, 124, fill=f"f-{roles[2 * wm + wn]}", stroke=f"s-{roles[2 * wm + wn]}",
                   sw=1)
            for i in range(4):
                s.rect(wx + 8, wy + 6 + i * 30, W / 2 - 16, 26, fill="fig-none", stroke="s-line", sw=0.8, rx=2)
            s.text(wx + W / 4, wy + 64, f"wave {2 * wm + wn}", size="small", bold=True)
    s.brace_h(x0, x0 + W, y0 - 12, "MT1 = 32·2·1·2 = 128 (N)", up=True)
    s.brace_v(x0 - 8, y0, y0 + 256, "")
    s.text(x0 - 14, y0 + 128, "MT0 = 256 (M)", size="small", rotate=-90)
    # explanation column
    tx = 300
    rows = [("MFMA  32 × 32 × 1, 2 blocks", "one instruction: 32 (M) × 64 (N)", "line"),
            ("MIBlockM = 1", "both blocks side by side along N", "line"),
            ("WaveTile 4 × 1", "4 MFMA tiles per wave along M → 128 × 64", "line"),
            ("Waves 2 × 2", "4 waves = 256 threads → macro tile 256 × 128", "line")]
    for i, (a, b, _) in enumerate(rows):
        y = 70 + i * 58
        s.text(tx, y, a, anchor="start", size="small", bold=True, mono=True)
        s.text(tx, y + 20, b, anchor="start", size="small")
    s.text(tx, 300, "Grey boxes: the 4 MFMA tiles (32 × 64 each) of one wave.", anchor="start", size="small",
           role="muted")
    return s


def fig_xcd_remap(name):
    s = Svg(name, 720, 250, "WorkGroupMappingXCC: keep consecutive tiles on one XCD")
    cell = 34
    roles = ["a", "b", "c", "d", "hl", "a", "b", "c"]
    for idx, (title, remap) in enumerate([("default: workgroup i → XCD i mod 8", False),
                                          ("remapped: tile t runs on XCD ⌊t / 2⌋", True)]):
        x0 = 30 + idx * 360
        s.text(x0 + 4 * cell, 22, title, size="small", bold=True)
        for t in range(16):
            r, c = t // 8, t % 8
            xcd = (t // 2) if remap else (t % 8)
            s.rect(x0 + c * cell, 40 + r * cell, cell, cell, fill=f"f-{roles[xcd]}", stroke=f"s-{roles[xcd]}", sw=0.8)
            s.text(x0 + c * cell + cell / 2, 40 + r * cell + 11, f"t{t}", size="tiny", mono=True)
            s.text(x0 + c * cell + cell / 2, 40 + r * cell + 25, f"X{xcd}", size="tiny", mono=True, role="muted")
        s.text(x0 + 4 * cell, 40 + 2 * cell + 18, "tiles in launch order, 2 rows of 8" if not remap else
               "neighbours share an L2", size="small", role="muted")
    s.text(360, 190, "With the default placement, tiles that share an A or B panel run on different XCDs and each",
           size="small")
    s.text(360, 208, "XCD's 4 MiB L2 fetches the panel separately. Remapping which tile each workgroup ID computes",
           size="small")
    s.text(360, 226, "turns neighbours into L2 hits. (Illustration: 16 tiles, groups of 2 per XCD.)", size="small")
    return s


def fig_tuning_flow(name):
    s = Svg(name, 720, 190, "Offline tuning of hipBLASLt")
    steps = [("1. log", "HIPBLASLT_LOG_MASK=32\nrecords bench commands", "a"),
             ("2. tune", "hipblaslt-bench with\nHIPBLASLT_TUNING_FILE", "b"),
             ("3. use", "HIPBLASLT_TUNING_\nOVERRIDE_FILE", "c")]
    for i, (t, d, role) in enumerate(steps):
        x = 30 + i * 240
        s.text(x + 90, 30, t, size="small", bold=True)
        s.box(x, 44, 180, 60, d, role=role, size="small", mono=False)
        if i < 2:
            s.arrow(x + 182, 74, x + 238, 74, role="ink")
    s.text(360, 140, "The tuning file maps each GEMM shape to a solution index, which is only valid for the",
           size="small")
    s.text(360, 158, "same library build and GPU architecture: re-tune after upgrading ROCm.", size="small",
           role="muted")
    return s
