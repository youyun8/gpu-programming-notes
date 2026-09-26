#!/usr/bin/env python3
"""Assemble the static site sources (MkDocs) from the repository.

    python3 scripts/build_site.py              # writes build/site-src/ and build/mkdocs.yml
    mkdocs build -f build/mkdocs.yml           # -> build/site/  (static HTML)
    mkdocs serve -f build/mkdocs.yml           # live preview on http://127.0.0.1:8000
    python3 scripts/build_site.py --bundle     # also writes build/gpu-programming-notes.md
                                               # (single file for pandoc -> PDF / EPUB)

Every tutorial becomes a page. Every problem becomes a page containing the
write-up followed by the full solution source. Relative links between files
are rewritten to the generated pages, and every referenced source file is
also published verbatim so it can be downloaded.
"""
import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
OUT = BUILD / "site-src"
ZH_ROOT = ROOT / "locale" / "zh-Hant"
ZH_SUFFIX = "zh-Hant"
PLATFORMS = {"leetgpu": "LeetGPU", "tensara": "Tensara"}
DIFFICULTIES = ["easy", "medium", "hard"]
STATUS_ICONS = {"solved": "✅", "wip": "🚧", "todo": "⬜"}
CODE_LANGUAGES = {".cu": "cuda", ".cuh": "cuda", ".hip": "cpp", ".h": "cpp", ".cpp": "cpp", ".py": "python",
                  ".sh": "bash"}
# Tutorial sub-directories with pages of their own: directory -> (navigation title,
# file-name prefix of the chapter they are listed after).
TUTORIAL_SECTIONS = {"gemm": ("04.x GEMM Deep Dive", "04-")}
# The tutorials are grouped into parts in the navigation: (title, chapter file-name prefixes).
# A sub-directory section goes into the part of the chapter it follows.
# Parts, in reading order; chapters appear in the order of their prefixes here, not by number.
TUTORIAL_PARTS = [
    ("Part I · Foundations", ("00-", "01-", "02-", "03-", "09-")),
    ("Part II · Parallel Patterns", ("10-", "11-", "12-", "13-")),
    ("Part III · Matrix Multiplication", ("04-",)),
    ("Part IV · Portable Model Kernels", ("14-", "15-")),
    ("Part V · AMD Production Kernels", ("05-", "06-", "07-", "16-")),
    ("Part VI · Publishing", ("08-",)),
]
LINK_RE = re.compile(r"(!?\[[^\]]*\])\(([^)\s]+)\)")
# A figure is an image alone on its line. SVG is inlined so its palette follows the theme.
FIGURE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+\.(?:svg|png|jpe?g|webp))\)[ \t]*$", re.M | re.I)


def parse_front_matter(text: str):
    """Return (meta, body). Values are kept as strings; lists as Python lists."""
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        meta[key.strip()] = value
    return meta, text[match.end():]


def site_path(repo_path: Path) -> Path:
    """Repository file -> generated page path (relative to OUT)."""
    rel = repo_path.relative_to(ROOT)
    if rel.name == "README.md":
        return rel.parent / "index.md"
    if rel.suffix == ".md":
        return rel
    if rel.suffix in CODE_LANGUAGES and rel.parent.parts[:1] == ("tutorials",):
        return rel.with_name(rel.name.replace(".", "-") + ".md")
    return rel


def translated_source(source: Path) -> Path:
    """Return the Traditional Chinese source that mirrors a repository file."""
    return ZH_ROOT / source.relative_to(ROOT)


def translated_page(page: Path) -> Path:
    """Return the suffix-style page name consumed by mkdocs-static-i18n."""
    return page.with_name(f"{page.stem}.{ZH_SUFFIX}{page.suffix}")



HOME_TEMPLATE = """---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# GPU Programming Notes

From your first CUDA kernel to Triton, Kimi K3, and AMD production kernels,
with a worked, tested solution to every LeetGPU and Tensara problem.

[Start the tutorials](tutorials/index.md){{ .md-button .md-button--primary }}
[Browse problems](leetgpu/index.md){{ .md-button }}
[Topics](tags.md){{ .md-button }}

<div class="stats">
<div><b>{total}</b>solved problems</div>
<div><b>{chapters}</b>tutorial chapters</div>
<div><b>100%</b>tested on the reference cases</div>
</div>

</div>

## Where to Start

<div class="grid cards" markdown>

-   :material-school:{{ .lg .middle }} **CUDA Foundations**

    ---

    Execution model, memory hierarchy, reductions and profiling, each derived
    from first principles with the cost model written out.

    [:octicons-arrow-right-24: Chapters 00–03, 09](tutorials/index.md)

-   :material-vector-combine:{{ .lg .middle }} **Parallel Patterns**

    ---

    Warp primitives, scan, stencils, and FlashAttention's online softmax,
    each with a tested example program.

    [:octicons-arrow-right-24: Chapters 10–13](tutorials/10-warp-primitives.md)

-   :material-matrix:{{ .lg .middle }} **Matrix Multiplication**

    ---

    From a tiled kernel to `cp.async` pipelines, warp tiling, Stream-K and
    tensor cores, one tested program per technique.

    [:octicons-arrow-right-24: Chapter 04 and 04.x](tutorials/04-tiled-matmul.md)

-   :material-chip:{{ .lg .middle }} **AMD GEMM Deep Dive**

    ---

    CDNA3 and MFMA, AITER assembly, hipBLASLt/TensileLite, and FlyDSL's
    quantized Kimi K3 MoE path.

    [:octicons-arrow-right-24: Chapters 05–07 and 16](tutorials/05-amd-cdna3-mfma.md)

-   :material-code-braces:{{ .lg .middle }} **LeetGPU: {leetgpu} Problems**

    ---

    Elementwise ops to attention, sorting, FFT and full transformer blocks.

    [:octicons-arrow-right-24: Problem index](leetgpu/index.md)

-   :material-lightning-bolt:{{ .lg .middle }} **Tensara: {tensara} Problems**

    ---

    Benchmark-style kernels, including MXFP4/MXFP8/NVFP4 quantised GEMMs.

    [:octicons-arrow-right-24: Problem index](tensara/index.md)

-   :material-cpu-64-bit:{{ .lg .middle }} **cuemu**

    ---

    A CPU emulator that runs every solution against the official reference
    tests, so no GPU is needed.

    [:octicons-arrow-right-24: How it works](tools/cuemu/index.md)

-   :material-language-python:{{ .lg .middle }} **Triton**

    ---

    Block-level Python from fused softmax and FlashAttention to Quark
    quantization, serving caches, and recurrent Kimi Delta Attention.

    [:octicons-arrow-right-24: Chapters 14–15](tutorials/14-triton.md)

-   :material-rocket-launch:{{ .lg .middle }} **Deploy Your Own Copy**

    ---

    GitHub Pages, any static host, or offline as EPUB/PDF.

    [:octicons-arrow-right-24: Chapter 08](tutorials/08-deploying-this-site.md)

</div>

## How Every Problem Page Is Organised

1. **Problem**: the task in my own words, with shapes and data types.
2. **Formulation**: the exact mathematics in TeX, followed by a symbol table
   that defines every symbol.
3. **Approach**: how the work is split across threads, blocks and warps, and why.
4. **Cost analysis**: FLOPs, bytes moved and arithmetic intensity, which tell
   you whether the kernel is memory- or compute-bound.
5. **Pitfalls** and **verification**: what goes wrong, and how the solution was tested.
6. **Solution**: the complete source, with line numbers.
"""


HOME_TEMPLATE_ZH_HANT = """---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# GPU 程式設計筆記

從第一個 CUDA kernel，一路學到 Triton、Kimi K3 與 AMD 生產環境 kernel。
網站也收錄每一題 LeetGPU 與 Tensara 的完整解法與測試。

[開始學習](tutorials/index.md){{ .md-button .md-button--primary }}
[瀏覽題目](leetgpu/index.md){{ .md-button }}
[依主題查找](tags.md){{ .md-button }}

<div class="stats">
<div><b>{total}</b>題已完成</div>
<div><b>{chapters}</b>個教學章節</div>
<div><b>100%</b>通過參考測試</div>
</div>

</div>

## 建議學習路徑

<div class="grid cards" markdown>

-   :material-school:{{ .lg .middle }} **基礎觀念**

    ---

    了解執行模型、記憶體階層、平行歸約與效能分析。

    [:octicons-arrow-right-24: 第 00–03、09 章](tutorials/index.md)

-   :material-vector-combine:{{ .lg .middle }} **平行運算模式**

    ---

    學習 warp 指令、scan、stencil，以及 FlashAttention 的線上 softmax。

    [:octicons-arrow-right-24: 第 10–13 章](tutorials/10-warp-primitives.md)

-   :material-matrix:{{ .lg .middle }} **矩陣乘法**

    ---

    從 tiled GEMM 到 `cp.async`、Stream-K 與 tensor core。

    [:octicons-arrow-right-24: 第 04、04.x 章](tutorials/04-tiled-matmul.md)

-   :material-language-python:{{ .lg .middle }} **Triton 與模型 kernel**

    ---

    從區塊層級程式設計，進入 Quark 量化、SGLang 快取與 Kimi Delta Attention。

    [:octicons-arrow-right-24: 第 14–15 章](tutorials/14-triton.md)

-   :material-chip:{{ .lg .middle }} **AMD 生產環境 kernel**

    ---

    CDNA3、MFMA、AITER、hipBLASLt，以及 FlyDSL 的 Kimi K3 MoE 路徑。

    [:octicons-arrow-right-24: 第 05–07、16 章](tutorials/05-amd-cdna3-mfma.md)

-   :material-code-braces:{{ .lg .middle }} **LeetGPU：{leetgpu} 題**

    ---

    從逐元素運算，到 attention、排序、FFT 與完整 transformer block。

    [:octicons-arrow-right-24: 題目索引](leetgpu/index.md)

-   :material-lightning-bolt:{{ .lg .middle }} **Tensara：{tensara} 題**

    ---

    包含 MXFP4、MXFP8 與 NVFP4 量化 GEMM 的效能導向題目。

    [:octicons-arrow-right-24: 題目索引](tensara/index.md)

-   :material-rocket-launch:{{ .lg .middle }} **部署自己的版本**

    ---

    發布到 GitHub Pages 或靜態主機，也可匯出 EPUB 與單一 Markdown。

    [:octicons-arrow-right-24: 第 08 章](tutorials/08-deploying-this-site.md)

</div>

## 每一題的固定結構

1. **題目**：用簡單文字說明工作、形狀與資料型別。
2. **數學定義**：列出公式，並在表格中定義每個符號。
3. **平行方法**：說明 thread、block 與 warp 如何分工。
4. **成本分析**：計算 FLOP、資料量與算術強度。
5. **常見錯誤與驗證**：解釋容易出錯之處與測試方式。
6. **完整解答**：附上可下載、含行號的原始碼。
"""


LIST_ITEM_RE = re.compile(r"^( *)([-*+]|\d+[.)])( +)\S")
FENCE_RE = re.compile(r"^ *(```|~~~)")


def normalize_lists(text: str) -> str:
    """Convert GitHub-style lists to what Python-Markdown (MkDocs) expects.

    GitHub accepts a list right after a paragraph line and nests with 2-3
    spaces; Python-Markdown needs a blank line before the list and 4 spaces
    per nesting level (for sub-lists, continuation paragraphs and fenced code
    inside items). Lines are re-indented to 4 * depth, where depth is the
    number of list items whose content column encloses the line.
    """
    out = []
    stack = []  # content column (original indentation) of each enclosing list item
    prev_blank = True
    prev_item = False
    fence, fence_shift = None, 0
    for line in text.split("\n"):
        if fence:
            if line.strip().startswith(fence):
                fence = None
            out.append(_shift(line, fence_shift))
            continue
        stripped = line.lstrip(" ")
        if not stripped:
            out.append("")
            prev_blank = True
            continue
        indent = len(line) - len(stripped)
        item = LIST_ITEM_RE.match(line)
        lazy = not item and not prev_blank and stack and indent < stack[-1]
        if not lazy:
            while stack and indent < stack[-1]:
                stack.pop()
        depth = len(stack)
        if item:
            if not prev_blank and not prev_item:
                out.append("")
            new_indent = 4 * depth
            stack.append(indent + len(item.group(2)) + len(item.group(3)))
        elif depth:
            new_indent = 4 * depth + (0 if lazy else indent - stack[-1])
        else:
            new_indent = indent
        new_line = " " * new_indent + stripped
        fence_match = FENCE_RE.match(line)
        if fence_match:
            fence, fence_shift = fence_match.group(1), new_indent - indent
        out.append(new_line)
        prev_blank = False
        prev_item = bool(item)
    return "\n".join(out)


def _shift(line: str, shift: int) -> str:
    if shift >= 0:
        return " " * shift + line if line else line
    strip = min(-shift, len(line) - len(line.lstrip(" ")))
    return line[strip:]


class SiteBuilder:
    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.static_files = set()
        self.warnings = []

    # ----- link handling ---------------------------------------------------------------
    def rewrite_links(self, text: str, source: Path, page: Path) -> str:
        """Rewrite relative links in `text` (read from repo file `source`) for `page`."""

        def repl(m):
            label, target = m.group(1), m.group(2)
            if re.match(r"^[a-z]+:|^#|^/", target):
                return m.group(0)
            path_part, _, anchor = target.partition("#")
            resolved = (source.parent / path_part).resolve()
            inside_repo = resolved == ROOT or ROOT in resolved.parents
            if not inside_repo or not resolved.exists():
                self.warnings.append(f"{source.relative_to(ROOT)}: broken link {target}")
                return m.group(0)
            if resolved.is_dir():
                resolved = resolved / "README.md"
                if not resolved.exists():
                    self.warnings.append(f"{source.relative_to(ROOT)}: link to directory without README {target}")
                    return m.group(0)
            dest = site_path(resolved)
            if dest == resolved.relative_to(ROOT) and resolved.suffix != ".md":
                self.static_files.add(resolved)  # published verbatim (e.g. a solution.cu)
            rel = Path(_relpath(dest, page.parent))
            return f"{label}({rel.as_posix()}{'#' + anchor if anchor else ''})"

        # Leave fenced code blocks untouched.
        parts = re.split(r"(```.*?```)", text, flags=re.S)
        return "".join(p if p.startswith("```") else LINK_RE.sub(repl, p) for p in parts)

    def inline_figures(self, text: str, source: Path) -> str:
        """Replace standalone images with accessible, IEEE-style numbered figures.

        SVG classes pick up the site's --fig-* colour properties after inlining. Raster
        images remain ordinary image elements. Numbering restarts on each page, like a
        self-contained technical article. `code` spans are kept; $math$ is left to KaTeX.
        """
        figure_number = 0

        def repl(m):
            nonlocal figure_number
            figure_number += 1
            path = (source.parent / m.group(2)).resolve()
            if not path.exists():
                self.warnings.append(f"{source.relative_to(ROOT)}: missing figure {m.group(2)}")
                return m.group(0)
            if path.suffix.lower() == ".svg":
                visual = re.sub(r"<\?xml[^>]*\?>", "", path.read_text()).strip()
                visual = " ".join(line.strip() for line in visual.splitlines())
                visual = re.sub(r"<svg\b", f'<svg role="img" aria-labelledby="figure-{figure_number}-caption"', visual, count=1)
            else:
                self.static_files.add(path)
                visual = f'<img src="{html.escape(m.group(2), quote=True)}" alt="">'
            caption = re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(m.group(1), quote=False))
            label = f"Fig. {figure_number}."
            return (f'<figure class="diagram" id="figure-{figure_number}">{visual}'
                    f'<figcaption id="figure-{figure_number}-caption">'
                    f'<span class="fig-label">{label}</span> {caption}</figcaption></figure>')

        parts = re.split(r"(```.*?```)", text, flags=re.S)
        return "".join(p if p.startswith("```") else FIGURE_RE.sub(repl, p) for p in parts)

    def write(self, page: Path, text: str):
        dest = OUT / page
        dest.parent.mkdir(parents=True, exist_ok=True)
        if page.suffix == ".md":
            front = re.match(r"^---\n.*?\n---\n", text, re.S)
            head = front.group(0) if front else ""
            text = head + normalize_lists(text[len(head):])
        dest.write_text(text)

    # ----- pages --------------------------------------------------------------------------
    def home(self):
        """Landing page (hero + cards) and an 'About' page rendered from README.md."""
        text = (ROOT / "README.md").read_text()
        for tag, platform in (("LEETGPU", "leetgpu"), ("TENSARA", "tensara")):
            text = re.sub(rf"<!-- BEGIN {tag} INDEX -->.*?<!-- END {tag} INDEX -->",
                          f"@@{platform}@@", text, flags=re.S)
        about = self.rewrite_links(text, ROOT / "README.md", Path("about.md"))
        for platform, name in PLATFORMS.items():
            about = about.replace(f"@@{platform}@@", f"See the [{name} index]({platform}/index.md).")
        self.write(Path("about.md"), about)

        zh_readme = translated_source(ROOT / "README.md")
        if zh_readme.exists():
            zh_text = zh_readme.read_text()
            for tag, platform in (("LEETGPU", "leetgpu"), ("TENSARA", "tensara")):
                zh_text = re.sub(rf"<!-- BEGIN {tag} INDEX -->.*?<!-- END {tag} INDEX -->",
                                 f"@@{platform}@@", zh_text, flags=re.S)
            zh_about_page = translated_page(Path("about.md"))
            zh_about = self.rewrite_links(zh_text, ROOT / "README.md", zh_about_page)
            zh_about = zh_about.replace("@@leetgpu@@", "請見 [LeetGPU 索引](leetgpu/index.md)。")
            zh_about = zh_about.replace("@@tensara@@", "請見 [Tensara 索引](tensara/index.md)。")
            self.write(zh_about_page, zh_about)
        else:
            self.warnings.append("README.md: missing Traditional Chinese translation")

        counts = {p: sum(1 for d in (ROOT / p).iterdir() if (d / "README.md").exists()) for p in PLATFORMS}
        chapters = len([p for p in (ROOT / "tutorials").glob("[0-9][0-9]-*.md")])
        self.write(Path("index.md"), HOME_TEMPLATE.format(
            leetgpu=counts["leetgpu"], tensara=counts["tensara"], total=sum(counts.values()), chapters=chapters))
        self.write(translated_page(Path("index.md")), HOME_TEMPLATE_ZH_HANT.format(
            leetgpu=counts["leetgpu"], tensara=counts["tensara"], total=sum(counts.values()), chapters=chapters))

    def tutorials(self):
        nav, code_nav = [], {}
        section_nav = {}  # sub-directory -> its pages (README first)
        tdir = ROOT / "tutorials"
        for src in sorted(tdir.rglob("*")):
            if src.is_dir() or src.name.startswith("."):
                continue
            page = site_path(src)
            if src.suffix == ".md":
                text = self.inline_figures(src.read_text(), src)
                self.write(page, self.rewrite_links(text, src, page))
                zh_src = translated_source(src)
                if zh_src.exists():
                    zh_page = translated_page(page)
                    zh_text = self.inline_figures(zh_src.read_text(), src)
                    self.write(zh_page, self.rewrite_links(zh_text, src, zh_page))
                else:
                    self.warnings.append(
                        f"{src.relative_to(ROOT)}: missing Traditional Chinese translation")
            elif src.suffix in CODE_LANGUAGES:
                lang = CODE_LANGUAGES[src.suffix]
                self.static_files.add(src)
                raw = _relpath(src.relative_to(ROOT), page.parent)
                body = (f"# {src.name}\n\n[Download `{src.name}`]({raw})\n\n"
                        f"````{lang} title=\"{src.relative_to(ROOT).as_posix()}\"\n{src.read_text().rstrip()}\n````\n")
                self.write(page, body)
            else:
                self.static_files.add(src)
                continue
            if src.suffix in CODE_LANGUAGES:
                rel = src.relative_to(tdir)
                if rel.parts[:2] == ("examples", "14-triton"):
                    group = "Chapter 14 · Triton Fundamentals"
                elif rel.parts[:2] == ("examples", "15-triton-k3"):
                    group = "Chapter 15 · Triton Model Kernels"
                elif rel.parts[:1] == ("examples",):
                    group = "Chapters 09–13 · CUDA Examples"
                elif rel.parts[:1] == ("gemm",):
                    group = "Chapter 04.x · GEMM Programs"
                elif rel.parts[:1] == ("amd",):
                    group = "AMD Examples"
                else:
                    group = "Other Examples"
                code_nav.setdefault(group, []).append({rel.as_posix(): page.as_posix()})
            elif src.parent == tdir:
                if src.name != "README.md":
                    nav.append(page.as_posix())
            elif src.parent.name in TUTORIAL_SECTIONS:
                pages = section_nav.setdefault(src.parent.name, [])
                pages.insert(0, page.as_posix()) if src.name == "README.md" else pages.append(page.as_posix())
        # Sections (tutorials/<dir>/*.md) go right after the chapter they extend.
        for dirname, pages in section_nav.items():
            title, after = TUTORIAL_SECTIONS[dirname]
            at = next((i + 1 for i, p in enumerate(nav) if isinstance(p, str) and Path(p).name.startswith(after)),
                      len(nav))
            nav.insert(at, {title: pages})
        # Group chapters (and the sections that follow them) into parts, in TUTORIAL_PARTS order.
        groups, current, loose = [], None, []
        for item in nav:
            if isinstance(item, str):
                name = Path(item).name
                current = next(((pi, prefixes.index(pre)) for pi, (_, prefixes) in enumerate(TUTORIAL_PARTS)
                                for pre in prefixes if name.startswith(pre)), None)
                if current is not None:
                    groups.append((current, [item]))
                    continue
            if current is None:
                loose.append(item)
            else:
                groups[-1][1].append(item)
        parts = []
        for (part, _), items in sorted(groups, key=lambda g: g[0]):
            title = TUTORIAL_PARTS[part][0]
            if not parts or title not in parts[-1]:
                parts.append({title: []})
            parts[-1][title].extend(items)
        code_sections = [{title: pages} for title, pages in code_nav.items()]
        return loose + parts + ([{"Example Code": code_sections}] if code_sections else [])

    def problems(self, platform: str):
        rows = {d: [] for d in DIFFICULTIES}
        rows_zh = {d: [] for d in DIFFICULTIES}
        nav = {d: [] for d in DIFFICULTIES}
        for pdir in sorted((ROOT / platform).iterdir()):
            readme = pdir / "README.md"
            if not readme.exists():
                continue
            meta, body = parse_front_matter(readme.read_text())
            page = site_path(readme)
            sources = sorted(p for p in pdir.iterdir() if p.suffix in (".cu", ".py") and p.is_file())
            difficulty = meta.get("difficulty") if meta.get("difficulty") in DIFFICULTIES else "easy"
            tags = meta.get("tags") or []
            title = meta.get("title", pdir.name)

            # The README's own "**Platform:** ..." line is replaced by a header bar.
            body = re.sub(r"^\*\*Platform:\*\*.*\n", "", body.strip(), count=1, flags=re.M)
            body = re.sub(r"^# .*\n", "", body, count=1)
            header = [f'<div class="problem-meta" markdown>',
                      f'<span class="badge {platform}">{PLATFORMS[platform]}</span>'
                      f'<span class="badge {difficulty}">{difficulty}</span>',
                      " ".join(f'<span class="chip">{t}</span>' for t in tags),
                      '<span class="spacer"></span>',
                      f'[:octicons-link-external-16: Statement]({meta.get("url", "")}){{ .md-button }}']
            for src in sources:
                header.append(f"[:material-download: {src.name}]({src.name}){{ .md-button }}")
            header.append("</div>")
            parts = ["---", f"title: {json.dumps(title)}", "hide:", "  - tags", f"description: {json.dumps(PLATFORMS[platform] + ' ' + difficulty + ' problem: ' + title)}", "tags:", *[f"  - {t}" for t in tags], "---", "", f"# {title}", "", *header, "",
                     self.rewrite_links(body, readme, page), ""]
            for src in sources:
                self.static_files.add(src)
                lang = CODE_LANGUAGES[src.suffix]
                lines = src.read_text().count("\n") + 1
                parts += [f"## Solution: `{src.name}`", "",
                          f"{lines} lines · [Download]({src.name}) · "
                          f"[View on GitHub]({self.repo_url}/blob/main/{src.relative_to(ROOT).as_posix()})", "",
                          f"````{lang} linenums=\"1\" title=\"{src.relative_to(ROOT).as_posix()}\"",
                          src.read_text().rstrip(), "````", ""]
            self.write(page, "\n".join(parts))
            chips = " ".join(f'<span class="chip">{t}</span>' for t in tags) or "–"
            rows[difficulty].append(
                f"| [{title}]({pdir.name}/index.md) | {chips} | [:octicons-link-external-16:]({meta.get('url', '')}) |")

            zh_readme = translated_source(readme)
            if zh_readme.exists():
                zh_meta, zh_body = parse_front_matter(zh_readme.read_text())
                zh_title = zh_meta.get("title", title)
                zh_body = re.sub(r"^\*\*(?:Platform|平台)：?\*\*.*\n", "", zh_body.strip(),
                                 count=1, flags=re.M)
                zh_body = re.sub(r"^# .*\n", "", zh_body, count=1)
                zh_page = translated_page(page)
                zh_header = [
                    '<div class="problem-meta" markdown>',
                    f'<span class="badge {platform}">{PLATFORMS[platform]}</span>'
                    f'<span class="badge {difficulty}">{difficulty}</span>',
                    " ".join(f'<span class="chip">{t}</span>' for t in tags),
                    '<span class="spacer"></span>',
                    f'[:octicons-link-external-16: 官方題目]({meta.get("url", "")}){{ .md-button }}',
                ]
                for source_file in sources:
                    zh_header.append(
                        f"[:material-download: {source_file.name}]({source_file.name}){{ .md-button }}")
                zh_header.append("</div>")
                zh_parts = [
                    "---", f"title: {json.dumps(zh_title)}", "hide:", "  - tags",
                    f"description: {json.dumps(PLATFORMS[platform] + ' ' + zh_title)}",
                    "tags:", *[f"  - {tag}" for tag in tags], "---", "", f"# {zh_title}", "",
                    *zh_header, "", self.rewrite_links(zh_body, readme, zh_page), "",
                ]
                for source_file in sources:
                    language = CODE_LANGUAGES[source_file.suffix]
                    lines = source_file.read_text().count("\n") + 1
                    zh_parts += [
                        f"## 完整解答：`{source_file.name}`", "",
                        f"{lines} 行 · [下載]({source_file.name}) · "
                        f"[在 GitHub 檢視]({self.repo_url}/blob/main/{source_file.relative_to(ROOT).as_posix()})",
                        "", f"````{language} linenums=\"1\" title=\"{source_file.relative_to(ROOT).as_posix()}\"",
                        source_file.read_text().rstrip(), "````", "",
                    ]
                self.write(zh_page, "\n".join(zh_parts))
                rows_zh[difficulty].append(
                    f"| [{zh_title}]({pdir.name}/index.md) | {chips} | "
                    f"[:octicons-link-external-16:]({meta.get('url', '')}) |")
            else:
                self.warnings.append(
                    f"{readme.relative_to(ROOT)}: missing Traditional Chinese translation")
            nav[difficulty].append(page.as_posix())
        count = sum(len(v) for v in rows.values())
        summary = " ".join(f'<span class="badge {d}">{len(rows[d])} {d}</span>' for d in DIFFICULTIES if rows[d])
        out = [f"# {PLATFORMS[platform]}", "",
               f"**{count} problems, all solved and tested.** {summary}", "",
               "Every page has a formal statement of the task (equations with a full symbol table), the parallel "
               "design, a cost analysis, pitfalls, and the complete solution source. The original problem "
               "statements are not reproduced; follow the :octicons-link-external-16: links.", "",
               "Browse by topic on the [tags](../tags.md) page.", ""]
        for d in DIFFICULTIES:
            if not rows[d]:
                continue
            out += [f"## {d.capitalize()} ({len(rows[d])})", "", "| Problem | Topics | Statement |",
                    "|---|---|:-:|", *rows[d], ""]
        self.write(Path(platform) / "index.md", "\n".join(out))
        zh_summary = " ".join(
            f'<span class="badge {d}">{len(rows_zh[d])} {d}</span>'
            for d in DIFFICULTIES if rows_zh[d])
        zh_out = [
            f"# {PLATFORMS[platform]}", "",
            f"**共 {count} 題，全部已有解答並通過測試。** {zh_summary}", "",
            "每一頁都包含簡明題意、完整符號表、平行化方法、成本分析、"
            "常見錯誤與完整原始碼。本站不重製原始題目；請使用 "
            ":octicons-link-external-16: 連結前往官方頁面。", "",
            "也可以在[主題索引](../tags.md)中依技術與領域查找。", "",
        ]
        difficulty_names = {"easy": "初階", "medium": "中階", "hard": "進階"}
        for difficulty in DIFFICULTIES:
            if not rows_zh[difficulty]:
                continue
            zh_out += [
                f"## {difficulty_names[difficulty]}（{len(rows_zh[difficulty])} 題）", "",
                "| 題目 | 主題 | 官方頁面 |", "|---|---|:-:|",
                *rows_zh[difficulty], "",
            ]
        self.write(translated_page(Path(platform) / "index.md"), "\n".join(zh_out))
        return nav

    def tools(self):
        pages = []
        for readme in sorted((ROOT / "tools").glob("*/README.md")):
            page = site_path(readme)
            self.write(page, self.rewrite_links(readme.read_text(), readme, page))
            zh_readme = translated_source(readme)
            if zh_readme.exists():
                zh_page = translated_page(page)
                self.write(zh_page, self.rewrite_links(zh_readme.read_text(), readme, zh_page))
            else:
                self.warnings.append(
                    f"{readme.relative_to(ROOT)}: missing Traditional Chinese translation")
            pages.append(page.as_posix())
        return pages

    def copy_static(self):
        shutil.copytree(ROOT / "site_assets", OUT / "assets", dirs_exist_ok=True)
        self.write(Path("tags.md"), "# Topics\n\nEvery problem page is tagged by technique and domain.\n\n"
                                    "<!-- material/tags -->\n")
        self.write(translated_page(Path("tags.md")),
                   "# 主題索引\n\n每一題都依照使用的技術與應用領域加上標籤。\n\n"
                   "<!-- material/tags -->\n")
        for src in sorted(self.static_files):
            dest = OUT / src.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


def _relpath(target: Path, start: Path) -> str:
    import os
    return os.path.relpath(target.as_posix(), start.as_posix() or ".")


def yaml_nav(nav, indent=0) -> str:
    lines = []
    for item in nav:
        pad = "  " * indent
        if isinstance(item, str):
            lines.append(f"{pad}- {item}")
        else:
            (title, children), = item.items()
            if isinstance(children, str):
                lines.append(f"{pad}- {title!r}: {children}")
            else:
                lines.append(f"{pad}- {title!r}:")
                lines.append(yaml_nav(children, indent + 1))
    return "\n".join(lines)


def bundle(order):
    """Concatenate pages into one Markdown file (for pandoc), fixing heading levels and links."""
    def unlink(m):
        # Keep external links and images; inter-page links become plain text.
        if m.group(1).startswith("!") or re.match(r"^[a-z]+:", m.group(2)):
            return m.group(0)
        return m.group(1)[1:-1]

    chunks = []
    for page in order:
        chunks.append(LINK_RE.sub(unlink, (OUT / page).read_text()).strip())
    out = BUILD / "gpu-programming-notes.md"
    out.write_text("\n\n\\newpage\n\n".join(chunks) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", action="store_true", help="also write a single Markdown file for pandoc")
    parser.add_argument("--strict", action="store_true", help="fail on broken relative links")
    parser.add_argument("--repo-url", default="https://github.com/youyun8/gpu-programming-notes")
    parser.add_argument("--site-url", default="", help="public URL of the site (enables canonical links / sitemap)")
    args = parser.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    b = SiteBuilder(args.repo_url)
    b.home()
    tutorial_nav = b.tutorials()
    problem_nav = {p: b.problems(p) for p in PLATFORMS}
    tool_nav = b.tools()
    b.copy_static()

    nav = [{"Home": ["index.md", "about.md"]},
           {"Tutorials": ["tutorials/index.md", *tutorial_nav]}]
    for platform, by_diff in problem_nav.items():
        section = [f"{platform}/index.md"]
        for d in DIFFICULTIES:
            if by_diff[d]:
                section.append({d.capitalize(): by_diff[d]})
        nav.append({PLATFORMS[platform]: section})
    nav.append({"Topics": "tags.md"})
    if tool_nav:
        nav.append({"Tools": [{Path(p).parent.name: p} for p in tool_nav]})

    (BUILD / "mkdocs.yml").write_text(
        "# Generated by scripts/build_site.py - edit mkdocs.yml at the repository root instead.\n"
        "INHERIT: ../mkdocs.yml\n"
        "docs_dir: site-src\n"
        "site_dir: site\n"
        f"repo_url: {args.repo_url}\n"
        + (f"site_url: {args.site_url}\n" if args.site_url else "")
        + "nav:\n" + yaml_nav(nav, 1) + "\n")

    for w in b.warnings:
        print("warning:", w, file=sys.stderr)
    pages = len(list(OUT.rglob("*.md")))
    print(f"wrote {pages} pages and {len(b.static_files)} downloadable files to {OUT.relative_to(ROOT)}/")
    if args.bundle:
        order = ["about.md", "tutorials/index.md"]
        def flatten(items):
            for item in items:
                if isinstance(item, str):
                    yield item
                elif "Example Code" not in item:
                    yield from flatten(next(iter(item.values())))

        order += list(flatten(tutorial_nav))
        for platform, by_diff in problem_nav.items():
            order.append(f"{platform}/index.md")
            for d in DIFFICULTIES:
                order += by_diff[d]
        print(f"wrote {bundle(order).relative_to(ROOT)}")
    return 1 if args.strict and b.warnings else 0


if __name__ == "__main__":
    sys.exit(main())
