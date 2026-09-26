"""A tiny SVG drawing library for the tutorial figures.

Figures use CSS classes instead of hard-coded colours. The classes read CSS
custom properties (``--fig-*``) with light-theme fallbacks, so a figure viewed
on GitHub (as an <img>) uses the fallbacks, while the site inlines the SVG and
redefines the properties for its dark theme (site_assets/stylesheets/extra.css).

Colour roles:
    ink, muted, line      text and strokes
    paper, panel          backgrounds
    a, b, c, d, hl        operand A, operand B, result C, a fourth role, highlight
Every role has a stroke/text class ``s-<role>`` and a fill class ``f-<role>``
(soft) and ``f-<role>2`` (stronger fill).
"""
from __future__ import annotations

import html
import re

# role -> (strong colour, soft fill, medium fill); light-theme values
PALETTE = {
    "ink": ("#1f2937", "#f3f4f6", "#e5e7eb"),
    "muted": ("#6b7280", "#f9fafb", "#e5e7eb"),
    "line": ("#9ca3af", "#f3f4f6", "#d1d5db"),
    "a": ("#2563eb", "#dbeafe", "#93c5fd"),
    "b": ("#c2410c", "#ffedd5", "#fdba74"),
    "c": ("#15803d", "#dcfce7", "#86efac"),
    "d": ("#7c3aed", "#ede9fe", "#c4b5fd"),
    "hl": ("#dc2626", "#fee2e2", "#fca5a5"),
}
PAPER = "#ffffff"
PANEL = "#f8fafc"


def _style() -> str:
    rules = [
        f".fig-paper{{fill:var(--fig-paper,{PAPER})}}",
        f".fig-panel{{fill:var(--fig-panel,{PANEL})}}",
        ".fig-none{fill:none}",
    ]
    for role, (strong, soft, mid) in PALETTE.items():
        rules.append(f".s-{role}{{stroke:var(--fig-{role},{strong})}}")
        rules.append(f".t-{role}{{fill:var(--fig-{role},{strong})}}")
        rules.append(f".f-{role}{{fill:var(--fig-{role}-soft,{soft})}}")
        rules.append(f".f-{role}2{{fill:var(--fig-{role}-mid,{mid})}}")
        rules.append(f".k-{role}{{fill:var(--fig-{role},{strong})}}")  # solid fill (markers, dots)
    rules += [
        ".fig-txt{font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;font-size:14px}",
        ".fig-mono{font-family:'Ubuntu Mono',ui-monospace,Menlo,monospace;font-size:14px}",
        ".fig-small{font-size:12px}",
        ".fig-tiny{font-size:9.5px}",
        ".fig-big{font-size:16px}",
        ".fig-bold{font-weight:600}",
        ".fig-it{font-style:italic}",
    ]
    return "".join(rules)


def esc(s: str) -> str:
    return html.escape(str(s), quote=False)


# x_s is a subscript only at the end of a word, so identifiers like __syncthreads stay intact.
_RICH = re.compile(r"(_\{[^}]*\}|\^\{[^}]*\}|(?<![_A-Za-z0-9][_])(?<=[A-Za-z0-9)])_[A-Za-z0-9](?![A-Za-z0-9_])"
                   r"|\^[A-Za-z0-9]|\*[^*]+\*)")


def rich(s: str) -> str:
    """Minimal markup: x_{sub}, x_s, x^{sup}, *italic*."""
    out = []
    for tok in _RICH.split(str(s)):
        if not tok:
            continue
        if tok.startswith("_{"):
            out.append(f'<tspan baseline-shift="sub" font-size="75%">{rich(tok[2:-1])}</tspan>')
        elif tok.startswith("^{"):
            out.append(f'<tspan baseline-shift="super" font-size="75%">{rich(tok[2:-1])}</tspan>')
        elif len(tok) == 2 and tok[0] == "_":
            out.append(f'<tspan baseline-shift="sub" font-size="75%">{esc(tok[1])}</tspan>')
        elif len(tok) == 2 and tok[0] == "^":
            out.append(f'<tspan baseline-shift="super" font-size="75%">{esc(tok[1])}</tspan>')
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            out.append(f'<tspan font-style="italic">{esc(tok[1:-1])}</tspan>')
        else:
            out.append(esc(tok))
    return "".join(out)


def _f(v) -> str:
    """Format a number compactly."""
    if isinstance(v, float):
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return s if s not in ("-0", "") else "0"
    return str(v)


class Svg:
    ARROW_ROLES = ("ink", "muted", "a", "b", "c", "d", "hl")

    def __init__(self, name: str, width: float, height: float, title: str = ""):
        self.name = name
        self.width = width
        self.height = height
        self.title = title
        self.items: list[str] = []

    # ----- primitives ---------------------------------------------------------------
    def add(self, raw: str):
        self.items.append(raw)

    def rect(self, x, y, w, h, fill="fig-paper", stroke="s-ink", sw=1.2, rx=0, extra=""):
        cls = " ".join(c for c in (fill, stroke) if c)
        self.add(f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" rx="{_f(rx)}" '
                 f'class="{cls}" stroke-width="{_f(sw)}"{extra}/>')

    def line(self, x1, y1, x2, y2, stroke="s-ink", sw=1.2, dash=None, arrow=None, arrow_start=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ""
        if arrow:
            m += f' marker-end="url(#{self.name}-ah-{arrow})"'
            if arrow_start:
                m += f' marker-start="url(#{self.name}-ah-{arrow})"'
        self.add(f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" class="{stroke}" '
                 f'stroke-width="{_f(sw)}"{d}{m}/>')

    def arrow(self, x1, y1, x2, y2, role="ink", sw=1.4, dash=None, both=False):
        self.line(x1, y1, x2, y2, stroke=f"s-{role}", sw=sw, dash=dash, arrow=role, arrow_start=both)

    def path(self, d, stroke="s-ink", fill="fig-none", sw=1.2, dash=None, arrow=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if arrow:
            extra += f' marker-end="url(#{self.name}-ah-{arrow})"'
        self.add(f'<path d="{d}" class="{fill} {stroke}" stroke-width="{_f(sw)}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{extra}/>')

    def circle(self, cx, cy, r, fill="k-ink", stroke="", sw=1):
        cls = " ".join(c for c in (fill, stroke) if c)
        self.add(f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" class="{cls}" stroke-width="{_f(sw)}"/>')

    def text(self, x, y, s, role="ink", anchor="middle", size=None, bold=False, mono=False, italic=False,
             rotate=None, baseline="central", plain=False):
        cls = ["fig-mono" if mono else "fig-txt", f"t-{role}"]
        if size == "small":
            cls.append("fig-small")
        elif size == "tiny":
            cls.append("fig-tiny")
        elif size == "big":
            cls.append("fig-big")
        if bold:
            cls.append("fig-bold")
        if italic:
            cls.append("fig-it")
        tr = f' transform="rotate({rotate} {_f(x)} {_f(y)})"' if rotate else ""
        self.add(f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{anchor}" dominant-baseline="{baseline}" '
                 f'class="{" ".join(cls)}"{tr}>{esc(s) if mono or plain else rich(s)}</text>')

    # ----- composites ---------------------------------------------------------------
    def box(self, x, y, w, h, label="", role="ink", fill=None, sw=1.2, rx=4, size=None, bold=False,
            mono=False, text_role=None, dash=None):
        fill = fill or ("fig-paper" if role in ("ink", "muted") else f"f-{role}")
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.rect(x, y, w, h, fill=fill, stroke=f"s-{role}", sw=sw, rx=rx, extra=extra)
        if label != "":
            lines = str(label).split("\n")
            lh = 13 if size == "small" else 17
            y0 = y + h / 2 - (len(lines) - 1) * lh / 2
            for i, ln in enumerate(lines):
                self.text(x + w / 2, y0 + i * lh, ln, role=text_role or "ink", size=size, bold=bold, mono=mono)

    def grid(self, x, y, rows, cols, cw, ch=None, fill_fn=None, stroke="s-line", sw=0.8, outline="s-ink",
             outline_sw=1.4):
        """A rows x cols grid of cells; fill_fn(r, c) -> fill class or None."""
        ch = ch or cw
        for r in range(rows):
            for c in range(cols):
                f = fill_fn(r, c) if fill_fn else None
                self.rect(x + c * cw, y + r * ch, cw, ch, fill=f or "fig-paper", stroke=stroke, sw=sw)
        if outline:
            self.rect(x, y, cols * cw, rows * ch, fill="fig-none", stroke=outline, sw=outline_sw)

    def brace_h(self, x1, x2, y, label="", role="ink", up=False, size="small"):
        """Horizontal dimension line with end ticks and a label."""
        d = -1 if up else 1
        self.line(x1, y, x2, y, stroke=f"s-{role}", sw=1)
        self.line(x1, y - 4, x1, y + 4, stroke=f"s-{role}", sw=1)
        self.line(x2, y - 4, x2, y + 4, stroke=f"s-{role}", sw=1)
        if label:
            self.text((x1 + x2) / 2, y + d * 11, label, role=role, size=size)

    def brace_v(self, x, y1, y2, label="", role="ink", left=True, size="small"):
        self.line(x, y1, x, y2, stroke=f"s-{role}", sw=1)
        self.line(x - 4, y1, x + 4, y1, stroke=f"s-{role}", sw=1)
        self.line(x - 4, y2, x + 4, y2, stroke=f"s-{role}", sw=1)
        if label:
            self.text(x - 6 if left else x + 6, (y1 + y2) / 2, label, role=role, size=size,
                      anchor="end" if left else "start")

    def legend(self, x, y, entries, gap=18, sw=14):
        """entries: list of (fill class, stroke role, label)."""
        for i, (fill, role, label) in enumerate(entries):
            yy = y + i * gap
            self.rect(x, yy - sw / 2 + 1, sw, sw - 2, fill=fill, stroke=f"s-{role}", sw=1, rx=2)
            self.text(x + sw + 6, yy, label, anchor="start", size="small")

    # ----- output -------------------------------------------------------------------
    def render(self) -> str:
        markers = "".join(
            f'<marker id="{self.name}-ah-{r}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="k-{r}"/></marker>'
            for r in self.ARROW_ROLES)
        title = f"<title>{esc(self.title)}</title>" if self.title else ""
        body = "\n".join(self.items)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(self.width)} {_f(self.height)}" '
                f'width="{_f(self.width)}" height="{_f(self.height)}" class="fig-svg" role="img">\n'
                f"{title}<style>{_style()}</style>\n<defs>{markers}</defs>\n"
                f'<rect x="0" y="0" width="{_f(self.width)}" height="{_f(self.height)}" class="fig-paper"/>\n'
                f"{body}\n</svg>\n")
