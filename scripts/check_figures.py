#!/usr/bin/env python3
"""Check that the text of every tutorial figure is legible.

    pip install playwright && python3 -m playwright install chromium   # once
    python3 scripts/check_figures.py [tutorials/figures/x.svg ...]

Each SVG is rendered in headless Chromium at its natural size and every
<text> element is measured. A figure fails when a label

- overlaps another label,
- sticks out of the figure,
- is crossed by a line, an arrow or the border of a box (text inside a box
  must fit inside it), or
- uses a font smaller than MIN_FONT_PX.

Set CHROMIUM=/path/to/chrome to use a specific browser binary.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "tutorials" / "figures"
MIN_FONT_PX = 11.0
GAP = 2.0   # minimum distance between two labels, px
EDGE = 6.0  # minimum distance from a label to the figure's edge, px

# Runs in the page: collect label boxes (ink approximated by trimming the line box),
# the segments of every stroked line / path, and the rectangles with a visible border.
MEASURE_JS = r"""
() => {
  const svg = document.querySelector('svg');
  const origin = svg.getBoundingClientRect();
  const rel = r => ({x0: r.left - origin.left, y0: r.top - origin.top,
                     x1: r.right - origin.left, y1: r.bottom - origin.top});
  const all = Array.from(svg.querySelectorAll('*'));
  const order = e => all.indexOf(e);
  const texts = [];
  for (const t of svg.querySelectorAll('text')) {
    const r = rel(t.getBoundingClientRect());
    const size = parseFloat(getComputedStyle(t).fontSize);
    // The line box includes ascender/descender space; trim it to approximate the ink.
    const rotated = (t.getAttribute('transform') || '').includes('rotate');
    const trimV = rotated ? 0 : (r.y1 - r.y0) * 0.12, trimH = rotated ? (r.x1 - r.x0) * 0.12 : 0;
    // For line crossings use the label's own (possibly rotated) frame: local bbox + page->local matrix.
    const bb = t.getBBox(), inv = t.getScreenCTM().inverse();
    const tv = bb.height * 0.12;
    texts.push({s: t.textContent, size, i: order(t),
                x0: r.x0 + trimH, x1: r.x1 - trimH, y0: r.y0 + trimV, y1: r.y1 - trimV,
                local: {x0: bb.x, x1: bb.x + bb.width, y0: bb.y + tv, y1: bb.y + bb.height - tv},
                inv: [inv.a, inv.b, inv.c, inv.d, inv.e, inv.f], ox: origin.left, oy: origin.top});
  }
  const segs = [];
  const visible = e => {
    const cs = getComputedStyle(e);
    return cs.stroke !== 'none' && parseFloat(cs.strokeWidth) > 0 && cs.visibility !== 'hidden';
  };
  const ctm = e => e.getScreenCTM();
  const toPage = (e, x, y) => {
    const p = svg.createSVGPoint(); p.x = x; p.y = y;
    const q = p.matrixTransform(ctm(e));
    return [q.x - origin.left, q.y - origin.top];
  };
  for (const e of svg.querySelectorAll('line, path, polyline')) {
    if (e.closest('marker') || !visible(e)) continue;
    const thin = parseFloat(getComputedStyle(e).strokeWidth) < 0.55;  // hairline grid lines
    const kind = e.hasAttribute('marker-end') || e.hasAttribute('marker-start') ? 'arrow' : (thin ? 'thin' : 'line');
    const len = e.getTotalLength();
    const n = Math.max(2, Math.ceil(len / 2));
    let prev = null;
    for (let i = 0; i <= n; i++) {
      const p = e.getPointAtLength(len * i / n);
      const cur = toPage(e, p.x, p.y);
      if (prev) segs.push({kind, a: prev, b: cur, i: order(e)});
      prev = cur;
    }
  }
  const rects = [], plates = [], dots = [];
  for (const e of svg.querySelectorAll('rect')) {
    const r = rel(e.getBoundingClientRect());
    const cs = getComputedStyle(e);
    // Opaque filled rectangles hide whatever was drawn before them (like the label plates).
    const opaque = cs.fill !== 'none' && parseFloat(cs.fillOpacity) >= 0.9 && parseFloat(cs.opacity) >= 0.9;
    if (opaque) plates.push({...r, i: order(e)});
    if (e.classList.contains('fig-plate') || !visible(e)) continue;
    rects.push({...r, i: order(e)});
  }
  for (const e of svg.querySelectorAll('circle')) dots.push({...rel(e.getBoundingClientRect()), i: order(e)});
  const box = svg.viewBox.baseVal;
  return {texts, segs, rects, plates, dots, w: box.width, h: box.height};
}
"""


def overlap(a, b, pad=0.0):
    return (min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]) > pad and
            min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]) > pad)


def seg_hits_box(seg, t, pad=-2.0):
    """Does the segment pass through the text box, or within -pad px of it?"""
    x0, y0, x1, y1 = t["x0"] + pad, t["y0"] + pad, t["x1"] - pad, t["y1"] - pad
    if x1 <= x0 or y1 <= y0:
        return False
    (ax, ay), (bx, by) = seg["a"], seg["b"]
    # Liang-Barsky clipping
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


def rect_border_hits(r, t, pad=1.0, margin=4.0):
    """Text that is partly inside and partly outside a bordered rectangle, or that
    sits inside it closer than `margin` px to its border."""
    inside = (t["x0"] >= r["x0"] + margin and t["x1"] <= r["x1"] - margin and
              t["y0"] >= r["y0"] + margin and t["y1"] <= r["y1"] - margin)
    return overlap(r, t, pad) and not inside and not (
        r["x0"] >= t["x0"] and r["x1"] <= t["x1"] and r["y0"] >= t["y0"] and r["y1"] <= t["y1"])


def inflate(b, d):
    return {**b, "x0": b["x0"] - d, "y0": b["y0"] - d, "x1": b["x1"] + d, "y1": b["y1"] + d}


def check(page, path):
    page.set_content(f"<html><body style='margin:0'>{path.read_text()}</body></html>")
    page.wait_for_timeout(50)
    m = page.evaluate(MEASURE_JS)
    problems = []
    texts = m["texts"]
    for k, t in enumerate(texts):
        label = repr(t["s"][:40])
        # Anything drawn before an opaque plate or filled box that covers the label is hidden by it.
        hidden_below = max([p["i"] for p in m["plates"] if p["i"] < t["i"] and
                            p["x0"] <= t["x0"] and p["x1"] >= t["x1"] and p["y0"] <= t["y0"] and p["y1"] >= t["y1"]],
                           default=-1)
        if t["size"] < MIN_FONT_PX:
            problems.append(f"{label}: font {t['size']:.1f}px < {MIN_FONT_PX}px")
        if t["x0"] < EDGE or t["y0"] < EDGE or t["x1"] > m["w"] - EDGE or t["y1"] > m["h"] - EDGE:
            problems.append(f"{label}: outside the figure (or closer than {EDGE}px to its edge)")
        for u in texts[k + 1:]:
            if overlap(inflate(t, GAP / 2), inflate(u, GAP / 2)):
                problems.append(f"{label} overlaps (or touches) {u['s'][:40]!r}")
        a, b, c, d, e, f = t["inv"]

        def to_local(pt):
            x, y = pt[0] + t["ox"], pt[1] + t["oy"]
            return (a * x + c * y + e, b * x + d * y + f)

        for seg in m["segs"]:
            local_seg = {"a": to_local(seg["a"]), "b": to_local(seg["b"])}
            if seg["kind"] != "thin" and seg["i"] > hidden_below and seg_hits_box(local_seg, t["local"]):
                problems.append(f"{label} is crossed by a {seg['kind']}")
                break
        for r in m["rects"]:
            if r["i"] > hidden_below and rect_border_hits(r, t):
                problems.append(f"{label} crosses the border of a box")
                break
        for d in m["dots"]:
            if d["i"] > hidden_below and overlap(d, t):
                problems.append(f"{label} overlaps a marker dot")
                break
    return problems


def main(argv):
    from playwright.sync_api import sync_playwright

    files = [Path(a) for a in argv] or sorted(FIGURES.glob("*.svg"))
    failed = 0
    with sync_playwright() as p:
        kwargs = {"args": ["--no-sandbox"]}
        if os.environ.get("CHROMIUM"):
            kwargs["executable_path"] = os.environ["CHROMIUM"]
        browser = p.chromium.launch(**kwargs)
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        for f in files:
            problems = check(page, f)
            if problems:
                failed += 1
                print(f"{f.name}:")
                for pr in problems:
                    print(f"  {pr}")
        browser.close()
    print(f"{len(files)} figures checked, {failed} with problems")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
