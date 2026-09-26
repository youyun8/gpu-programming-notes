"""Figures for tutorials/08-deploying-this-site.md."""
from .svg import Svg


def fig_pipeline(name):
    s = Svg(name, 720, 250, "From the repository to a static site")
    cols = [("repository", ["README.md", "tutorials/*.md", "tutorials/figures/", "leetgpu/, tensara/",
                            "mkdocs.yml"], "ink", 20),
            ("scripts/build_site.py", ["pages + solution sources", "inline SVG figures", "rewritten links",
                                       "generated nav"], "a", 260),
            ("mkdocs build", ["build/site/", "static HTML, search", "KaTeX, fonts vendored"], "c", 500)]
    for title, items, role, x in cols:
        s.text(x + 100, 24, title, size="small", bold=True, mono=True)
        s.rect(x, 36, 200, 30 + 24 * len(items), fill=f"f-{role}" if role != "ink" else "fig-panel",
               stroke=f"s-{role}", sw=1.2, rx=6)
        for i, it in enumerate(items):
            s.text(x + 14, 58 + i * 24, it, anchor="start", size="small", mono=it.endswith(("md", "svg", "/",
                                                                                              "yml")))
    s.arrow(222, 100, 258, 100, role="a")
    s.arrow(462, 100, 498, 100, role="c")
    s.text(360, 200, "scripts/build_figures.py regenerates the SVGs from Python; CI checks that both the figures",
           size="small", role="muted")
    s.text(360, 218, "and the README index are current, then builds the site with --strict.", size="small",
           role="muted")
    return s
