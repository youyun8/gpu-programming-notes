import unittest
from pathlib import Path

import build_site


class SiteBuilderTests(unittest.TestCase):
    def test_ieee_figure_numbers_restart_per_page(self):
        builder = build_site.SiteBuilder("https://example.test/repo")
        source = build_site.ROOT / "tutorials" / "README.md"
        image = "![Learning path](figures/overview-learning-path.svg)"
        rendered = builder.inline_figures(f"{image}\n\n{image}", source)

        self.assertIn('<span class="fig-label">Fig. 1.</span> Learning path', rendered)
        self.assertIn('<span class="fig-label">Fig. 2.</span> Learning path', rendered)
        self.assertIn('aria-labelledby="figure-1-caption"', rendered)
        self.assertEqual(builder.inline_figures(image, source).count("Fig. 1."), 1)

    def test_traditional_chinese_page_uses_i18n_suffix(self):
        page = Path("tutorials/14-triton.md")
        self.assertEqual(
            build_site.translated_page(page),
            Path("tutorials/14-triton.zh-Hant.md"),
        )

    def test_every_article_has_a_translation(self):
        sources = list((build_site.ROOT / "tutorials").rglob("*.md"))
        sources += list((build_site.ROOT / "leetgpu").glob("*/README.md"))
        sources += list((build_site.ROOT / "tensara").glob("*/README.md"))
        sources += list((build_site.ROOT / "tools").glob("*/README.md"))
        missing = [
            source.relative_to(build_site.ROOT).as_posix()
            for source in sources
            if not build_site.translated_source(source).exists()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
