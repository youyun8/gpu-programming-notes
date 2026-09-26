// Render TeX with KaTeX after every (instant-navigation) page load.
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
    macros: {
      "\\ceil": "\\left\\lceil #1 \\right\\rceil",
      "\\floor": "\\left\\lfloor #1 \\right\\rfloor",
      "\\R": "\\mathbb{R}",
    },
  });
});
