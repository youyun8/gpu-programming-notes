# 08 – 部署本站

> **第六部 · 發布** · 先備知識：無（Python、Git） ·
> 返回：[教學索引](README.md)

此儲存庫會渲染成靜態網站，內容包括：
- 每篇教學；
- 每個 LeetGPU 與 Tensara 問題頁面，包含解說及完整解答原始碼；
- 範例程式。

它也能匯出成離線格式：壓縮的 HTML 網站、EPUB、單一 Markdown 檔，以及在已安裝 LaTeX 時產生的 PDF。本章說明 pipeline 的運作方式，並提供四種發布網站的方法。

**你將學會**

- `scripts/build_site.py` 如何將儲存庫轉成 MkDocs source（頁面、導覽、圖片、連結），以及它刻意排除哪些內容；
- 如何在本機建置及預覽網站；
- 四種發布方式：透過 Actions 發布到 GitHub Pages、`mkdocs gh-deploy`、任意靜態 host，以及離線格式（HTML zip、EPUB、PDF）；
- 發布前 CI 會檢查哪些事項；
- 如何新增問題、章節與圖片，以及數學式和程式碼如何渲染。

## 1. 網站的建置方式

```
repository                       scripts/build_site.py              mkdocs build
───────────────────────────      ─────────────────────────────      ─────────────────
README.md                  ──►   build/site-src/index.md      ──►   build/site/index.html
tutorials/*.md, amd/*.hip  ──►   tutorials/*.md, *-hip.md     ──►   …/tutorials/…
leetgpu/NNN/README.md      ─┐
leetgpu/NNN/solution.cu    ─┴►   leetgpu/NNN/index.md         ──►   leetgpu/NNN/index.html
                                 (+ solution.cu, downloadable)
mkdocs.yml (theme, etc.)   ──►   build/mkdocs.yml (INHERIT + generated nav)
```

![儲存庫如何變成靜態網站](figures/ch08-pipeline.svg)

`scripts/build_site.py` 會做七件事：

1. **教學頁面。** `tutorials/` 下每個 Markdown 檔都會變成頁面。當中的每個程式碼檔（`.hip`、`.h` 等）也會有可渲染的頁面與下載連結。
2. **問題頁面。** 每個問題資料夾會成為一個頁面：
   - 移除 README 的 front matter；
   - 接著放解說；
   - 然後是 `## Solution: solution.cu` 與完整原始碼（含行號與複製按鈕）、下載連結，以及「在 GitHub 檢視」連結。
3. **索引頁面。** 建立 `leetgpu/index.md` 與 `tensara/index.md`，並依難度分組問題表格。
4. **連結。** 改寫相對連結，讓它們能在網站運作：
   - 指向問題資料夾的連結改到該問題頁；
   - 指向 `README.md` 的連結改到它轉成的 `index.md`；
   - 指向原始碼檔的連結會原樣發布該檔案。

   使用 `--strict` 時，任何失效連結都會讓建置失敗。
5. **導覽。** 產生 `build/mkdocs.yml`。該檔會繼承根目錄 `mkdocs.yml` 的 theme 與 Markdown extension，再加入完整 navigation tree。
6. **靜態 asset。** 將 `site_assets/` 複製到網站中的 `assets/`：stylesheet、favicon、KaTeX loader，以及 vendored KaTeX 與字型檔（第 9 節）。
7. **圖片。** 只有一張 Markdown 圖片的教學行會轉成 `<figure>`。圖說依 IEEE 風格編為 `Fig. 1.`、`Fig. 2.`。SVG 會 inline，顏色取自 `--fig-*` CSS property，因此會跟隨明暗主題；點陣圖片則保留一般 image element。在 GitHub 上，同一行仍顯示為一般圖片。

Theme 使用 [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)。它在 `requirements-docs.txt` 中固定低於 MkDocs 2.0，因為 MkDocs 2.0 移除了 Material 所依賴的 plugin system。

`mkdocs-static-i18n` 會將英文版建在網站根目錄，繁體中文版則建在 `/zh-Hant/`。右下角的 `Aa` 控制項會將主題、字體大小、內容寬度與行距儲存在 `localStorage`，也能切換到另一語言的相同頁面。

### 1.1 不會發布的內容

- **問題敘述。** LeetGPU 挑戰文字採 CC BY-NC-ND，Tensara 問題儲存庫則沒有授權，所以此儲存庫不會複製它們。
  - 每個頁面都連到官方敘述。
  - 頁面只包含我自己的摘要與程式碼。
  - 部署時也請維持如此：`scripts/fetch_upstream.sh` 會將上游定義 clone 到 git-ignored 的 `.upstream/`，網站 builder 絕不讀取它。
- **AITER / hipBLASLt 原始碼。** 它們採 MIT 授權，但第 06–07 章只引用短摘錄，並連到上游儲存庫。

## 2. 在本機建置與預覽

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-docs.txt

python3 scripts/build_site.py --strict        # -> build/site-src/, build/mkdocs.yml
mkdocs serve -f build/mkdocs.yml              # http://127.0.0.1:8000, live reload
mkdocs build --strict -f build/mkdocs.yml     # -> build/site/ (plain static files)
```

`make serve` 與 `make site` 封裝了相同命令。

`mkdocs serve` 監看的是 `build/site-src/`，不是儲存庫。編輯教學或解答後，請再次執行 `scripts/build_site.py`，預覽就會重新載入。

`build/site/` 完全是靜態內容：HTML、CSS、JS、client-side search index 與解答檔案。任何 web server 都能託管，也可直接從磁碟開啟。

## 3. 選項 A – GitHub Pages（自動，建議）

`.github/workflows/pages.yml` 會在每次 push 到 `main` 時建置並部署。

1. 在 GitHub 前往 **Settings → Pages → Build and deployment**，將 **Source** 設為 **GitHub Actions**。
   - 私有儲存庫的 Pages 需要付費方案（Pro、Team 或 Enterprise）。使用免費方案時，請先將儲存庫設為公開。
2. Push 到 `main`，或從 **Actions → Deploy site → Run workflow** 手動執行 workflow。
3. Workflow 的 `deploy` job 會印出 URL，例如 `https://<user>.github.io/gpu-programming-notes/`。

Workflow 會執行：

```yaml
- uses: actions/configure-pages@v5          # gives the public base URL
- run: python3 scripts/build_site.py --strict --bundle --site-url "<base_url>/" ...
- run: mkdocs build --strict -f build/mkdocs.yml
- run: pandoc ... -o build/site/downloads/gpu-programming-notes.epub   # offline formats
- uses: actions/upload-pages-artifact@v3    # path: build/site
- uses: actions/deploy-pages@v4
```

若要使用**自訂網域**：
1. 在 **Settings → Pages** 新增網域。
2. 建立 DNS record：將 `CNAME` 指向 `<user>.github.io`。

之後 `configure-pages` 會回報新的 base URL，sitemap 與 canonical link 也會自動跟隨。

## 4. 選項 B – `mkdocs gh-deploy`（手動，不用 Actions）

```bash
python3 scripts/build_site.py --strict --site-url https://<user>.github.io/gpu-programming-notes/
mkdocs gh-deploy -f build/mkdocs.yml --force
```

這會建置網站，並 force-push 到 `gh-pages` branch。接著將 **Settings → Pages → Source** 設為 *Deploy from a branch*，選擇 `gh-pages` 與 `/ (root)`。

不要與選項 A 同時使用，否則兩者會爭用 Pages source。

## 5. 選項 C – 任意靜態 Host

上傳 `build/site/` 資料夾：

| Host | 方法 |
|------|-----|
| Netlify | Build command：`pip install -r requirements-docs.txt && python3 scripts/build_site.py --strict && mkdocs build -f build/mkdocs.yml`；publish directory：`build/site` |
| Cloudflare Pages | 相同 build command 與 output directory；設定 `PYTHON_VERSION=3.12` |
| Vercel | 相同，framework preset 選「Other」 |
| 自有 server | `rsync -av --delete build/site/ user@host:/var/www/gpu-notes/`，再用 nginx/caddy 提供服務 |
| 快速在任意處啟動 | `python3 -m http.server -d build/site 8000` |

若網站位於子路徑（`https://host/notes/`），請傳入 `--site-url https://host/notes/`。MkDocs 使用相對 URL，所以無論如何頁面都能運作；`site_url` 只影響 sitemap 與 canonical link。

## 6. 選項 D – 離線格式

| 格式 | 命令 | 備註 |
|--------|---------|-------|
| 壓縮 HTML | `cd build && zip -r gpu-notes-html.zip site` | 解壓後開啟 `site/index.html`；搜尋也能離線運作 |
| 單一 Markdown | `python3 scripts/build_site.py --bundle` → `build/gpu-programming-notes.md` | 依導覽順序包含所有教學與問題；頁面間連結改成純文字 |
| EPUB | `pandoc build/gpu-programming-notes.md --resource-path=build/site-src --toc -o gpu-notes.epub` | 不需 LaTeX；適合電子書閱讀器與平板 |
| PDF | `pandoc build/gpu-programming-notes.md --resource-path=build/site-src --toc --pdf-engine=xelatex -V geometry:margin=2cm -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" -o gpu-notes.pdf` | 需要 TeX distribution（`texlive-xetex`）。長程式碼行換行效果不佳；橫向（`-V geometry:landscape`）會有幫助 |
| 列印單頁 | 在任意頁面使用瀏覽器的 **Print → Save as PDF** | Material 的 print stylesheet 會隱藏導覽 |

Pages workflow 會將壓縮 HTML、單一 Markdown 與 EPUB 發布到已部署網站的 `downloads/`。

## 7. 以 CI 維持網站可信度

`.github/workflows/ci.yml` 會在每次 push 時執行；CI 顯示紅色代表網站將發布損壞或錯誤內容：

| Job | 保證事項 |
|-----|--------------------|
| `nvcc compile check` | 每個 `solution.cu`、GEMM 教學程式與範例程式都能用真正的 CUDA toolkit 編譯 |
| `cuemu tests (leetgpu / tensara)` | 每個解答都能在 CPU emulator 上產生與平台參考實作相符的結果（見 [tools/cuemu](../tools/cuemu/README.md)） |
| `GEMM tutorial programs (cuemu)` | `tutorials/gemm/` 中每個程式都能在 CPU emulator 通過其 `--test` shape |
| `Chapter 09-13 example programs (cuemu)` | `tutorials/examples/` 中每個程式都能在 CPU emulator 通過檢查 |
| `Chapters 14-15 Triton kernels (interpreter)` | `tutorials/examples/14-triton/` 與 `15-triton-k3/` 的 Triton kernel 在 Triton interpreter 中與 PyTorch 參考結果相符 |
| `AMD tutorial code` | `tutorials/amd/mfma_gemm.hip` 能以 `-Werror` 為 gfx942 編譯 |
| `README index and site build` | README 表格與圖片為最新版本，且網站能在無失效連結的情況下建置 |

## 8. 日後新增內容

```bash
scripts/fetch_upstream.sh && python3 scripts/sync_problems.py   # scaffold new upstream problems
# ...solve, write the README (status: solved)...
python3 tools/cuemu/run_tests.py leetgpu/NNN-new-problem        # test on CPU
python3 scripts/build_index.py                                  # refresh README tables
git commit -am "LeetGPU NNN: …" && git push                     # CI tests, Pages redeploys
```

新的教學章節需要 `tutorials/` 中的 Markdown 檔、`tutorials/README.md` 中的一列，以及 `scripts/build_site.py` 的 `TUTORIAL_PARTS` 內的編號 prefix；後者會將章節分部並排序。導覽會據此產生。

圖片不是手繪：每張圖都是 `scripts/figures/<chapter>.py` 中的一個 Python function，使用 `scripts/figures/svg.py` 的小型 SVG helper。編輯後執行 `python3 scripts/build_figures.py`（或 `make figures`），並 commit 重新產生的 SVG；若圖片過期，CI 會失敗。`scripts/check_figures.py` 會在 headless Chromium 中渲染每張圖，若 label 互相重疊、超出圖片、被線或 box border 穿過，或小於 11 px，就會失敗。Label 未對齊也會失敗：文字沒有在容器 box 或所標示 box 的下方置中，以及同一 box 內（或都在 box 外）的 label 幾乎、卻非完全位於同一欄或 baseline。`make figures` 也會執行此檢查。必須放在線或 grid 上的 label 可用 `plate=True` 加上不透明背景。`tutorials/` 子目錄中的頁面（如 `gemm/`）會在導覽中排在 `scripts/build_site.py` 的 `TUTORIAL_SECTIONS` 所指定章節後。

每篇英文文章都在 `locale/zh-Hant/` 下有相同路徑的繁體中文來源。檔名、相對連結、圖片路徑、程式碼與公式應保持不變，只翻譯文章與可見標籤。Strict build 會回報缺少的翻譯。

## 9. 數學式與程式碼如何渲染

每個問題頁面與教學都以 TeX 撰寫公式，並在每個 display formula 後用表格解釋所有符號。組件如下：

| 組件 | 位置 | 功能 |
|---|---|---|
| `pymdownx.arithmatex`（generic mode） | `mkdocs.yml` | 在 HTML 中保持 `$...$` 與 `$$...$$` 不變，並包在 `<span class="arithmatex">` / `<div class="arithmatex">` 中 |
| KaTeX 0.16 | `site_assets/vendor/katex/` | 在瀏覽器渲染這些 span；包含所需字型 |
| `site_assets/javascripts/katex.js` | Loader | 每次載入頁面時呼叫 `renderMathInElement`，包括 Material instant navigation（`document$.subscribe`），並定義 `\ceil`、`\floor`、`\R` macro |
| Ubuntu Mono、Inter | `site_assets/vendor/fonts/`、`stylesheets/fonts.css` | 自行託管的字型；`theme.font: false` 會阻止 Material 載入 Google Fonts |
| `stylesheets/extra.css` | Theme | 顏色、問題 header、formula block、table，以及使用 Ubuntu Mono 的 code |

不會從 CDN 取得任何內容，因此網站（和壓縮 HTML）能離線及在防火牆後運作。Vendored 檔案約增加 0.8 MB。

符合本站風格的頁面如下：

````markdown
## Formulation

$$
\text{out}_i = \frac{x_i}{\sqrt{\frac{1}{N}\sum_{j=0}^{N-1} x_j^2 + \epsilon}}
$$

| Symbol | Meaning |
|---|---|
| $x_i$ | input element |
| $N$ | row length |
| $\epsilon$ | small constant for numerical stability |
````

渲染結果：

$$
\text{out}_i = \frac{x_i}{\sqrt{\frac{1}{N}\sum_{j=0}^{N-1} x_j^2 + \epsilon}}
$$

| 符號 | 意義 |
|---|---|
| $x_i$ | 輸入 element |
| $N$ | row 長度 |
| $\epsilon$ | 維持數值穩定性的小常數 |

避免 KaTeX 與 Markdown 互相干擾的規則：

- `$$` block 前後保留空白行。
- Display formula 內的行絕不以 `+ `、`- ` 或 `1. ` 開頭：Markdown 會將它解讀為 list item，破壞 block。請把 operator 放在上一行末尾。
- Table cell 內的 math 絕不寫裸露的 `|`（會結束 cell）。使用 `\lvert x \rvert`、`\mid` 或 `\Vert`。
- 不要把 display math 放入有縮排的 list item；先結束 list。
- 用瀏覽器檢查：執行 `python3 -m http.server -d build/site`，尋找紅色 KaTeX error 文字，或以 Playwright 計算 `.katex-error` element。

## 重點整理

1. 網站是產生的：編輯儲存庫中的 README、教學與 `scripts/figures/`，絕不要編輯 `build/`。
2. `scripts/build_site.py --strict` 與 `mkdocs build --strict` 會因失效 link 或 anchor 而失敗；CI 也會檢查 README index、圖片及其可讀性。
3. 透過內附 workflow 發布到 GitHub Pages 是零維護選項；同一個 `build/site/` 目錄也能由任意靜態 host 提供，或壓縮供離線使用。
4. 絕不複製平台問題敘述；只發布摘要、解答與連結。
