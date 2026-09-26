/* Accessible reading controls shared by the English and Traditional Chinese sites. */
(() => {
  const root = document.documentElement;
  const key = "gpu-notes-reading-preferences";
  const defaults = { theme: "system", scale: "md", width: "standard", spacing: "normal" };
  const options = {
    scale: ["sm", "md", "lg", "xl"],
    width: ["narrow", "standard", "wide"],
    spacing: ["compact", "normal", "relaxed"],
  };

  function load() {
    try {
      return { ...defaults, ...JSON.parse(localStorage.getItem(key) || "{}") };
    } catch (_) {
      return { ...defaults };
    }
  }

  let prefs = load();

  function save() {
    localStorage.setItem(key, JSON.stringify(prefs));
  }

  function apply() {
    root.dataset.readingScale = prefs.scale;
    root.dataset.readingWidth = prefs.width;
    root.dataset.readingSpacing = prefs.spacing;
    root.lang = location.pathname.includes("/zh-Hant/") ? "zh-Hant" : "en";

    const mediaDark = matchMedia("(prefers-color-scheme: dark)").matches;
    const scheme = prefs.theme === "system" ? (mediaDark ? "slate" : "default") : prefs.theme;
    const palette = document.querySelector(`input[data-md-color-scheme="${scheme}"]`);
    if (palette && !palette.checked) palette.click();
  }

  function languageUrl() {
    const marker = "/zh-Hant/";
    if (location.pathname.includes(marker)) {
      return location.pathname.replace(marker, "/") + location.search + location.hash;
    }
    const home = document.querySelector(".md-header__button.md-logo")?.href || `${location.origin}/`;
    const base = new URL(home).pathname.replace(/\/?$/, "/");
    const relative = location.pathname.startsWith(base)
      ? location.pathname.slice(base.length)
      : location.pathname.slice(1);
    return `${base}zh-Hant/${relative}${location.search}${location.hash}`;
  }

  function control(label, values, get, set) {
    const group = document.createElement("div");
    group.className = "reading-controls__group";
    const title = document.createElement("span");
    title.textContent = label;
    group.append(title);
    const buttons = document.createElement("div");
    buttons.className = "reading-controls__choices";
    values.forEach(([value, text]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = text;
      button.dataset.value = value;
      button.setAttribute("aria-pressed", String(get() === value));
      button.addEventListener("click", () => {
        set(value);
        save();
        apply();
        buttons.querySelectorAll("button").forEach((item) =>
          item.setAttribute("aria-pressed", String(item.dataset.value === value)));
      });
      buttons.append(button);
    });
    group.append(buttons);
    return group;
  }

  function mount() {
    document.querySelector(".reading-controls")?.remove();
    const zh = location.pathname.includes("/zh-Hant/");
    const host = document.createElement("div");
    host.className = "reading-controls";

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "reading-controls__toggle";
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-label", zh ? "調整閱讀外觀" : "Reading appearance");
    toggle.innerHTML = '<span aria-hidden="true">Aa</span>';

    const panel = document.createElement("div");
    panel.className = "reading-controls__panel";
    panel.hidden = true;
    const heading = document.createElement("strong");
    heading.textContent = zh ? "閱讀設定" : "Reading settings";
    panel.append(heading);

    panel.append(
      control(zh ? "主題" : "Theme",
        [["system", zh ? "系統" : "System"], ["default", zh ? "淺色" : "Light"], ["slate", zh ? "深色" : "Dark"]],
        () => prefs.theme, (v) => { prefs.theme = v; }),
      control(zh ? "字體大小" : "Text size",
        [["sm", "A−"], ["md", "A"], ["lg", "A+"], ["xl", "A++"]],
        () => prefs.scale, (v) => { prefs.scale = v; }),
      control(zh ? "內容寬度" : "Page width",
        [["narrow", zh ? "窄" : "Narrow"], ["standard", zh ? "標準" : "Standard"], ["wide", zh ? "寬" : "Wide"]],
        () => prefs.width, (v) => { prefs.width = v; }),
      control(zh ? "行距" : "Line spacing",
        [["compact", zh ? "緊密" : "Tight"], ["normal", zh ? "標準" : "Normal"], ["relaxed", zh ? "寬鬆" : "Relaxed"]],
        () => prefs.spacing, (v) => { prefs.spacing = v; }),
    );

    const language = document.createElement("a");
    language.className = "reading-controls__language";
    language.href = languageUrl();
    language.textContent = zh ? "Read in English" : "閱讀繁體中文版";
    panel.append(language);

    toggle.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
      toggle.setAttribute("aria-expanded", String(!panel.hidden));
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !panel.hidden) {
        panel.hidden = true;
        toggle.setAttribute("aria-expanded", "false");
        toggle.focus();
      }
    }, { once: true });

    host.append(toggle, panel);
    document.body.append(host);
    apply();
  }

  apply();
  if (typeof document$ !== "undefined") document$.subscribe(mount);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (prefs.theme === "system") apply();
  });
})();
