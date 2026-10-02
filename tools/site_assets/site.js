// Theme switch, mobile chapter menu, figure zoom and Mermaid diagrams.
const root = document.documentElement;
const media = window.matchMedia("(prefers-color-scheme: dark)");

function effectiveTheme() {
  return root.dataset.theme || (media.matches ? "dark" : "light");
}
function syncThemeButton() {
  const btn = document.getElementById("theme");
  if (!btn) return;
  const dark = effectiveTheme() === "dark";
  btn.textContent = dark ? "Light mode" : "Dark mode";
  btn.setAttribute("aria-pressed", String(dark));
}
document.getElementById("theme")?.addEventListener("click", () => {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) { /* storage may be blocked */ }
  syncThemeButton();
});
media.addEventListener?.("change", syncThemeButton);
syncThemeButton();

// Mobile: chapters drawer
const side = document.getElementById("side");
const menu = document.getElementById("menu");
menu?.addEventListener("click", () => {
  const open = side.classList.toggle("open");
  menu.setAttribute("aria-expanded", String(open));
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && side?.classList.contains("open")) {
    side.classList.remove("open");
    menu?.setAttribute("aria-expanded", "false");
    menu?.focus();
  }
});
document.addEventListener("click", (e) => {
  if (side?.classList.contains("open") && !side.contains(e.target) && e.target !== menu) {
    side.classList.remove("open");
    menu?.setAttribute("aria-expanded", "false");
  }
});

// Figure zoom: open the full-size image in a dialog instead of leaving the page
const dialog = document.getElementById("zoom");
if (dialog && typeof dialog.showModal === "function") {
  const img = dialog.querySelector("img");
  const cap = dialog.querySelector("p");
  document.querySelectorAll("a.zoom, td a[href$='.png']").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const fig = a.closest("figure");
      img.src = a.getAttribute("href");
      img.alt = a.querySelector("img")?.alt || "";
      cap.textContent = fig?.querySelector("figcaption")?.textContent || img.alt;
      dialog.showModal();
    });
  });
  dialog.addEventListener("click", () => dialog.close());
}

// Mermaid diagrams (drawn on a light plate in both themes, like the figures)
if (document.querySelector("pre.mermaid")) {
  const { default: mermaid } = await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs");
  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    securityLevel: "strict",
    fontFamily: "IBM Plex Sans, Segoe UI, sans-serif",
    themeVariables: {
      fontSize: "15px",
      primaryColor: "#e6effb",
      primaryBorderColor: "#2a78d6",
      primaryTextColor: "#14213d",
      secondaryColor: "#e2f5ee",
      tertiaryColor: "#f6f8fb",
      lineColor: "#5d6b82",
      textColor: "#14213d",
      clusterBkg: "#f6f8fb",
      clusterBorder: "#b9c3d4",
      edgeLabelBackground: "#ffffff",
    },
    flowchart: { htmlLabels: true, curve: "basis" },
  });
  await mermaid.run({ querySelector: "pre.mermaid" });
}
