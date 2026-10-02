"""Build the study guide as a static website (GitHub Pages).

Turns README.md and docs/*.md into HTML pages with a chapter sidebar,
rendered equations (MathJax), Mermaid diagrams and click-to-zoom figures.

    pip install markdown-it-py mdit-py-plugins
    python tools/build_site.py            # writes _site/
    python tools/build_site.py OUT_DIR    # writes somewhere else
"""
import html
import re
import shutil
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "_site"
REPO_URL = "https://github.com/Tariq-15/temp111"

THESIS_TITLE = ("Evaluating Selective Unlearning for Temporal Regression Models: "
                "CP-FIM and Retrain-Referenced Evidence on Financial and Energy Time Series")
SHORT = {1: "Big picture", 2: "Literature", 3: "Data", 4: "Models", 5: "CP-FIM", 6: "Methods",
         7: "Metrics", 8: "Results", 9: "Fixes, stats", 10: "Conclusion", 11: "Cheat sheet",
         12: "Viva Q&A", 13: "Figures"}
AUTHORS = ["Md. Tariquzzaman", "Anirban Saha", "Sharaf Binte Younus", "Sadia Binte Kamal"]


# --------------------------------------------------------------------------- helpers
def slugify(text):
    """GitHub-style heading anchors, so links written for GitHub keep working."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def build_md():
    md = MarkdownIt("commonmark", {"html": True, "typographer": False})
    md.enable("table").enable("strikethrough")
    md.use(anchors_plugin, min_level=1, max_level=4, slug_func=slugify,
           permalink=True, permalinkSymbol="#", permalinkBefore=False)
    return md


MD = build_md()


def protect_math(src):
    """Swap math for placeholders so Markdown does not touch backslashes or underscores."""
    store = []

    def keep(kind, body):
        store.append((kind, body))
        return len(store) - 1

    def block(m):
        return f"\n\nMJXBLOCK{keep('block', m.group(1).strip())}X\n\n"

    src = re.sub(r"```math\n(.*?)```", block, src, flags=re.S)
    # leave other fenced code alone
    parts = re.split(r"(```.*?```)", src, flags=re.S)
    for i, part in enumerate(parts):
        if part.startswith("```"):
            continue
        parts[i] = re.sub(r"(?<![\\$])\$([^\s$][^$\n]*?)\$(?!\$)",
                          lambda m: f"MJXINL{keep('inline', m.group(1))}X", part)
    return "".join(parts), store


def restore_math(out, store):
    def block(m):
        return f'<div class="math-block">\\[{html.escape(store[int(m.group(1))][1])}\\]</div>'

    out = re.sub(r"<p>MJXBLOCK(\d+)X</p>", block, out)
    return re.sub(r"MJXINL(\d+)X",
                  lambda m: f'<span class="math">\\({html.escape(store[int(m.group(1))][1])}\\)</span>', out)


def rewrite_links(out, page_rel):
    """.md links -> .html; README -> index.html; non-site files -> GitHub."""
    page_dir = (ROOT / page_rel).parent

    def fix(m):
        attr, url = m.group(1), m.group(2)
        if url.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        path, _, frag = url.partition("#")
        frag = "#" + frag if frag else ""
        target = (page_dir / path).resolve()
        if path.endswith(".md"):
            new = re.sub(r"README\.md$", "index.html", path)
            new = re.sub(r"\.md$", ".html", new)
            return f'{attr}="{new}{frag}"'
        if target.suffix.lower() in (".png", ".svg", ".jpg") or "figures" in target.parts:
            return m.group(0)
        rel = target.relative_to(ROOT).as_posix()
        return f'{attr}="{REPO_URL}/blob/main/{rel}{frag}"'

    return re.sub(r'(href|src)="([^"]+)"', fix, out)


def make_figure(img, caption=""):
    src = re.search(r'src="([^"]+)"', img).group(1)
    capt = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><a class="zoom" href="{src}">{img}</a>{capt}</figure>'


def figures(out):
    # image with an italic caption line, in the same paragraph or the next one
    out = re.sub(r"<p>(<img [^>]+>)\s*<em>(.*?)</em></p>",
                 lambda m: make_figure(m.group(1), m.group(2)), out, flags=re.S)
    out = re.sub(r"<p>(<img [^>]+>)</p>\s*<p><em>(.*?)</em></p>",
                 lambda m: make_figure(m.group(1), m.group(2)), out, flags=re.S)
    # images inside table cells: make them zoomable too
    out = re.sub(r"<td>(<img [^>]+>)</td>",
                 lambda m: f'<td><a class="zoom" href="{re.search(r"src=.([^\"]+)", m.group(1)).group(1)}">{m.group(1)}</a></td>', out)
    # several images stacked in one paragraph, or one image alone
    return re.sub(r"<p>((?:<img [^>]+>\s*)+)</p>",
                  lambda m: "".join(make_figure(i) for i in re.findall(r"<img [^>]+>", m.group(1))), out)


def post(out, store, page_rel):
    out = restore_math(out, store)
    out = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
                 r'<div class="diagram"><pre class="mermaid">\1</pre></div>', out, flags=re.S)
    out = out.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    out = figures(out)
    out = rewrite_links(out, page_rel)
    out = out.replace('<img src=', '<img loading="lazy" src=')
    return out


# --------------------------------------------------------------------------- pages
def read_pages():
    pages = []
    for f in sorted((ROOT / "docs").glob("*.md")):
        text = f.read_text(encoding="utf-8")
        h1 = re.search(r"^# (\d+)\. (.+)$", text, flags=re.M)
        sub = ""
        nav = re.search(r"^\*(.*?)←.*\*\s*$", text, flags=re.M)
        if nav:
            sub = nav.group(1).strip().rstrip(".").strip()
            text = text.replace(nav.group(0), "", 1)
        text = re.sub(r"^# .+\n", "", text, count=1, flags=re.M)
        text = re.sub(r"^---\s*\n", "", text.lstrip("\n"), count=1)  # the page header already has a rule
        pages.append({"src": f"docs/{f.name}", "out": f"docs/{f.stem}.html", "num": int(h1.group(1)),
                      "title": h1.group(2).strip(), "sub": sub, "body": text})
    return pages


def windows_svg(pages, current, prefix, big=False):
    """Chapter navigation drawn as overlapping sliding windows over a time axis."""
    stride, width, lane_h, gap = 44, 124, 26, 7
    n = len(pages)
    total_w = stride * (n - 1) + width
    lanes = 3
    axis_y = lanes * (lane_h + gap) + 6
    h = axis_y + 14
    parts = [f'<svg class="windows{" big" if big else ""}" viewBox="-2 -2 {total_w + 4} {h + 4}" '
             f'role="navigation" aria-label="Chapters, drawn as overlapping sliding windows">']
    parts.append(f'<line class="axis" x1="0" y1="{axis_y}" x2="{total_w}" y2="{axis_y}"/>')
    for t in range(0, total_w + 1, 22):
        parts.append(f'<line class="tick" x1="{t}" y1="{axis_y - 3}" x2="{t}" y2="{axis_y + 3}"/>')
    for i, p in enumerate(pages):
        x = i * stride
        y = (i % lanes) * (lane_h + gap)
        state = "current" if p["num"] == current else ("done" if current and p["num"] < current else "")
        href = prefix + Path(p["out"]).name
        aria_current = ' aria-current="page"' if state == "current" else ""
        parts.append(
            f'<a href="{href}" class="win {state}" aria-label="Chapter {p["num"]}: {html.escape(p["title"])}"'
            f'{aria_current}>'
            f'<title>{p["num"]}. {html.escape(p["title"])}</title>'
            f'<rect x="{x}" y="{y}" width="{width}" height="{lane_h}" rx="6"/>'
            f'<text x="{x + 12}" y="{y + lane_h / 2 + 4.5}">{p["num"]}'
            f'<tspan class="lbl" dx="8">{html.escape(SHORT.get(p["num"], ""))}</tspan></text></a>')
    parts.append("</svg>")
    return "".join(parts)


FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&"
         "family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&"
         "family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap")


def page_html(title, body, pages, current, depth, sub="", hero=""):
    up = "../" * depth
    doc_prefix = "" if depth else "docs/"
    side = [f'<a class="side-home{" active" if current == 0 else ""}" href="{up}index.html">Overview</a>', "<ol>"]
    for p in pages:
        href = f"{up}{p['out']}"
        act = ' class="active" aria-current="page"' if p["num"] == current else ""
        side.append(f'<li><a href="{href}"{act}><span class="n">{p["num"]}</span>{html.escape(p["title"])}</a></li>')
    side.append("</ol>")
    prev_next = ""
    if current:
        idx = current - 1
        prev = pages[idx - 1] if idx > 0 else None
        nxt = pages[idx + 1] if idx + 1 < len(pages) else None
        pv = (f'<a class="pn prev" href="{up}{prev["out"]}"><span>Previous</span>{prev["num"]}. {html.escape(prev["title"])}</a>'
              if prev else f'<a class="pn prev" href="{up}index.html"><span>Previous</span>Overview</a>')
        nx = (f'<a class="pn next" href="{up}{nxt["out"]}"><span>Next</span>{nxt["num"]}. {html.escape(nxt["title"])}</a>'
              if nxt else f'<a class="pn next" href="{up}index.html"><span>Back to</span>Overview</a>')
        prev_next = f'<nav class="prevnext" aria-label="Chapter navigation">{pv}{nx}</nav>'
    strip = "" if current == 0 else f'<div class="strip">{windows_svg(pages, current, "")}</div>'
    header = ""
    if current:
        header = (f'<header class="chapter-head"><p class="chapter-num">Chapter {current} of {len(pages)}</p>'
                  f'<h1>{html.escape(title)}</h1>' + (f'<p class="chapter-sub">{html.escape(sub)}</p>' if sub else "")
                  + "</header>")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} | CP-FIM thesis study guide</title>
<meta name="description" content="Illustrated study guide for the BRAC University thesis on selective unlearning for temporal regression (CP-FIM).">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Crect x=%272%27 y=%274%27 width=%2718%27 height=%278%27 rx=%272%27 fill=%27%23b9d3f5%27/%3E%3Crect x=%277%27 y=%2712%27 width=%2718%27 height=%278%27 rx=%272%27 fill=%27%232a78d6%27/%3E%3Crect x=%2712%27 y=%2720%27 width=%2718%27 height=%278%27 rx=%272%27 fill=%27%23b9d3f5%27/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{up}assets/site.css">
<script>try{{var t=localStorage.getItem('theme');if(t)document.documentElement.dataset.theme=t;}}catch(e){{}}</script>
<script>
window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},
options:{{ignoreHtmlClass:'mermaid'}},chtml:{{scale:0.98}}}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="shell">
<aside class="side" id="side">
  <a class="brand" href="{up}index.html">CP-FIM thesis<br><span>study guide</span></a>
  <nav aria-label="Chapters">{''.join(side)}</nav>
  <div class="side-foot">
    <button type="button" class="theme" id="theme" aria-pressed="false">Dark mode</button>
    <a href="{REPO_URL}">Source on GitHub</a>
  </div>
</aside>
<div class="main-wrap">
  <div class="topbar">
    <button type="button" class="menu" id="menu" aria-controls="side" aria-expanded="false">Chapters</button>
    <a class="top-brand" href="{up}index.html">CP-FIM thesis study guide</a>
  </div>
  <main id="main" class="content">
  {hero}{strip}{header}
  {body}
  {prev_next}
  </main>
  <footer class="foot">
    <p>Study aid for the B.Sc. thesis at Brac University, October 2026. The thesis PDF is the authoritative source; every number here is copied from it.</p>
  </footer>
</div>
</div>
<dialog id="zoom" aria-label="Enlarged figure"><img alt=""><p></p><button type="button" class="close">Close</button></dialog>
<script type="module" src="{up}assets/site.js"></script>
</body>
</html>
"""


def home(pages, readme):
    body = readme
    body = re.sub(r"^# .+\n", "", body, count=1, flags=re.M)
    body = re.sub(r"^\*\*(Thesis title|Authors|Supervisor|Degree):\*\*.*\n", "", body, flags=re.M)
    body = body.replace("This repository explains", "This guide explains", 1)
    hero = f"""<section class="hero">
  <p class="kicker">B.Sc. thesis, Department of Computer Science and Engineering, Brac University, October 2026</p>
  <h1 class="hero-title">{html.escape(THESIS_TITLE).replace("CP-FIM", "CP‑FIM")}</h1>
  <p class="hero-authors">{", ".join(AUTHORS[:-1])} and {AUTHORS[-1]}</p>
  <p class="hero-sup">Supervised by Dr. Md. Golam Rabiul Alam, Professor, Department of CSE</p>
  <div class="hero-strip">
    {windows_svg(pages, 0, "docs/", big=True)}
    <p class="strip-label">Thirteen chapters, drawn the way the thesis draws its training data: as overlapping sliding windows. Pick any window to start reading.</p>
  </div>
</section>"""
    return body, hero


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for child in OUT.iterdir():  # empty the folder but keep it (a preview server may be using it)
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    (OUT / "docs").mkdir()
    shutil.copytree(ROOT / "figures", OUT / "figures")
    shutil.copytree(ROOT / "tools" / "site_assets", OUT / "assets")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    pages = read_pages()
    for p in pages:
        src, store = protect_math(p["body"])
        body = post(MD.render(src), store, p["src"])
        (OUT / p["out"]).write_text(page_html(p["title"], body, pages, p["num"], 1, p["sub"]), encoding="utf-8")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    body_md, hero = home(pages, readme)
    src, store = protect_math(body_md)
    body = post(MD.render(src), store, "README.md")
    (OUT / "index.html").write_text(page_html("Overview", body, pages, 0, 0, hero=hero), encoding="utf-8")
    print(f"built {len(pages) + 1} pages into {OUT}")


if __name__ == "__main__":
    main()
