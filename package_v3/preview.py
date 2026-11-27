#!/usr/bin/env python3
"""One self-contained HTML page with every rendered card, launch draft and chart, for a quick read-through.

  python3 package_v3/preview.py [--dist package_v3/dist] [--charts package_v3/charts] [--out package_v3/preview.html]

Markdown is converted by the small renderer below (headings, paragraphs, lists, quotes, code, tables, links, images,
bold, inline code): no external packages and no external assets. Chart images whose URL ends in charts/<name>.png
are inlined from --charts (SVG, base64 data URI; PNG when no SVG exists). Light and dark aware.
"""
from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ("Inkling-2-Mini-{p}", "Inkling-2-Mini-{p}-GGUF", "Inkling-2-Mini-{p}-LoRA", "Inkling-2-Mini-Unbound",
         "Inkling-2-Mini-Unbound-GGUF", "Inkling-2-Mini-Unbound-Basalt", "github")


# ---- inline markdown -----------------------------------------------------------------
class Images:
    def __init__(self, charts: Path) -> None:
        self.charts, self.cache = charts, {}

    def src(self, url: str) -> str:
        m = re.search(r"charts/([A-Za-z0-9_\-]+)\.(png|svg)$", url)
        if not m:
            return ""
        name = m.group(1)
        if name not in self.cache:
            svg, png = self.charts / f"{name}.svg", self.charts / f"{name}.png"
            if svg.exists():
                self.cache[name] = "data:image/svg+xml;base64," + base64.b64encode(svg.read_bytes()).decode()
            elif png.exists():
                self.cache[name] = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
            else:
                self.cache[name] = ""
        return self.cache[name]


def inline(text: str, imgs: Images) -> str:
    codes: list[str] = []

    def keep_code(m: re.Match[str]) -> str:
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    t = re.sub(r"`([^`]+)`", keep_code, text)
    t = html.escape(t, quote=False)

    def img(m: re.Match[str]) -> str:
        src = imgs.src(m.group(2))
        alt = html.escape(m.group(1))
        return f'<img alt="{alt}" src="{src}">' if src else f'<span class="missing">[image not found: {alt}]</span>'

    t = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", img, t)
    t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    return re.sub("\x00(\\d+)\x00", lambda m: codes[int(m.group(1))], t)


# ---- block markdown --------------------------------------------------------------------
def table(lines: list[str], imgs: Images) -> str:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if not re.match(r"^\|[-|: ]+\|$", ln.strip())]
    head, body = rows[0], rows[1:]
    out = ["<div class='tw'><table><thead><tr>" + "".join(f"<th>{inline(c, imgs)}</th>" for c in head) + "</tr></thead><tbody>"]
    out += ["<tr>" + "".join(f"<td>{inline(c, imgs)}</td>" for c in r) + "</tr>" for r in body]
    return "\n".join(out) + "</tbody></table></div>"


def md(text: str, imgs: Images) -> str:
    if text.startswith("---\n"):
        text = text[text.index("\n---\n", 4) + 5:]
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
        elif m := re.match(r"^(#{1,4}) (.*)", ln):
            lvl = len(m.group(1)) + 1
            out.append(f"<h{lvl}>{inline(m.group(2), imgs)}</h{lvl}>")
            i += 1
        elif ln.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(table(lines[i:j], imgs))
            i = j
        elif ln.startswith(">"):
            j = i
            while j < len(lines) and lines[j].startswith(">"):
                j += 1
            out.append("<blockquote>" + md("\n".join(x[1:].lstrip() for x in lines[i:j]), imgs) + "</blockquote>")
            i = j
        elif re.match(r"^\s*(- |\d+\. )", ln):
            j, items = i, []
            ordered = bool(re.match(r"^\s*\d+\. ", ln))
            while j < len(lines) and (re.match(r"^\s*(- |\d+\. )", lines[j]) or (lines[j].startswith("  ") and items)):
                if re.match(r"^\s*(- |\d+\. )", lines[j]) and not lines[j].startswith("    "):
                    items.append(re.sub(r"^\s*(- |\d+\. )", "", lines[j]))
                else:
                    items[-1] += " " + lines[j].strip()
                j += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x, imgs)}</li>" for x in items) + f"</{tag}>")
            i = j
        elif not ln.strip():
            i += 1
        else:
            j = i
            while j < len(lines) and lines[j].strip() and not re.match(r"^(#|\||>|```|\s*- |\s*\d+\. )", lines[j]):
                j += 1
            out.append("<p>" + inline(" ".join(x.strip() for x in lines[i:j]), imgs) + "</p>")
            i = max(j, i + 1)
    return "\n".join(out)


CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e6e5e1;--acc:#2a78d6;--code:#f1f0ec;--warn:#b4472f}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe7;--mut:#a9a8a3;--line:#33322f;--acc:#6fa3e6;--code:#22211f;--warn:#e07a62}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;z-index:2;overflow-x:auto;white-space:nowrap}
nav a{margin-right:14px;color:var(--acc);text-decoration:none;font-size:13px}
main{max-width:1100px;margin:0 auto;padding:16px}
section{border:1px solid var(--line);border-radius:8px;padding:8px 20px 20px;margin:24px 0}
.label{font:600 12px/1 monospace;color:var(--mut);margin-top:12px}
h2{font-size:24px}h3{font-size:19px;margin-top:28px}h4,h5{font-size:16px}
a{color:var(--acc)}code{background:var(--code);padding:1px 4px;border-radius:4px;font-size:13px}
pre{background:var(--code);padding:10px;border-radius:6px;overflow-x:auto}pre code{padding:0}
blockquote{border-left:3px solid var(--warn);margin:12px 0;padding:2px 14px;color:var(--fg)}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:13px;margin:10px 0}
th,td{border:1px solid var(--line);padding:4px 8px;vertical-align:top}th{background:var(--code);text-align:left}
img{max-width:100%;height:auto;background:#fcfcfb;border-radius:6px;margin:8px 0}.missing{color:var(--warn)}
"""


def page(sections: list[tuple[str, str, str]]) -> str:
    nav = "".join(f'<a href="#{sid}">{html.escape(title)}</a>' for sid, title, _ in sections)
    body = "".join(f'<section id="{sid}"><div class="label">{html.escape(title)}</div>{content}</section>'
                   for sid, title, content in sections)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>v3 package preview</title><style>{CSS}</style>"
            f"</head><body><nav>{nav}</nav><main>{body}</main></body></html>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", default=str(HERE / "dist"))
    ap.add_argument("--charts", default=str(HERE / "charts"))
    ap.add_argument("--docs", default=str(HERE), help="folder holding CHANGELOG.md and RELEASE_NOTES.md")
    ap.add_argument("--out", default=str(HERE / "preview.html"))
    a = ap.parse_args()
    dist, imgs = Path(a.dist), Images(Path(a.charts))
    prime = next((p.name.removeprefix("Inkling-2-Mini-") for p in dist.glob("Inkling-2-Mini-*-LoRA")), "Prime")
    prime = prime.removesuffix("-LoRA")
    files = [dist / o.format(p=prime) / "README.md" for o in ORDER]
    files += [Path(a.docs) / n for n in ("RELEASE_NOTES.md", "CHANGELOG.md")] + sorted((dist / "launch").glob("*.md"))
    sections = []
    for f in files:
        if f.exists():
            title = f.parent.name if f.name == "README.md" else f.name
            sid = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            sections.append((sid, title, md(f.read_text(), imgs)))
    charts = "".join(f'<h3>{n}</h3><img alt="{n}" src="{imgs.src("charts/" + n + ".svg")}">'
                     for n in sorted({p.stem for p in Path(a.charts).glob("*.svg")}))
    sections.append(("all-charts", "all charts", charts))
    Path(a.out).write_text(page(sections))
    print(f"wrote {a.out} ({len(sections)} sections, {Path(a.out).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
