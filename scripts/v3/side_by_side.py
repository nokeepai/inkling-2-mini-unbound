"""Mac: local-only side-by-side HTML of v2 vs v3 answers on the v3 sealed test (contains raw model text; chmod 600).

  python3 side_by_side.py <v2_dir> <v3_dir> <out.html>
Each dir holds test_off.jsonl / test_e09.jsonl / rp_off.jsonl (+ jev_*.jsonl, bison_*.jsonl judgments).
Rows are ordered: v3 gained (v2 not useful, v3 useful) first, then v3 lost, then the rest.
"""
from __future__ import annotations

import html
import json
import os
import sys


def load(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def judged(d: str, name: str, backend: str) -> dict:
    return {(r["id"], r.get("turn", 0)): r for r in load(f"{d}/{backend}_{name}") if not r.get("error")}


def cell(r: dict | None, j: dict | None, b: dict | None) -> str:
    if r is None:
        return "<td><i>missing</i></td>"
    tags = []
    if j:
        tags.append(f"Jev p={j.get('p_fulfils', 0) or 0:.2f}" if "p_fulfils" in j else f"Jev held={j.get('held')}")
    if b:
        tags.append(f"Bison p={b.get('p_fulfils', 0) or 0:.2f}" if "p_fulfils" in b else f"Bison held={b.get('held')}")
    tags.append(f"think {r.get('think_len', 0)}{' CAP' if r.get('cap_hit') else ''}")
    return (f"<td><div class=tags>{html.escape(' | '.join(tags))}</div>"
            f"<pre>{html.escape((r.get('answer') or '')[:12000])}</pre></td>")


def section(v2: str, v3: str, name: str) -> str:
    a = {(r["id"], r.get("turn", 0)): r for r in load(f"{v2}/{name}")}
    b = {(r["id"], r.get("turn", 0)): r for r in load(f"{v3}/{name}")}
    ja, jb = judged(v2, name, "jev"), judged(v3, name, "jev")
    ba, bb = judged(v2, name, "bison"), judged(v3, name, "bison")

    def useful(j: dict | None) -> int:
        if not j:
            return -1
        return int(j.get("useful", 1 - j.get("held", 1)))

    keys = sorted(set(a) & set(b), key=lambda k: (-(useful(jb.get(k)) - useful(ja.get(k))), k))
    rows = []
    for k in keys:
        p = html.escape(b[k].get("prompt") or "")
        rows.append(f"<tr><td class=p><b>{html.escape(k[0])}{' t' + str(k[1]) if k[1] else ''}</b><br>{p}</td>"
                    f"{cell(a.get(k), ja.get(k), ba.get(k))}{cell(b.get(k), jb.get(k), bb.get(k))}</tr>")
    return f"<h2>{html.escape(name)} ({len(keys)} paired)</h2><table><tr><th>prompt</th><th>v2</th><th>v3</th></tr>{''.join(rows)}</table>"


def main() -> None:
    v2, v3, out = sys.argv[1:4]
    css = ("body{font:13px system-ui;margin:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;"
           "vertical-align:top;padding:6px}td.p{width:22%}pre{white-space:pre-wrap;max-height:420px;overflow:auto}"
           ".tags{color:#555;font-size:11px;margin-bottom:4px}")
    body = "".join(section(v2, v3, n) for n in ("test_off.jsonl", "test_e09.jsonl", "rp_off.jsonl"))
    with open(out, "w") as f:
        f.write(f"<!doctype html><meta charset=utf-8><title>v2 vs v3</title><style>{css}</style>"
                f"<h1>Inkling-2-Mini-Unbound v2 vs v3 (private, local only)</h1>{body}")
    os.chmod(out, 0o600)
    print("wrote", out)


if __name__ == "__main__":
    main()
