"""Render rows as a GitHub-style markdown table and parse such tables back."""


def escape_cell(value):
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def unescape_cell(text):
    return text.replace("\\|", "|")


def _pad(text, width, align):
    """Pad to width; centred text puts any odd leftover space on the right."""
    gap = width - len(text)
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def _rule(width, align):
    if align == "center":
        return ":" + "-" * (width - 2) + ":"
    if align == "right":
        return "-" * (width - 1) + ":"
    if align == "left":
        return ":" + "-" * (width - 1)
    return "-" * width


def column_widths(headers, rows, minimum=3):
    widths = [max(minimum, len(escape_cell(h))) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(escape_cell(cell)))
    return widths


def _normalize_rows(headers, rows):
    """Copy rows, padding short ones with empty cells."""
    out = []
    for row in rows:
        if len(row) > len(headers):
            raise ValueError("row has more cells than headers")
        out.append(list(row) + [""] * (len(headers) - len(row)))
    return out


def format_table(headers, rows, align=None):
    align = list(align) if align else [None] * len(headers)
    if len(align) != len(headers):
        raise ValueError("one alignment per column")
    rows = _normalize_rows(headers, rows)
    widths = column_widths(headers, rows)

    def line(cells):
        padded = [_pad(escape_cell(c), w, a) for c, w, a in zip(cells, widths, align)]
        return "| " + " | ".join(padded) + " |"

    lines = [line(headers), "| " + " | ".join(_rule(w, a) for w, a in zip(widths, align)) + " |"]
    lines.extend(line(r) for r in rows)
    return "\n".join(lines)


def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, buf, i = [], [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            buf.append("\\|")
            i += 2
            continue
        if line[i] == "|":
            cells.append("".join(buf))
            buf = []
        else:
            buf.append(line[i])
        i += 1
    cells.append("".join(buf))
    return [unescape_cell(c.strip()) for c in cells]


def _alignment(rule):
    rule = rule.strip()
    if rule.startswith(":") and rule.endswith(":"):
        return "center"
    if rule.endswith(":"):
        return "right"
    if rule.startswith(":"):
        return "left"
    return None


def parse_table(md):
    lines = [ln for ln in md.strip().splitlines() if ln.strip()]
    headers = split_row(lines[0])
    aligns = [_alignment(r) for r in split_row(lines[1])]
    rows = [split_row(ln) for ln in lines[2:]]
    return headers, rows, aligns


def from_dicts(records, columns=None, align=None):
    columns = list(columns) if columns else sorted({k for rec in records for k in rec})
    rows = [[rec.get(col) for col in columns] for rec in records]
    return format_table(columns, rows, align)


def column_totals(headers, rows, numeric):
    """Append a totals row summing the named numeric columns."""
    totals = []
    for i, h in enumerate(headers):
        if h in numeric:
            totals.append(sum(float(r[i]) for r in rows if i < len(r) and r[i] not in ("", None)))
        else:
            totals.append("total" if i == 0 else "")
    return rows + [[int(t) if isinstance(t, float) and t.is_integer() else t for t in totals]]
