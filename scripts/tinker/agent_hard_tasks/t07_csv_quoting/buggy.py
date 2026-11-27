"""Minimal CSV reader/writer with quoting, escaped quotes and embedded newlines."""


def _split_records(text):
    """Split text into records, keeping newlines that sit inside quoted fields."""
    records, buf, in_quotes = [], [], False
    for ch in text:
        if ch == '"':
            in_quotes = not in_quotes
        if ch == "\n" and not in_quotes:
            records.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        records.append("".join(buf))
    return records


def parse_line(line, delimiter=","):
    """Parse one record into a list of field strings."""
    fields, buf = [], []
    in_quotes = False
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < n and line[i + 1] == '"':
                    buf.append('"')
                    i += 1
                    continue
                in_quotes = False
            else:
                buf.append(ch)
        elif ch == '"':
            in_quotes = True
        elif ch == delimiter:
            fields.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        fields.append("".join(buf))
    return fields


def parse(text, delimiter=","):
    """All records of a document; blank lines are skipped."""
    return [parse_line(rec.rstrip("\r"), delimiter) for rec in _split_records(text) if rec.strip("\r") != ""]


def read_dicts(text, delimiter=","):
    rows = parse(text, delimiter)
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    out = []
    for row in body:
        if len(row) > len(header):
            raise ValueError(f"row has {len(row)} fields, header has {len(header)}")
        padded = row + [""] * (len(header) - len(row))
        out.append(dict(zip(header, padded)))
    return out


def _needs_quotes(text, delimiter):
    if text != text.strip():
        return True
    return any(c in text for c in (delimiter, "\n", "\r"))


def _quote(field, delimiter):
    text = "" if field is None else str(field)
    if _needs_quotes(text, delimiter):
        return '"' + text.replace('"', '""') + '"'
    return text


def write_row(fields, delimiter=","):
    return delimiter.join(_quote(f, delimiter) for f in fields)


def to_text(rows, delimiter=","):
    return "\n".join(write_row(r, delimiter) for r in rows)


def write_dicts(records, columns, delimiter=","):
    rows = [list(columns)]
    for rec in records:
        rows.append([rec.get(col, "") for col in columns])
    return to_text(rows, delimiter)


def column(text, name, delimiter=","):
    return [rec[name] for rec in read_dicts(text, delimiter)]


def sniff_delimiter(sample, candidates=",;\t|"):
    """Pick the candidate that splits the first record into the most fields."""
    first = _split_records(sample)[0] if sample else ""
    best, best_count = ",", 1
    for cand in candidates:
        count = len(parse_line(first, cand))
        if count > best_count:
            best, best_count = cand, count
    return best
