"""Greedy text wrapping, full justification, centering and truncation."""


def split_words(text):
    return [w for w in text.split(" ") if w]


def wrap(text, width):
    """Greedy wrap; words longer than width are split into width-sized chunks."""
    words = split_words(text)
    lines = []
    current = []
    length = 0
    for word in words:
        if len(word) > width:
            if current:
                lines.append(current)
                current, length = [], 0
            for start in range(0, len(word), width):
                lines.append([word[start:start + width]])
            continue
        needed = len(word) if not current else length + 1 + len(word)
        if needed <= width:
            current.append(word)
            length = needed
        else:
            lines.append(current)
            current = [word]
            length = len(word)
    if current:
        lines.append(current)
    return [" ".join(line) for line in lines]


def justify_line(line, width):
    """Spread spaces between words; leftmost gaps get the extra spaces."""
    words = line.split(" ")
    if len(words) == 1:
        return line.ljust(width)
    gaps = len(words) - 1
    letters = sum(len(w) for w in words)
    spaces = width - letters
    base = spaces // gaps
    extra = spaces % gaps
    out = ""
    for i in range(gaps):
        pad = base + 1 if i < extra else base
        out += words[i] + " " * pad
    return out + words[-1]


def justify(text, width):
    lines = wrap(text, width)
    if not lines:
        return []
    body = [justify_line(line, width) for line in lines[:-1]]
    return body + [lines[-1].ljust(width)]


def center(line, width):
    if len(line) >= width:
        return line
    total = width - len(line)
    left = total // 2
    return " " * left + line + " " * (total - left)


def truncate(text, width, marker="..."):
    if len(text) <= width:
        return text
    if width <= len(marker):
        return marker[:width]
    return text[:width - len(marker)] + marker


def line_count(text, width):
    return len(wrap(text, width))


def raggedness(lines, width):
    """Sum of squared trailing gaps, ignoring the last line."""
    total = 0
    for line in lines[:-1]:
        gap = width - len(line)
        total += gap * gap
    return total
