"""Tokenizer for arithmetic text: numbers, identifiers, operators and parentheses."""

OPERATORS = "+-*/%^<>"
TWO_CHAR = ("**", "//", "<=", ">=", "==", "!=")


def is_ident_start(ch):
    return ch.isalpha() or ch == "_"


def read_number(text, i):
    """Return (token, next_index) for a number starting at i (one dot allowed)."""
    start = i
    seen_dot = False
    while i < len(text) and (text[i].isdigit() or (text[i] == "." and not seen_dot)):
        if text[i] == ".":
            seen_dot = True
        i += 1
    return text[start:i], i


def read_ident(text, i):
    start = i
    while i < len(text) and (text[i].isalnum() or text[i] == "_"):
        i += 1
    return text[start:i], i


def tokenize(text):
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == " ":
            i += 1
        elif ch.isdigit() or (ch == "." and i + 1 < len(text) and text[i + 1].isdigit()):
            tok, i = read_number(text, i)
            tokens.append(("NUM", tok))
        elif is_ident_start(ch):
            tok, i = read_ident(text, i)
            tokens.append(("ID", tok))
        elif text[i:i + 2] in TWO_CHAR:
            tokens.append(("OP", text[i:i + 2]))
            i += 2
        elif ch in OPERATORS:
            tokens.append(("OP", ch))
            i += 1
        elif ch in "()":
            tokens.append(("LP" if ch == "(" else "RP", ch))
            i += 1
        else:
            raise ValueError(f"unexpected character {ch!r} at {i}")
    return tokens


def balanced(tokens):
    depth = 0
    for kind, _ in tokens:
        if kind == "LP":
            depth += 1
        elif kind == "RP":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def max_depth(tokens):
    depth = best = 0
    for kind, _ in tokens:
        if kind == "LP":
            depth += 1
            best = max(best, depth)
        elif kind == "RP":
            depth -= 1
    return best


def mark_unary(tokens):
    """Rename + and - to U+ / U- where they cannot be binary operators."""
    out = []
    for i in range(len(tokens)):
        kind, val = tokens[i]
        prev = tokens[i - 1][0] if i > 0 else None
        if kind == "OP" and val in ("+", "-") and (prev is None or prev == "OP" or prev == "LP"):
            out.append(("OP", "U" + val))
        else:
            out.append((kind, val))
    return out


def identifiers(tokens):
    seen = []
    for kind, val in tokens:
        if kind == "ID" and val not in seen:
            seen.append(val)
    return seen
