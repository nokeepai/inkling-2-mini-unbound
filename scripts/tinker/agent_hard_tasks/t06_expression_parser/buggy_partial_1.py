"""Tokenizer, recursive-descent parser and evaluator for arithmetic expressions."""
import math

_FUNCS = {"min": min, "max": max, "abs": abs, "sqrt": math.sqrt}


class ParseError(Exception):
    pass


def tokenize(text):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch.isdigit() or ch == ".":
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            tokens.append(("num", float(text[i:j])))
            i = j
        elif ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(("name", text[i:j]))
            i = j
        elif ch in "+-*/^(),%":
            tokens.append(("op", ch))
            i += 1
        else:
            raise ParseError(f"unexpected character {ch!r}")
    tokens.append(("end", None))
    return tokens


class Parser:
    def __init__(self, tokens, variables):
        self.tokens = tokens
        self.pos = 0
        self.variables = variables

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, value):
        tok = self.advance()
        if tok != ("op", value):
            raise ParseError(f"expected {value!r}, got {tok[1]!r}")

    def parse(self):
        value = self.expr()
        if self.peek()[0] != "end":
            raise ParseError(f"trailing input at {self.peek()[1]!r}")
        return value

    def expr(self):
        value = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.advance()[1]
            rhs = self.term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term(self):
        value = self.unary()
        while self.peek() in (("op", "*"), ("op", "/"), ("op", "%")):
            op = self.advance()[1]
            rhs = self.unary()
            if op == "*":
                value = value * rhs
            elif op == "/":
                value = value / rhs
            else:
                value = value % rhs
        return value

    def unary(self):
        if self.peek() == ("op", "+"):
            self.advance()
            return self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        while self.peek() == ("op", "^"):
            self.advance()
            base = base ** self.atom()
        return base

    def atom(self):
        tok = self.advance()
        if tok[0] == "num":
            return tok[1]
        if tok == ("op", "-"):
            return -self.atom()
        if tok == ("op", "("):
            value = self.expr()
            self.expect(")")
            return value
        if tok[0] == "name":
            name = tok[1]
            if self.peek() == ("op", "("):
                self.advance()
                args = self.arguments()
                if name not in _FUNCS:
                    raise ParseError(f"unknown function {name}")
                return _FUNCS[name](*args)
            if name in self.variables:
                return self.variables[name]
            raise ParseError(f"unknown variable {name}")
        raise ParseError(f"unexpected token {tok[1]!r}")

    def arguments(self):
        args = []
        if self.peek() == ("op", ")"):
            self.advance()
            return args
        while True:
            args.append(self.expr())
            tok = self.advance()
            if tok == ("op", ")"):
                return args
            if tok != ("op", ","):
                raise ParseError("expected ',' or ')'")


def evaluate(text, variables=None):
    value = Parser(tokenize(text), variables or {}).parse()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def evaluate_all(lines, variables=None):
    """Evaluate 'name = expr' lines in order; later lines may use earlier names."""
    env = dict(variables or {})
    results = {}
    for line in lines:
        name, sep, expr = line.partition("=")
        if not sep:
            raise ParseError(f"missing '=' in {line!r}")
        name = name.strip()
        env[name] = evaluate(expr, env)
        results[name] = env[name]
    return results
