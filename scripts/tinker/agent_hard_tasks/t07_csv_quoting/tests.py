import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

assert parse_line('a,b,c') == ['a', 'b', 'c'], "simple record"
assert parse_line('x,"y, z",w') == ['x', 'y, z', 'w'], "delimiter inside quotes"
assert parse_line('"say ""hi"", ok",2') == ['say "hi", ok', '2'], 'record "say ""hi"", ok",2'
assert parse_line('a,,b,') == ['a', '', 'b', ''], "record a,,b,"
assert parse('h1,h2\n"multi\nline",2\n') == [['h1', 'h2'], ['multi\nline', '2']], "newline inside quotes"
assert read_dicts('name,qty,note\nbolt,3,\nnut,,x') == [{'name': 'bolt', 'qty': '3', 'note': ''}, {'name': 'nut', 'qty': '', 'note': 'x'}], "dict rows"
assert write_row(['a', None, 3, ' pad']) == 'a,,3," pad"', "write_row"
rows = [['id', 'text'], ['1', 'plain'], ['2', 'has, comma'], ['3', '5" ruler'], ['4', ''], ['5', 'end']]
assert parse(to_text(rows)) == rows, "round trip of " + repr(to_text(rows))
assert sniff_delimiter('a;b;c\n1;2;3') == ';', "sniff delimiter"
assert column(write_dicts([{'k': 'x'}, {'k': 'y', 'v': '1'}], ['k', 'v']), 'v') == ['', '1'], "write and read dicts"
