import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

assert rle_encode("aaabcc") == "3a1b2c", "rle of aaabcc"
assert rle_decode("3a1b2c") == "aaabcc", "decode 3a1b2c"
assert runs("xxyz") == [("x", 2), ("y", 1), ("z", 1)], "runs of xxyz"
assert rle_decode(rle_encode("x" * 12 + "y")) == "x" * 12 + "y", "rle round trip of 12 x and a y"
codes = build_codes("abracadabra")
assert sorted(len(c) for c in codes.values()) == [1, 3, 3, 3, 3], "code lengths for abracadabra"
bits, codes = huffman_encode("abracadabra")
assert huffman_decode(bits, codes) == "abracadabra", "huffman round trip"
assert average_code_length("abracadabra") == 2.091, "average code length"
bits, codes = huffman_encode("zzzz")
assert (bits, huffman_decode(bits, codes)) == ("0000", "zzzz"), "huffman of zzzz"
for sample in ["", "a", "hello world", "aaaaaaaaaabbbbbbbbbbbbc", "mississippi"]:
    bits, codes = pack(sample)
    assert unpack(bits, codes) == sample, f"pack/unpack of {sample!r}"
assert encoded_size("aaaaaaaaaaaaaaaaaaaa") == 5, "encoded size of 20 a"
