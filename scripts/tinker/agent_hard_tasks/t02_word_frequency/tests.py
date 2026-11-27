import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from solution import *

assert tokenize("Hello, world!") == ["hello", "world"], "basic tokens"
assert tokenize("don't -- stop") == ["don't", "stop"], "apostrophes and dashes"
counts = word_frequencies("Zebra apple, zebra! Apple mango.")
assert counts == {"zebra": 2, "apple": 2, "mango": 1}, "frequencies"
assert top_n(counts, 2) == [("apple", 2), ("zebra", 2)], "top two words"
assert top_n({"b": 1, "a": 1, "c": 3}, 3) == [("c", 3), ("a", 1), ("b", 1)], "ranking"
assert average_word_length("wait ... what") == 4.0, "average word length"
assert lexical_diversity("go go stop") == 0.667, "lexical diversity"
r = report("The cat and the hat ... the cat sat.", 2)
assert r.splitlines() == ["cat: 2 (50.0%)", "hat: 1 (25.0%)"], r
assert split_sentences("One. Two!  Three") == ["One.", "Two!", "Three"], "sentences"
assert sentence_lengths("A b c. ... D e.") == [3, 2], "sentence lengths"
assert compare_texts("red red blue", "blue green green", 2) == (["red"], ["green"]), "compare"
