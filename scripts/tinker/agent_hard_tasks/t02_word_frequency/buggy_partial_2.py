"""Tokenizer and word-frequency report for plain English text."""

_PUNCT = ".,;:!?\"'()[]{}<>-_*/"

DEFAULT_STOPWORDS = frozenset({"a", "an", "and", "the", "of", "to", "in", "is", "it", "on"})


def normalize(token):
    """Lower-case a token and trim surrounding punctuation (inner apostrophes stay)."""
    return token.strip(_PUNCT).lower()


def split_sentences(text):
    sentences, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            sentence = "".join(buf).strip()
            if sentence.strip(".!?"):
                sentences.append(sentence)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return sentences


def tokenize(text):
    """Split text into normalized word tokens."""
    words = []
    for raw in text.replace("--", " ").split():
        word = normalize(raw)
        words.append(word)
    return words


def word_frequencies(text, stopwords=DEFAULT_STOPWORDS):
    counts = {}
    for word in tokenize(text):
        if word in stopwords:
            continue
        counts[word] = counts.get(word, 0) + 1
    return counts


def top_n(counts, n):
    """The n most frequent (word, count) pairs; equal counts are ordered alphabetically."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]


def average_word_length(text):
    words = tokenize(text)
    if not words:
        return 0.0
    return round(sum(len(w) for w in words) / len(words), 2)


def lexical_diversity(text):
    words = tokenize(text)
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)


def longest_words(text, n=3):
    unique = set(tokenize(text))
    return sorted(unique, key=lambda w: (-len(w), w))[:n]


def sentence_lengths(text):
    return [len(tokenize(s)) for s in split_sentences(text)]


def report(text, n=3, stopwords=DEFAULT_STOPWORDS):
    """One line per top word: 'word: count (share%)', share of all counted words."""
    counts = word_frequencies(text, stopwords)
    total = len(tokenize(text))
    lines = []
    for word, count in top_n(counts, n):
        share = 100.0 * count / total if total else 0.0
        lines.append(f"{word}: {count} ({share:.1f}%)")
    return "\n".join(lines)


def summary(text):
    words = tokenize(text)
    return {
        "sentences": len(split_sentences(text)),
        "words": len(words),
        "unique": len(set(words)),
        "avg_length": average_word_length(text),
        "diversity": lexical_diversity(text),
        "longest": longest_words(text, 1),
    }


def compare_texts(a, b, n=5):
    """Words in the top n of exactly one of the two texts, as (only_a, only_b)."""
    top_a = {w for w, _ in top_n(word_frequencies(a), n)}
    top_b = {w for w, _ in top_n(word_frequencies(b), n)}
    return sorted(top_a - top_b), sorted(top_b - top_a)
