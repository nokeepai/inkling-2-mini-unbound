"""Run-length encoding plus a small Huffman coder, and a pack/unpack pipeline."""
import heapq


def rle_encode(text):
    """'aaabcc' -> '3a1b2c'. Input text must not contain digits."""
    if not text:
        return ""
    if any(ch.isdigit() for ch in text):
        raise ValueError("digits cannot be run-length encoded")
    out = []
    current, count = text[0], 1
    for ch in text:
        if ch == current:
            count += 1
        else:
            out.append(f"{count}{current}")
            current, count = ch, 1
    out.append(f"{count}{current}")
    return "".join(out)


def rle_decode(code):
    out, num = [], 0
    for ch in code:
        if ch.isdigit():
            num = int(ch)
        else:
            if num == 0:
                raise ValueError("missing run length")
            out.append(ch * num)
            num = 0
    if num:
        raise ValueError("dangling run length")
    return "".join(out)


def runs(text):
    """List of (char, length) pairs, via the RLE string."""
    code = rle_encode(text)
    pairs, num = [], ""
    for ch in code:
        if ch.isdigit():
            num += ch
        else:
            pairs.append((ch, int(num)))
            num = ""
    return pairs


def frequencies(text):
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return freq


def build_codes(text):
    """Prefix-free bit strings per symbol; ties are broken by symbol order then creation."""
    freq = frequencies(text)
    if not freq:
        return {}
    if len(freq) == 1:
        return {next(iter(freq)): ""}
    heap = []
    for order, (sym, count) in enumerate(sorted(freq.items())):
        heap.append((count, order, {sym: ""}))
    heapq.heapify(heap)
    order = len(heap)
    while len(heap) > 1:
        c1, _, left = heapq.heappop(heap)
        c2, _, right = heapq.heappop(heap)
        merged = {s: "0" + code for s, code in left.items()}
        merged.update({s: "1" + code for s, code in right.items()})
        heapq.heappush(heap, (c1 + c2, order, merged))
        order += 1
    return heap[0][2]


def huffman_encode(text, codes=None):
    codes = build_codes(text) if codes is None else codes
    return "".join(codes[ch] for ch in text), codes


def huffman_decode(bits, codes):
    reverse = {code: sym for sym, code in codes.items()}
    out, buf = [], ""
    for bit in bits:
        buf += bit
        if buf in reverse:
            out.append(reverse[buf])
            buf = ""
    if buf:
        raise ValueError("trailing bits")
    return "".join(out)


def pack(text):
    """RLE first, then Huffman-code the RLE string. Returns (bits, codes)."""
    return huffman_encode(rle_encode(text))


def unpack(bits, codes):
    return rle_decode(huffman_decode(bits, codes))


def encoded_size(text):
    bits, codes = pack(text)
    return len(bits)


def compression_ratio(text):
    if not text:
        return 1.0
    return round(encoded_size(text) / (8 * len(text)), 3)


def average_code_length(text):
    codes = build_codes(text)
    if not text:
        return 0.0
    return round(sum(len(codes[ch]) for ch in text) / len(text), 3)
