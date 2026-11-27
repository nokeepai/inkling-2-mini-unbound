"""ISBN-10, ISBN-13 and Luhn check digits, plus card-number masking."""


def digits_only(text):
    return [c for c in text if c != "-" and c != " "]


def luhn_checksum(number):
    digits = [int(c) for c in digits_only(number)]
    total = 0
    # double every second digit counting from the right
    parity = len(digits) % 2
    for i in range(len(digits)):
        d = digits[i]
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10


def luhn_valid(number):
    chars = digits_only(number)
    if len(chars) < 2 or not all(c.isdigit() for c in chars):
        return False
    return luhn_checksum(number) == 0


def luhn_check_digit(partial):
    """Digit to append so that partial + digit is Luhn-valid."""
    for d in range(10):
        if luhn_checksum(partial + str(d)) == 0:
            return d
    return -1


def isbn10_check(first9):
    total = 0
    for i in range(9):
        total += (10 - i) * int(first9[i])
    r = (11 - total % 11) % 11
    return "X" if r == 10 else str(r)


def isbn10_valid(isbn):
    chars = digits_only(isbn)
    if len(chars) != 10:
        return False
    if not all(c.isdigit() for c in chars[:9]):
        return False
    return isbn10_check("".join(chars[:9])) == chars[9].upper()


def isbn13_check(first12):
    total = 0
    for i in range(12):
        weight = 1 if i % 2 == 0 else 3
        total += weight * int(first12[i])
    return str((10 - total % 10) % 10)


def isbn13_valid(isbn):
    chars = digits_only(isbn)
    if len(chars) != 13 or not all(c.isdigit() for c in chars):
        return False
    return isbn13_check("".join(chars[:12])) == chars[12]


def isbn10_to_13(isbn10):
    chars = digits_only(isbn10)
    body = "978" + "".join(chars[:9])
    return body + isbn13_check(body)


def mask_card(number, visible=4):
    chars = digits_only(number)
    hidden = len(chars) - visible
    if hidden <= 0:
        return "".join(chars)
    return "*" * hidden + "".join(chars[hidden:])


def group_digits(number, size=4):
    chars = "".join(digits_only(number))
    return " ".join(chars[i:i + size] for i in range(0, len(chars), size))
