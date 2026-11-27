"""Change-making: greedy, optimal (dynamic programming) and counting combinations."""

US_COINS = [25, 10, 5, 1]


def greedy_change(amount, coins):
    """Largest-coin-first change as {coin: count}, or None if it cannot finish."""
    if amount < 0:
        raise ValueError("negative amount")
    result = {}
    for coin in sorted(coins, reverse=True):
        count = amount // coin
        if count > 0:
            result[coin] = count
            amount -= count * coin
    return result if amount == 0 else None


def min_coins(amount, coins):
    """Fewest coins to make amount, or -1 if impossible."""
    impossible = amount + 1
    best = [0] + [impossible] * amount
    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value and best[value - coin] + 1 < best[value]:
                best[value] = best[value - coin] + 1
    return best[amount] if best[amount] <= amount else -1


def optimal_change(amount, coins):
    impossible = amount + 1
    best = [0] + [impossible] * amount
    last = [0] * (amount + 1)
    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value and best[value - coin] + 1 < best[value]:
                best[value] = best[value - coin] + 1
                last[value] = coin
    if amount > 0 and best[amount] > amount:
        return None
    result = {}
    while amount > 0:
        coin = last[amount]
        result[coin] = result.get(coin, 0) + 1
        amount -= coin
    return result


def count_ways(amount, coins):
    """Number of unordered coin combinations summing to amount."""
    ways = [1] + [0] * amount
    for coin in coins:
        for value in range(coin, amount + 1):
            ways[value] += ways[value - coin]
    return ways[amount]


def coin_total(change):
    return sum(coin * n for coin, n in change.items())


def first_greedy_failure(coins, limit):
    """First amount up to limit where greedy is worse than optimal, else 0."""
    for amount in range(1, limit + 1):
        g = greedy_change(amount, coins)
        best = min_coins(amount, coins)
        if g is None:
            if best != -1:
                return amount
            continue
        if sum(g.values()) > best:
            return amount
    return 0


def make_change_from_till(amount, till):
    """Greedy change using limited stock; till maps coin -> available count."""
    result = {}
    for coin in sorted(till, reverse=True):
        use = min(till[coin], amount // coin)
        if use > 0:
            result[coin] = use
            amount -= use * coin
    return result if amount == 0 else None
