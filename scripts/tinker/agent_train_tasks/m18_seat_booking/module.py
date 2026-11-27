"""Theater seat booking map with lettered rows and numbered seats."""


class SeatMap:
    def __init__(self, rows, seats_per_row):
        self.rows = rows
        self.width = seats_per_row
        self.grid = [[False] * seats_per_row for _ in range(rows)]

    def parse(self, label):
        row = ord(label[0].upper()) - ord("A")
        seat = int(label[1:]) - 1
        if row < 0 or row >= self.rows or seat < 0 or seat >= self.width:
            raise ValueError("no such seat")
        return row, seat

    def label(self, row, seat):
        return chr(ord("A") + row) + str(seat + 1)

    def is_free(self, label):
        r, s = self.parse(label)
        return not self.grid[r][s]

    def book(self, labels):
        """Book all seats or none."""
        coords = [self.parse(x) for x in labels]
        if len(set(coords)) != len(coords):
            return False
        for r, s in coords:
            if self.grid[r][s]:
                return False
        for r, s in coords:
            self.grid[r][s] = True
        return True

    def cancel(self, label):
        r, s = self.parse(label)
        was = self.grid[r][s]
        self.grid[r][s] = False
        return was

    def free_count(self):
        return sum(row.count(False) for row in self.grid)

    def row_string(self, row):
        return "".join("X" if taken else "." for taken in self.grid[row])

    def find_block(self, n):
        """First run of n adjacent free seats, scanning rows front to back."""
        if n <= 0 or n > self.width:
            return None
        for r in range(self.rows):
            run = 0
            for s in range(self.width):
                run = 0 if self.grid[r][s] else run + 1
                if run == n:
                    return [self.label(r, s - n + 1 + i) for i in range(n)]
        return None

    def best_block(self, n):
        """Block of n nearest the centre seat of the middle-most row that fits."""
        middle = (self.rows - 1) / 2
        order = sorted(range(self.rows), key=lambda r: (abs(r - middle), r))
        centre = (self.width - n) / 2
        for r in order:
            starts = sorted(range(self.width - n + 1), key=lambda s: (abs(s - centre), s))
            for s in starts:
                if not any(self.grid[r][s:s + n]):
                    return [self.label(r, s + i) for i in range(n)]
        return None

    def occupancy(self):
        total = self.rows * self.width
        return (total - self.free_count()) * 100 // total
