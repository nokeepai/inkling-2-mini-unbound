"""Round-robin tournament standings: points, goal difference and tiebreaks."""

WIN_POINTS = 3
DRAW_POINTS = 1


def parse_result(line):
    """Parse 'Home 2-1 Away' into (home, home_goals, away_goals, away)."""
    left, right = line.split("-")
    home_parts = left.strip().rsplit(" ", 1)
    away_parts = right.strip().split(" ", 1)
    return home_parts[0], int(home_parts[1]), int(away_parts[0]), away_parts[1]


class Table:
    def __init__(self, teams):
        # per team: wins, draws, losses, goals for, goals against
        self.stats = {t: [0, 0, 0, 0, 0] for t in teams}

    def record(self, home, hg, ag, away):
        h = self.stats[home]
        a = self.stats[away]
        h[3] += hg
        h[4] += ag
        a[3] += ag
        a[4] += hg
        if hg > ag:
            h[0] += 1
            a[2] += 1
        elif hg < ag:
            a[0] += 1
            h[2] += 1
        else:
            h[1] += 1
            a[1] += 1

    def record_line(self, line):
        self.record(*parse_result(line))

    def points(self, team):
        s = self.stats[team]
        return s[0] * WIN_POINTS + s[1] * DRAW_POINTS

    def played(self, team):
        s = self.stats[team]
        return s[0] + s[1] + s[2]

    def goal_diff(self, team):
        s = self.stats[team]
        return s[3] - s[4]

    def standings(self):
        """Order by points, then goal difference, then goals for, then name."""
        keyed = []
        for team in self.stats:
            key = (-self.points(team), -self.goal_diff(team), -self.stats[team][3], team)
            keyed.append(key)
        keyed.sort()
        return [k[3] for k in keyed]

    def position(self, team):
        return self.standings().index(team) + 1

    def leader_gap(self, team):
        order = self.standings()
        return self.points(order[0]) - self.points(team)


def fixtures(teams):
    """All pairings for a single round robin, in a fixed order."""
    out = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            out.append((teams[i], teams[j]))
    return out


def clinched(table, team, games_left):
    """True if no other team can reach team's points with the games remaining."""
    mine = table.points(team)
    for other in table.stats:
        if other != team and table.points(other) + games_left * WIN_POINTS >= mine:
            return False
    return True


def unbeaten(table):
    return sorted(t for t, s in table.stats.items() if s[2] == 0 and s[0] + s[1] > 0)
