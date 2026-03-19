def pareto_frontier(models):

    frontier = []

    for m in models:

        dominated = False

        for n in models:

            if (
                n["accuracy"] >= m["accuracy"]
                and n["params"] <= m["params"]
                and n != m
            ):
                dominated = True
                break

        if not dominated:
            frontier.append(m)

    return frontier