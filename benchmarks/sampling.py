from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def stratified_sample(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["repo"])].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    names = sorted(groups)
    while len(selected) < min(count, len(rows)):
        progressed = False
        for name in names:
            if groups[name]:
                selected.append(groups[name].pop())
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    return selected
