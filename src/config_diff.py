"""Config-diff audit for H1 (pack weakness 3, mandatory).

Reads the hand arm's recipe declaration and the Lightning arm's
effective config (built from the actual optimizer objects) and
writes the per-key comparison to results/config_diff.json. Any key
that differs is published, not smoothed over.
"""
from __future__ import annotations

import json

from .common import RESULTS, recipe_dict, write_json


def run_diff() -> dict:
    from .lightning_arm import effective_config

    hand = recipe_dict()
    lit = effective_config()
    rows = []
    for key in sorted(set(hand) | set(lit)):
        hv, lv = hand.get(key), lit.get(key)
        rows.append(
            {
                "setting": key,
                "hand_loop": hv,
                "lightning": lv,
                "equal": json.dumps(hv, sort_keys=True) == json.dumps(lv, sort_keys=True),
            }
        )
    diff = {"rows": rows, "n_differ": sum(1 for r in rows if not r["equal"])}
    write_json(RESULTS / "config_diff.json", diff)
    return diff


if __name__ == "__main__":
    d = run_diff()
    for r in d["rows"]:
        mark = "OK " if r["equal"] else "DIFF"
        print(f"{mark} {r['setting']}: hand={r['hand_loop']} lightning={r['lightning']}")
    print("differing keys:", d["n_differ"])
