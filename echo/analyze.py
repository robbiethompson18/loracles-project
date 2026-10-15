"""Summarise echo_worker.py output (summary.jsonl) into the tables that answer the echo question.

Cell = median over start steps of the student's own animal's best whole-word rank (of ~40k) at one
layer, taken at either end of the z-scored J-lens readout: "12+" means it ranked 12th from the top,
"7-" 7th from the bottom (the echo predicts extremes of either sign). "others" is the same for the
other four animals (median over them), the within-readout null. Bank rows show the "bank" group.

  python echo/analyze.py OUT_DIR/summary.jsonl [--layer 32]
"""
import argparse, json, statistics
from collections import defaultdict

ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl"]
UPD = [f"upd{K}{x}" for K in (1, 25) for x in ("", "_lin", "_B", "_A")]
ORDER = [("ckpt", "ckpt")] + [(d, a) for d in ("cached", "adam-num", "adam-bank") for a in UPD] + \
        [(d, a) for d in ("num", "bank", "rand") for a in ("dense", "fresh", "lora", "B", "A")]


def load(paths):
    """Rows, last one wins per (student, S, data, arm). Early runs wrote update rows with data/arm swapped."""
    rows = {}
    for f in paths:
        for line in open(f):
            r = json.loads(line)
            if r["data"].startswith("upd"):
                r["data"], r["arm"] = r["arm"], r["data"]
            rows[(r["student"], r["S"], r["data"], r["arm"])] = r
    return list(rows.values())


def best(g):
    """(rank, sign) at whichever end the group sits closest to."""
    return (g["rank_top"], "+") if g["rank_top"] <= g["rank_bot"] else (g["rank_bot"], "-")


def fmt(xs):
    if not xs:
        return "."
    r = statistics.median(x for x, _ in xs)
    plus = sum(s == "+" for _, s in xs)
    return f"{r:>6.0f} {plus}/{len(xs)}+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", nargs="+")
    ap.add_argument("--layer", default="32")
    ap.add_argument("--s0", action="store_true", help="only S = 0 (init) rows")
    args = ap.parse_args()
    recs = load(args.summary)
    own, oth, bank = defaultdict(list), defaultdict(list), defaultdict(list)
    students = []
    for r in recs:
        if args.s0 != (r["S"] == 0):
            continue
        d = r["decode"].get(args.layer) or r["decode"].get(int(args.layer))
        if d is None:
            continue
        key = (r["student"], r["data"], r["arm"])
        if r["student"] not in students:
            students.append(r["student"])
        a = r["animal"]
        if a:
            own[key].append(best(d["groups"][a]))
        oth[key] += [best(d["groups"][x]) for x in ANIMALS if x != a]
        bank[key].append(best(d["groups"]["bank"]))
    w = 22
    print(f"layer {args.layer}{' (S=0 only)' if args.s0 else ''}: median best rank (of ~40k words) over start steps, "
          "then how many of them sat at the top (+) end")
    print(f"{'arm':16s}" + "".join(f"{s + ' own':>{w}s}{s + ' others':>{w}s}" for s in students))
    for data, arm in ORDER:
        if not any((s, data, arm) in oth for s in students):
            continue
        print(f"{data + '-' + arm:16s}" + "".join(f"{fmt(own[(s, data, arm)]):>{w}s}{fmt(oth[(s, data, arm)]):>{w}s}" for s in students))
    print("\nbank group")
    print(f"{'arm':16s}" + "".join(f"{s:>{w}s}" for s in students))
    for data, arm in ORDER:
        if not any((s, data, arm) in bank for s in students):
            continue
        print(f"{data + '-' + arm:16s}" + "".join(f"{fmt(bank[(s, data, arm)]):>{w}s}" for s in students))


if __name__ == "__main__":
    main()
