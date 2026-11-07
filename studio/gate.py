"""Promotion gate: a new model replaces the current default only if it is measurably better.

Rules (from the plan):
  * it wins at least 55% of blind A/B votes against the current model (ties count half),
    with at least MIN_VOTES votes, and
  * no format's automatic-check score drops by more than 5 points.
"""
import json, os

from chat_engine import RUNS, weights_stamp
from evaluate import counts
from studio import db

WIN_RATE = 0.55
MAX_DROP = 5.0
MIN_VOTES = 10


def latest_eval(run, quick=None):
    """The newest complete evaluation of the run's current weights (older ones are stale)."""
    h = os.path.join(RUNS, run, "eval", "history.jsonl")
    if not os.path.exists(h):
        return None
    since = weights_stamp(os.path.join(RUNS, run))
    rows = [r for r in (json.loads(l) for l in open(h) if l.strip())
            if counts(r) and r["time"] >= since]
    if quick is not None:
        rows = [r for r in rows if r.get("quick") == quick]
    return rows[-1] if rows else None


def check_drops(new, cur):
    """Formats whose score fell by more than MAX_DROP, as {format: (new, current)}."""
    out = {}
    for k, v in (cur or {}).get("per_format", {}).items():
        nv = (new or {}).get("per_format", {}).get(k)
        if nv is not None and nv < v - MAX_DROP:
            out[k] = (nv, v)
    return out


def decide(candidate, current):
    """Return {"pass": bool | None, "reasons": [...], ...}; None = not enough evidence yet."""
    if not current:
        return {"pass": True, "reasons": ["No current default model, so the first one is promoted."]}
    # compare like with like: full evaluations if both have one, else quick ones
    ne, ce = latest_eval(candidate, quick=False), latest_eval(current, quick=False)
    if not (ne and ce):
        ne, ce = latest_eval(candidate, quick=True), latest_eval(current, quick=True)
    reasons, ok = [], True
    if not ne or not ce:
        return {"pass": None, "reasons": ["Run Evaluate on both models (since their last training) first."],
                "candidate_eval": ne, "current_eval": ce}
    drops = check_drops(ne, ce)
    if drops:
        ok = False
        reasons += [f"{k}: checks fell from {c} to {n}" for k, (n, c) in drops.items()]
    # only votes on pairs made after both models' latest training count
    since = max(weights_stamp(os.path.join(RUNS, candidate)), weights_stamp(os.path.join(RUNS, current)))
    st = (db.pair_stats(candidate, current, since=since) or [None])[0]
    if not st or st["votes"] < MIN_VOTES:
        n = st["votes"] if st else 0
        return {"pass": None if ok else False,
                "reasons": reasons + [f"Needs at least {MIN_VOTES} blind votes ({n} so far)."],
                "pairs": st, "candidate_eval": ne, "current_eval": ce}
    if st["win_rate"] < WIN_RATE:
        ok = False
        reasons.append(f"Won {st['win_rate']:.0%} of votes; needs {WIN_RATE:.0%}.")
    else:
        reasons.append(f"Won {st['win_rate']:.0%} of {st['votes']} votes.")
    return {"pass": ok, "reasons": reasons, "pairs": st, "candidate_eval": ne, "current_eval": ce}
