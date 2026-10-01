"""EXPLORATORY, excluded from the Milestone 3 verdict: the frozen grid ended
at ε = 0.5 and the best validation model sat on that edge.  This extends the
resolution axis (ε = 0.7, 1.0, context family C1, seed 0) with the same
pipeline and scores the models on the sealed test with the same scorer.
Recorded as exploratory in MILESTONE3_PROTOCOL.md."""
from __future__ import annotations
import json, os
import numpy as np
from emergence.lang import discover as D, evaluate as E
from emergence.lang.m3 import OUT, substrate, load_splits, val_suite, predict_all, score, log

def main():
    sub, _ = substrate(); splits = load_splits(); vs = val_suite(sub, splits)
    X = sub.pieces(splits["T_pieces"]); S = {fam: sub.behave(X, strings) for fam, strings in splits["T"].items()}
    out = {}
    for eps in (0.7, 1.0):
        model, disc = D.extract(sub, eps, np.random.default_rng(0), ctx_len=1, withhold=splits["W"], prefixes=splits["D_pool"])
        val = E.evaluate_fidelity(model, sub, vs, disc["contexts"]); cx = E.measure_complexity(model)
        pr = predict_all(model, disc["contexts"], sub, splits, f"explore_eps{eps}")
        sc = {fam: score(pr["predictions"][fam], S[fam]) for fam in S}
        out[f"eps{eps}_C1_s0"] = dict(K=model.K, val=val, complexity=cx, scores=sc)
        log(f"EXPLORATORY eps={eps} C1: K={model.K} val={val['fidelity_argmax']:.3f} bits={cx['total_bits_argmax']:.0f} test " + " ".join(f"{fam}={sc[fam]['fidelity_argmax']:.3f}" for fam in sc))
    json.dump(out, open(os.path.join(OUT, "exploratory_eps.json"), "w"), default=float)

if __name__ == "__main__":
    main()
