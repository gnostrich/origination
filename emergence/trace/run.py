"""Driver: backward tracing of final interfaces over existing checkpoints.

    python -m emergence.trace.run trace --run results/grok/mlp_s0
    python -m emergence.trace.run trace --run results/grok/transformer_s0_nowd --ref results/grok/transformer_s0
    python -m emergence.trace.run trace --run results/world/perm4_s0
    python -m emergence.trace.run report
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

OUT = "results/trace"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trace", "report"])
    ap.add_argument("--run")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "trace":
        torch.set_num_threads(1)
        name = Path(a.run).name
        out = os.path.join(OUT, f"{name}{'_smoke' if a.smoke else ''}.json")
        if a.smoke:  # execution check only: keep two checkpoints
            _limit_checkpoints(a.run)
        if "world" in a.run:
            from emergence.trace.s4 import trace
            trace(a.run, out=out)
        else:
            from emergence.trace.modadd import trace
            trace(a.run, ref_dir=a.ref, out=out)
    else:
        from emergence.trace.report import report
        report()


def _limit_checkpoints(run):
    """Monkey-patch torch.load so the tracer sees only the first and last checkpoint."""
    orig = torch.load

    def patched(*args, **kw):
        d = orig(*args, **kw)
        if isinstance(d, dict) and "checkpoints" in d:
            ks = sorted(d["checkpoints"])
            d["checkpoints"] = {k: d["checkpoints"][k] for k in (ks[0], ks[-1])}
        return d
    torch.load = patched


if __name__ == "__main__":
    main()
