"""Timelines, transition timing and microscopic comparison for the traced runs."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

from emergence.trace.common import rise_times

OUT = "results/trace"

PROPS = {  # property series (rising unless noted); s4-only or modadd-only keys are skipped when absent
    "test_acc": +1, "train_acc": +1,
    "identity_probe": +1, "identity_knn": +1,
    "causal_remove": +1, "causal_keep": +1,
    "subst": +1, "subst_ctx": +1, "ctx_indep": +1, "fit": +1,
    "composition": +1, "reuse": +1, "recursive": +1, "assoc": +1,
    "q_classes": -1, "q_meta": +1, "q_closure": +1, "q_assoc": +1, "q_latin": +1, "q_coherence": +1, "q_lawfree": +1,
}
CLICK_PROPS = ["identity_probe", "causal_remove", "subst", "ctx_indep", "subst_ctx", "fit", "composition", "recursive"]
MICRO = ["param_norm", "eff_rank", "top5_share", "between_within", "margin", "interference", "sens_between",
         "jac_ctx_dep", "layer_align", "drift", "weight_drift"]
SPACING = {"modadd": 500, "s4": 400}


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        if f.endswith("summary.json"):
            continue
        r = json.load(open(f))
        runs[Path(f).stem] = r
    return runs


def series(rows, key):
    return [rows_k if rows_k is not None else np.nan for rows_k in (r.get(key) for r in rows)]


def analyse_run(name, r):
    rows = r["rows"]
    steps = [x["step"] for x in rows]
    sp = SPACING[r["system"]]
    timing = {}
    for k, d in PROPS.items():
        v = series(rows, k)
        if np.all(np.isnan(np.asarray(v, dtype=float))):
            continue
        timing[k] = rise_times(steps, v, direction=d)
    micro = {}
    for k in MICRO:
        v = series(rows, k)
        if np.all(np.isnan(np.asarray(v, dtype=float))):
            continue
        micro[k] = rise_times(steps, v)
    # candidate click: the median t50 of the interface properties present, and which co-transition within one spacing
    t50s = {k: timing[k]["t50"] for k in CLICK_PROPS if k in timing and timing[k]["t50"] is not None and timing[k]["change"] > 0.1}
    click = None
    co = []
    if len(t50s) >= 3:
        med = float(np.median(list(t50s.values())))
        co = [k for k, t in t50s.items() if abs(t - med) <= sp]
        if len(co) >= 3:
            click = dict(t50=med, members=co, width=float(np.median([timing[k]["width"] for k in co if timing[k]["width"] is not None] or [np.nan])))
    acc = timing.get("test_acc", {})
    rel = None
    if click and acc.get("t50") is not None:
        if acc.get("t90") is not None and click["t50"] > acc["t90"] + sp:
            rel = "after generalisation"
        elif acc.get("t10") is not None and click["t50"] < acc["t10"] - sp:
            rel = "before generalisation"
        else:
            rel = "during generalisation"
    return dict(name=name, system=r["system"], steps=steps, timing=timing, micro=micro, t50s=t50s, click=click,
                relation=rel, acc=acc, final=rows[-1], first=rows[0])


def fmt(t):
    return "–" if t is None else str(int(t))


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load()
    an = {n: analyse_run(n, r) for n, r in runs.items()}
    L = ["# Backward tracing of final interfaces: results", "",
         "Each run's final interfaces (the frozen extractor's behavioural classes at the last checkpoint) are traced back "
         "through every checkpoint.  Properties are measured on held-out instances against the frozen final partition.  "
         "For every series the persistent 10/50/90 % crossing steps of its rise between its 10th and 90th percentiles "
         "are reported; width = t90 − t10.  A *candidate click* is a set of ≥ 3 interface properties whose t50 fall "
         "within one checkpoint spacing of their median.", ""]
    # ---- timelines
    for n, a in an.items():
        r = runs[n]
        rows = r["rows"]
        L += [f"## {n} ({a['system']}, K = {r['K']})", ""]
        if a["system"] == "modadd":
            cols = ["step", "train_acc", "test_acc", "identity_probe", "identity_knn", "causal_remove", "causal_keep", "causal_remove_rand",
                    "subst", "subst_rand", "subst_ctx", "composition", "recursive", "assoc", "q_classes", "q_meta", "q_closure",
                    "eff_rank", "between_within", "margin", "interference", "sens_between", "layer_align", "param_norm", "drift"]
        else:
            cols = ["step", "train_acc", "test_acc", "identity_probe", "identity_knn", "causal_remove", "causal_keep", "causal_remove_rand",
                    "subst", "subst_rand", "subst_F1", "subst_F2", "subst_F5b", "ctx_indep", "fit", "composition", "reuse", "recursive",
                    "q_classes", "q_meta", "q_closure", "eff_rank", "between_within", "margin", "interference", "sens_between",
                    "jac_ctx_dep", "layer_align", "param_norm", "drift"]
        L += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for x in rows:
            L.append("| " + " | ".join(
                (str(x[c]) if c in ("step", "q_classes") else (f"{x[c]:.2f}" if isinstance(x.get(c), (int, float)) and x.get(c) is not None and np.isfinite(x[c]) else "–"))
                for c in cols) + " |")
        L.append("")
        L += ["Transition timing (t10 / t50 / t90 / width, steps):", ""]
        L.append("| series | t10 | t50 | t90 | width | low → high |")
        L.append("|---|---|---|---|---|---|")
        for k, t in a["timing"].items():
            L.append(f"| {k} | {fmt(t['t10'])} | {fmt(t['t50'])} | {fmt(t['t90'])} | {fmt(t['width'])} | {t['lo']:.2f} → {t['hi']:.2f} |")
        for k, t in a["micro"].items():
            L.append(f"| micro:{k} | {fmt(t['t10'])} | {fmt(t['t50'])} | {fmt(t['t90'])} | {fmt(t['width'])} | {t['lo']:.3g} → {t['hi']:.3g} ({'↑' if t['direction'] > 0 else '↓'}) |")
        L.append("")
        if a["click"]:
            L.append(f"**Candidate click** at t50 ≈ {a['click']['t50']:.0f} (members: {', '.join(a['click']['members'])}; median width {a['click']['width']:.0f} steps); "
                     f"test accuracy t10/t50/t90 = {fmt(a['acc'].get('t10'))}/{fmt(a['acc'].get('t50'))}/{fmt(a['acc'].get('t90'))} → click **{a['relation']}**.")
        else:
            L.append(f"**No candidate click** (interface-property t50s: {a['t50s']}).")
        L.append("")
    # ---- cross-run summary
    L += ["## Summary across runs", "",
          "| run | test acc t50 (width) | identity t50 | causal t50 | subst t50 | ctx/fit t50 | composition t50 | recursive t50 | click t50 (members) | relation |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, a in an.items():
        T = a["timing"]
        g = lambda k: fmt(T[k]["t50"]) if k in T else "–"
        ctxk = "ctx_indep" if "ctx_indep" in T else "subst_ctx"
        ck = f"{a['click']['t50']:.0f} ({len(a['click']['members'])})" if a["click"] else "–"
        L.append(f"| {n} | {fmt(a['acc'].get('t50'))} ({fmt(a['acc'].get('width'))}) | {g('identity_probe')} | {g('causal_remove')} | {g('subst')} | "
                 f"{g(ctxk)}/{g('fit')} | {g('composition')} | {g('recursive')} | {ck} | {a['relation'] or '–'} |")
    L.append("")
    # ---- microscopic quantities relative to the click
    L += ["## Microscopic quantities: t50 of their change minus the click t50 (steps; negative = leads the click)", "",
          "| run | click t50 | acc t50 | " + " | ".join(MICRO) + " |", "|---|---|---|" + "---|" * len(MICRO)]
    for n, a in an.items():
        if not a["click"]:
            continue
        c = a["click"]["t50"]
        L.append(f"| {n} | {c:.0f} | {fmt(a['acc'].get('t50'))} | " + " | ".join(
            (f"{a['micro'][k]['t50'] - c:+.0f}" if k in a["micro"] and a["micro"][k]["t50"] is not None and a["micro"][k]["change"] > 0 else "–") for k in MICRO) + " |")
    L.append("")
    # ---- figures
    for system, fname in (("modadd", "trace_modadd.png"), ("s4", "trace_s4.png")):
        names = [n for n in an if an[n]["system"] == system]
        if not names:
            continue
        fig, axes = plt.subplots(len(names), 2, figsize=(13, 2.8 * len(names)), squeeze=False)
        for i, n in enumerate(names):
            rows = runs[n]["rows"]
            st = [x["step"] for x in rows]
            ax = axes[i, 0]
            for k, col in (("test_acc", "k"), ("identity_probe", "C0"), ("causal_remove", "C1"), ("subst", "C2"),
                           ("ctx_indep", "C3"), ("subst_ctx", "C3"), ("fit", "C4"), ("composition", "C5"), ("recursive", "C6")):
                v = series(rows, k)
                if not np.all(np.isnan(np.asarray(v, dtype=float))):
                    ax.plot(st, v, color=col, lw=1.2 if k != "test_acc" else 2, label=k)
            ax.set_xscale("symlog", linthresh=100)
            ax.set_title(f"{n}: interface properties", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            if i == 0:
                ax.legend(fontsize=6, ncol=3)
            ax = axes[i, 1]
            for k, col in (("eff_rank", "C0"), ("between_within", "C1"), ("margin", "C2"), ("interference", "C3"),
                           ("sens_between", "C4"), ("layer_align", "C5"), ("param_norm", "C6"), ("drift", "C7"), ("jac_ctx_dep", "C8")):
                v = np.asarray(series(rows, k), dtype=float)
                if np.all(np.isnan(v)):
                    continue
                lo, hi = np.nanmin(v), np.nanmax(v)
                ax.plot(st, (v - lo) / (hi - lo + 1e-12), color=col, lw=1, label=f"{k} [{lo:.2g},{hi:.2g}]")
            ax.plot(st, series(rows, "test_acc"), color="k", lw=2, label="test_acc")
            ax.set_xscale("symlog", linthresh=100)
            ax.set_title(f"{n}: microscopic quantities (min-max normalised)", fontsize=8)
            if i == 0:
                ax.legend(fontsize=5, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=110)
    open(os.path.join(OUT, "TRACE_REPORT.md"), "w").write("\n".join(L))
    json.dump({n: {k: v for k, v in a.items() if k not in ("final", "first")} for n, a in an.items()},
              open(os.path.join(OUT, "summary.json"), "w"), default=float)
    print("\n".join(l for l in L if not l.startswith("| ") or l.startswith("| run") or l.startswith("| series")))
    return an


if __name__ == "__main__":
    report()
