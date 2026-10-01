"""Milestone 3 report: sealed-test results, baselines, frontier, nested
contexts, reproducibility, interpretation, verdict inputs."""
from __future__ import annotations
import json, os
import numpy as np
from emergence.lang.m3 import OUT


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    disc = json.load(open(os.path.join(OUT, "discovery_results.json")))
    prim = json.load(open(os.path.join(OUT, "test_primary.json")))
    base = json.load(open(os.path.join(OUT, "test_baselines.json")))
    front = json.load(open(os.path.join(OUT, "test_frontier.json")))
    an = json.load(open(os.path.join(OUT, "analysis.json")))
    frozen = json.load(open(os.path.join(OUT, "frozen.json")))
    expl = json.load(open(os.path.join(OUT, "exploratory_eps.json"))) if os.path.exists(os.path.join(OUT, "exploratory_eps.json")) else {}
    train = json.load(open(os.path.join(OUT, "train_log.json")))[-1]
    fams = ["T1", "T2", "T3", "T4"]
    L = ["# Milestone 3: sealed-test results", "",
         f"Substrate: GRU world model of the kinematic cart; held-out next-bin accuracy {train['test_acc_16']:.3f} (length 16), {train['test_acc_24']:.3f} (length 24; steps 17–24: {train['acc_24_last8']:.3f}); {prim['substrate_bits']:.3g} bits.", "",
         f"Primary model (validation rule, frozen before test): **{frozen['primary']}**, K = {prim['K']}, L(M) = {prim['complexity']['total_bits_argmax']:.0f} bits (argmax) / {prim['complexity']['total_bits_dist']:.0f} (distribution), "
         f"{prim['complexity']['exceptions']} exceptions over {prim['complexity']['n_first_fits']} first-order fits, undefined fits {prim['complexity']['residual_undefined']:.3f}.", "",
         f"Prospective predictions file `{os.path.basename(frozen['predictions_file'])}` (sha256[:16] {frozen['sha256_16']}) was written at {frozen['stamp']}, before the substrate was executed on the sealed test (see `protocol_log.txt`).", "",
         "## Sealed test: primary model and baselines (argmax fidelity; abstention = error)", "",
         "| model | K | bits | T1 interpolation | T2 unseen compositions | T3 withheld chunks | T4 longer (8–12) | T4 coverage | whole-string T1 / T4 |",
         "|---|---|---|---|---|---|---|---|---|"]
    sc = prim["scores"]
    L.append(f"| **primary interface model** | {prim['K']} | {prim['complexity']['total_bits_argmax']:.0f} | " + " | ".join(f"**{sc[f]['fidelity_argmax']:.3f}**" for f in fams)
             + f" | {sc['T4']['coverage']:.2f} | {sc['T1']['whole_string']:.2f} / {sc['T4']['whole_string']:.2f} |")
    for n, b in base.items():
        s = b["scores"]
        L.append(f"| {n} | {b['K']} | {b['bits']:.0f} | " + " | ".join(f"{s[f]['fidelity_argmax']:.3f}" for f in fams) + f" | {s['T4']['coverage']:.2f} | {s['T1']['whole_string']:.2f} / {s['T4']['whole_string']:.2f} |")
    L.append(f"| full substrate | – | {prim['substrate_bits']:.3g} | 1 | 1 | 1 | 1 | 1 | 1 / 1 |")
    L += ["", "JS fidelity (1 − JS distance) of the primary: " + ", ".join(f"{f} {sc[f]['fidelity_js']:.3f}" for f in fams), "",
          "Per-step fidelity of the primary on T4 (composition depth): " + ", ".join(f"{i+1}: {v:.2f}" for i, v in enumerate(sc["T4"]["by_step"])), "",
          f"Causal commutation (do(I = i) through up to 5 distinct realisations, 8-step T4 prefixes, {prim['causal']['n_types_tested']} interfaces tested): "
          f"realisations agree {prim['causal']['commutation']:.3f}; agree with the abstract prediction {prim['causal']['abstract_agreement']:.3f}.", "",
          f"Reuse: each interface is reached by {prim['reuse']['interface_reuse']:.1f} first-order fits on average; {prim['reuse']['fits_used']} of {prim['reuse']['of_first_order']} first-order fits are exercised by the test queries, {prim['reuse']['fit_reuse']:.0f} times each on average.", "",
          "## Frontier (all discovered models, seed 0, scored on the sealed test after freezing; not used for selection)", "",
          "| model | K | bits | validation | T1 | T2 | T3 | T4 | T4 coverage |", "|---|---|---|---|---|---|---|---|---|"]
    for n, v in sorted(front.items(), key=lambda kv: kv[1]["complexity"]["total_bits_argmax"]):
        if not n.endswith("s0"):
            continue
        s = v["scores"]
        L.append(f"| {n} | {v['K']} | {v['complexity']['total_bits_argmax']:.0f} | {disc[n]['val']['fidelity_argmax']:.3f} | " + " | ".join(f"{s[f]['fidelity_argmax']:.3f}" for f in fams) + f" | {s['T4']['coverage']:.2f} |")
    if expl:
        L += ["", "Exploratory (excluded from the verdict; grid edge check): " + "; ".join(
            f"{n}: K={v['K']}, validation {v['val']['fidelity_argmax']:.3f}, bits {v['complexity']['total_bits_argmax']:.0f}, test " + " ".join(f"{f}={v['scores'][f]['fidelity_argmax']:.3f}" for f in fams) for n, v in expl.items())]
    ne = an["nested"]
    L += ["", "## Nested context families at the primary ε (seed 0)", "",
          "| family | K | bits | validation | T1 | T2 | T3 | T4 |", "|---|---|---|---|---|---|---|---|"]
    for c in ("C1", "C2", "C3"):
        L.append(f"| {c} | {ne[c]['K']} | {ne[c]['bits']:.0f} | {ne[c]['val']:.3f} | " + " | ".join(f"{ne[c]['test'][f]:.3f}" for f in fams) + " |")
    L += ["", f"Refinement: fraction of C2 interfaces inside a single C1 interface {ne['C2_refines_C1']:.2f}; of C3 inside a single C2 interface {ne['C3_refines_C2']:.2f}; ARI(C1, C2) {ne['ari_C1_C2']:.2f}, ARI(C2, C3) {ne['ari_C2_C3']:.2f}.", "",
          "## Reproducibility at the primary (ε, C), seeds 0–2", "",
          "Representational agreement (ARI of discovery partitions): " + ", ".join(f"{k}: {v:.2f}" for k, v in an["reproducibility"]["ari"].items()), "",
          "Behavioural distance (disagreement of sealed-test predictions): " + ", ".join(f"{k}: {v:.3f}" for k, v in an["reproducibility"]["behavioural_distance"].items()), "",
          "## Interpretation (simulator unsealed after all of the above)", ""]
    I = an["interpretation"]
    L += [f"Variance of the sealed latent explained by the interface partition of the discovery pool: position R² = {I['R2_x']:.3f}, velocity R² = {I['R2_v']:.3f}. "
          f"ARI with the 8 observation bins {I['x_bin_partition_ari']:.2f}; with a position×velocity grid (8×6, {I['n_xv_cells_occupied']} cells occupied) {I['xv_grid_ari']:.2f}.",
          "", "Largest interfaces (mean ± sd of the sealed position and velocity of their realisations):", ""]
    for d in sorted(I["per_interface"], key=lambda d: -d["n"])[:12]:
        L.append(f"- I{d['k']}: n = {d['n']}, x = {d['x_mean']:.2f} ± {d['x_std']:.2f}, v = {d['v_mean']:+.3f} ± {d['v_std']:.3f}")
    xs = np.median([d["x_std"] for d in I["per_interface"]]); vs = np.median([d["v_std"] for d in I["per_interface"]])
    L += ["", f"Median within-interface spread: position {xs:.3f} (track length 1, observation bin width 0.125), velocity {vs:.3f} (range ±0.15)."]
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, v in front.items():
        if not n.endswith("s0"):
            continue
        col = {"C1": "C0", "C2": "C1", "C3": "C2"}[disc[n]["ctx"]]
        ax[0].plot(v["complexity"]["total_bits_argmax"], v["scores"]["T4"]["fidelity_argmax"], "o", color=col, alpha=0.8)
        ax[0].plot(v["complexity"]["total_bits_argmax"], v["scores"]["T1"]["fidelity_argmax"], "^", color=col, alpha=0.4)
    for n, b in base.items():
        ax[0].plot(b["bits"], b["scores"]["T4"]["fidelity_argmax"], "s", color="k", alpha=0.6); ax[0].annotate(n, (b["bits"], b["scores"]["T4"]["fidelity_argmax"]), fontsize=6)
    ax[0].plot(prim["complexity"]["total_bits_argmax"], sc["T4"]["fidelity_argmax"], "*", color="r", ms=14, label="primary (T4)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("description complexity L(M), bits"); ax[0].set_ylabel("sealed fidelity: T4 circles, T1 triangles")
    ax[0].set_title("frontier: blue C1, orange C2, green C3; black = baselines (T4)", fontsize=8); ax[0].legend(fontsize=7)
    for f in fams:
        ax[1].plot(range(1, len(sc[f]["by_step"]) + 1), sc[f]["by_step"], "o-", label=f"primary {f}")
    for n in ("kmeans_pca8", "random"):
        ax[1].plot(range(1, len(base[n]["scores"]["T4"]["by_step"]) + 1), base[n]["scores"]["T4"]["by_step"], "--", label=f"{n} T4")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("per-step fidelity"); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=7); ax[1].set_title("composition depth", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "m3_frontier.png"), dpi=120)
    open(os.path.join(OUT, "M3_REPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
