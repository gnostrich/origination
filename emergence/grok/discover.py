"""The discovery experiment: an ordinary sequence task with no planted algebra.

Source.  A random recurrent generator (tanh Elman network, ``d_teacher``
hidden units, vocabulary ``V``, gain ``g``) sampled through a softmax.  Its
specification contains no group, monoid, table, state partition or class
ontology; whether its conditional structure has finitely many behavioural
types is unknown to us in advance.  The teacher's own quotient, extracted
with the same machinery, is the task's structure (not written down by us).

Student.  A GRU trained by next-token cross-entropy on teacher sequences,
checkpointed through training.

Frozen pipeline (unchanged machinery, data-distribution pools):
  interface search → behavioural classes of prefixes under a suffix pool →
  stability under noise → transitions (class, token) → class → canonical
  automaton, monoid invariants, coherence.

Tests, all label-free (levels A–E in the README):
  * B  predictive sufficiency on *unseen* prefixes and *unseen* suffixes,
       against random partitions with the same class sizes;
  * C  mid-sequence patching: replace the state at a random position of a
       held-out sequence by its class representative (vs a state from another
       class) and compare the following predictions;
  * D  closure/coherence of the induced transitions;
  * E  invariants of the generated monoid (associativity of the action is
       automatic and is not counted);
  * threshold stability (eps), cross-seed ARI / automaton identity,
    student vs teacher quotient, untrained and weight-shuffled controls,
    checkpoint trajectory against test loss.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .compare import adjusted_rand_index
from .extract import _js
from .extract_seq import SeqExtractConfig, _behaviour, _classify, canonical_automaton, extract_seq
from .interface import search_interface, trace_substrate
from .rnn import GRUWorldModel
from .seqsub import GRUSubstrate


# ---------------------------------------------------------------------------
# source


class Teacher:
    def __init__(self, V=6, d=12, gain=2.5, out_scale=3.0, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.d, self.gain, self.out_scale = V, d, gain, out_scale
        self.W = rng.standard_normal((d, d)) / np.sqrt(d)
        self.E = rng.standard_normal((V, d))
        self.b = rng.standard_normal(d) * 0.3
        self.Vout = rng.standard_normal((d, V)) / np.sqrt(d)
        self.h0 = np.zeros(d)

    def step(self, h, x):  # h (B,d), x (B,) tokens -> h'
        return np.tanh(self.gain * (h @ self.W + self.E[x] + self.b))

    def logits(self, h):
        return self.out_scale * (h @ self.Vout)

    def sample(self, n, L, rng):
        h = np.repeat(self.h0[None], n, 0)
        x = rng.integers(0, self.V, n)
        seq = [x]
        for t in range(L - 1):
            h = self.step(h, x)
            p = np.exp(self.logits(h)); p /= p.sum(1, keepdims=True)
            x = np.array([rng.choice(self.V, p=pi) for pi in p])
            seq.append(x)
        return np.stack(seq, 1)

    def entropy_rate(self, seqs):
        """Mean conditional entropy of the next token along the given sequences (irreducible loss)."""
        n, L = seqs.shape
        h = np.repeat(self.h0[None], n, 0)
        H = []
        for t in range(L - 1):
            h = self.step(h, seqs[:, t])
            p = np.exp(self.logits(h)); p /= p.sum(1, keepdims=True)
            H.append(-(p * np.log(p + 1e-12)).sum(1))
        return float(np.mean(H))


class TeacherSubstrate:
    """The teacher as a sequence substrate (its hidden state is the configuration)."""

    def __init__(self, T: Teacher):
        self.T = T; self.d = T.d

    def run_prefixes(self, prefixes):
        H = torch.empty(len(prefixes), self.d)
        for i, p in enumerate(prefixes):
            h = self.T.h0[None].copy()
            for x in p:
                h = self.T.step(h, np.array([x]))
            H[i] = torch.tensor(h[0], dtype=torch.float32)
        return H

    def continue_(self, cfg, actions):
        h = cfg.numpy().astype(float)
        if h.shape[0] != actions.shape[0]:
            h = np.repeat(h, actions.shape[0], 0)
        A = actions.numpy()
        out = []
        for t in range(A.shape[1]):
            h = self.T.step(h, A[:, t])
            out.append(self.T.logits(h))
        return torch.tensor(np.stack(out, 1), dtype=torch.float32), torch.tensor(h, dtype=torch.float32)

    def noise(self, cfg, sigma):
        rms = cfg.pow(2).mean(-1, keepdim=True).sqrt().mean()
        return cfg + sigma * rms * torch.randn_like(cfg)

    def select(self, cfg, idx):
        return cfg[idx]

    def batch_size(self, cfg):
        return cfg.shape[0]


# ---------------------------------------------------------------------------
# student


@dataclass
class DiscoverConfig:
    V: int = 6
    d_teacher: int = 12
    gain: float = 2.5
    teacher_seed: int = 0
    seq_len: int = 32
    n_train: int = 8192
    n_test: int = 2048
    d_student: int = 64
    steps: int = 6000
    batch: int = 256
    lr: float = 2e-3
    wd: float = 0.01
    ckpt_every: int = 250
    seed: int = 0
    threads: int = 1


def train_student(cfg: DiscoverConfig, teacher: Teacher, data, out_dir: Path):
    torch.set_num_threads(cfg.threads); torch.manual_seed(cfg.seed)
    (Atr, Ate) = data
    Xtr = torch.tensor(Atr[:, :-1]); Ytr = torch.tensor(Atr[:, 1:])
    Xte = torch.tensor(Ate[:, :-1]); Yte = torch.tensor(Ate[:, 1:])
    model = GRUWorldModel(cfg.V, cfg.V, d=cfg.d_student)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = set([0, 10, 25, 50, 100] + list(range(cfg.ckpt_every, cfg.steps + 1, cfg.ckpt_every)))
    log, ckpts = [], {}
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    for step in range(cfg.steps + 1):
        if step in sched:
            model.eval()
            with torch.no_grad():
                ltr, _ = model(Xtr[:2048]); lte, _ = model(Xte)
                r = {"step": step, "train_loss": float(F.cross_entropy(ltr.reshape(-1, cfg.V), Ytr[:2048].reshape(-1))),
                     "test_loss": float(F.cross_entropy(lte.reshape(-1, cfg.V), Yte.reshape(-1))),
                     "test_acc": float((lte.argmax(-1) == Yte).float().mean()), "elapsed_s": time.time() - t0}
            model.train(); log.append(r)
            ckpts[step] = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
            print(f"[student s{cfg.seed} g{cfg.gain}] step {step:5d} train {r['train_loss']:.3f} test {r['test_loss']:.3f} acc {r['test_acc']:.3f}", flush=True)
        if step == cfg.steps:
            break
        bi = torch.tensor(rng.integers(0, len(Xtr), cfg.batch))
        lg, _ = model(Xtr[bi])
        loss = F.cross_entropy(lg.reshape(-1, cfg.V), Ytr[bi].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "checkpoints": ckpts, "log": log}, out_dir / "checkpoints.pt")
    json.dump({"config": asdict(cfg), "log": log}, open(out_dir / "train_log.json", "w"), indent=1)
    return model, ckpts, log


# ---------------------------------------------------------------------------
# pools and tests


def make_pools(seqs, n_prefix, n_suffix, suffix_len, rng, max_len=16):
    """Prefixes: initial segments of held-out sequences (plus the empty prefix).
    Suffixes: random windows of held-out sequences."""
    prefixes = [()]
    for _ in range(n_prefix - 1):
        i = rng.integers(len(seqs)); L = int(rng.integers(1, max_len + 1))
        prefixes.append(tuple(int(x) for x in seqs[i, :L]))
    suff = []
    for _ in range(n_suffix):
        i = rng.integers(len(seqs)); s = int(rng.integers(0, seqs.shape[1] - suffix_len))
        suff.append(seqs[i, s:s + suffix_len])
    return prefixes, torch.tensor(np.stack(suff), dtype=torch.long)


def sufficiency_unseen(sub, res, eps, seqs, rng, n_new=300, n_suff=24, suffix_len=6, n_random=100):
    """Level B: unseen prefixes classified by behaviour on the training suffix pool;
    do they behave like their class representative on *fresh* suffixes?  Compared
    with random partitions of the same sizes."""
    leaders, rep_idx, suff_train = res["_leaders"], res["_rep_idx"], res["_suffixes"]
    new_prefixes, suff_new = make_pools(seqs, n_new, n_suff, suffix_len, rng)
    Hn = sub.run_prefixes(new_prefixes)
    B_train = _behaviour(sub, Hn, suff_train)
    labels = _classify(B_train, leaders, eps)
    ok = labels >= 0
    B_new = _behaviour(sub, Hn, suff_new)
    # representatives' behaviour on the fresh suffixes
    reps_H = sub.run_prefixes([res["_prefixes"][i] for i in rep_idx])
    B_rep_new = _behaviour(sub, reps_H, suff_new)
    d = np.array([_js(B_new[i][None], B_rep_new[labels[i]][None]).mean() if ok[i] else np.nan for i in range(len(new_prefixes))])
    match = float(np.nanmean(d < eps)) if ok.any() else 0.0
    mean_d = float(np.nanmean(d)) if ok.any() else float("nan")
    # random partitions with the same class sizes: assign each new prefix a random class
    rand = []
    k = len(rep_idx)
    for _ in range(n_random):
        rl = rng.integers(0, k, len(new_prefixes))
        dr = np.array([_js(B_new[i][None], B_rep_new[rl[i]][None]).mean() for i in range(len(new_prefixes))])
        rand.append(float(np.mean(dr < eps)))
    return {"frac_unseen_classified": float(ok.mean()), "match_on_fresh_suffixes": match, "mean_js_to_rep": mean_d,
            "random_partition_match_mean": float(np.mean(rand)), "random_partition_match_max": float(np.max(rand))}


def patching_test(sub, res, eps, seqs, rng, n=200, horizon=8):
    """Level C: at a random position t of a held-out sequence, replace the
    state by (a) the representative of the state's class, (b) a representative
    of another class, (c) a random other state of the same class; compare the
    next-token predictions over the following ``horizon`` tokens (mean JS)."""
    leaders, rep_idx, suff_train = res["_leaders"], res["_rep_idx"], res["_suffixes"]
    reps_H = sub.run_prefixes([res["_prefixes"][i] for i in rep_idx])
    k = len(rep_idx)
    out = {"same_rep": [], "other_rep": [], "same_member": []}
    members = {c: [i for i, l in enumerate(res["partition"]) if l == c] for c in range(k)}
    Hall = sub.run_prefixes(res["_prefixes"])
    for _ in range(n):
        i = rng.integers(len(seqs)); t = int(rng.integers(1, seqs.shape[1] - horizon))
        prefix = tuple(int(x) for x in seqs[i, :t]); rest = torch.tensor(seqs[i, t:t + horizon][None], dtype=torch.long)
        h = sub.run_prefixes([prefix])
        lab = int(_classify(_behaviour(sub, h, suff_train), leaders, eps)[0])
        if lab < 0:
            continue
        lg0, _ = sub.continue_(h, rest); p0 = F.softmax(lg0, -1).numpy()
        def js_from(hp):
            lg, _ = sub.continue_(hp, rest); return float(_js(F.softmax(lg, -1).numpy()[0], p0[0]).mean())
        out["same_rep"].append(js_from(reps_H[lab][None]))
        other = int(rng.choice([c for c in range(k) if c != lab])) if k > 1 else lab
        out["other_rep"].append(js_from(reps_H[other][None]))
        mem = members[lab]
        if len(mem) > 1:
            j = int(rng.choice(mem)); out["same_member"].append(js_from(Hall[j][None]))
    return {key: {"mean_js": float(np.mean(v)) if v else float("nan"), "frac_below_eps": float(np.mean(np.array(v) < eps)) if v else float("nan"), "n": len(v)}
            for key, v in out.items()}


def explained_behaviour(res, eps):
    """How much of the behavioural dispersion the partition explains: within-class
    mean JS to the class representative vs. total mean JS to a random representative."""
    return {"n_classes": res["n_classes"], "n_stable": res["n_stable_classes"] if "n_stable_classes" in res else None}


def run_extraction(sub, V, prefixes, suffixes, eps, rng):
    cfg = SeqExtractConfig(eps_beh=eps)
    res = extract_seq(sub, V, cfg, rng, prefixes=prefixes, suffixes=suffixes)
    res["_prefixes"] = prefixes
    return res


def summarize(res):
    mon = res["monoid"]
    return {"n_classes": res["n_classes"], "n_reachable": res["n_reachable_classes"], "metastability": res["metastability"],
            "closure": res["closure"], "frac_products_known": res["frac_products_known"], "coherence": res["action_coherence"],
            "discreteness": res["discreteness"], "lawfree": res["crystallization_lawfree"], "nondegenerate": res["nondegenerate"],
            "monoid": {k: v for k, v in mon.items() if k != "actions_bijective"}, "canonical_table": res["canonical_table"]}


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/discover")
    ap.add_argument("--gain", type=float, default=2.5)
    ap.add_argument("--teacher_seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--n_prefix", type=int, default=500)
    ap.add_argument("--phase", default="all", help="train | analyse | all")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out) / f"g{a.gain}_t{a.teacher_seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg0 = DiscoverConfig(gain=a.gain, teacher_seed=a.teacher_seed, steps=a.steps, threads=a.threads)
    teacher = Teacher(V=cfg0.V, d=cfg0.d_teacher, gain=a.gain, seed=a.teacher_seed)
    drng = np.random.default_rng(1000 + a.teacher_seed)
    Atr = teacher.sample(cfg0.n_train, cfg0.seq_len, drng)
    Ate = teacher.sample(cfg0.n_test, cfg0.seq_len, drng)
    Aheld = teacher.sample(2048, cfg0.seq_len, drng)  # for pools / unseen tests
    H_rate = teacher.entropy_rate(Ate)
    json.dump({"teacher": {"V": cfg0.V, "d": cfg0.d_teacher, "gain": a.gain, "seed": a.teacher_seed},
               "entropy_rate": H_rate}, open(out / "teacher.json", "w"), indent=1)
    print(f"teacher gain {a.gain} seed {a.teacher_seed}: conditional entropy rate {H_rate:.3f} nats (uniform {np.log(cfg0.V):.3f})", flush=True)
    seeds = [int(s) for s in a.seeds.split(",")]

    if a.phase in ("train", "all"):
        for s in seeds:
            train_student(replace(cfg0, seed=s), teacher, (Atr, Ate), out / f"student_s{s}")

    if a.phase in ("analyse", "all"):
        prng = np.random.default_rng(7)
        prefixes, suffixes = make_pools(Aheld, a.n_prefix, 24, 6, prng)
        report = {"entropy_rate": H_rate, "students": {}, "teacher": None, "controls": {}}
        # teacher's own quotient (the task's structure)
        tsub = TeacherSubstrate(teacher)
        tres = run_extraction(tsub, cfg0.V, prefixes, suffixes, a.eps, np.random.default_rng(3))
        report["teacher"] = summarize(tres)
        report["teacher"]["partition"] = tres["partition"]
        print(f"teacher quotient: classes {tres['n_classes']} reach {tres['n_reachable_classes']} meta {tres['metastability']:.2f} "
              f"closure {tres['closure']:.2f} coh {tres['action_coherence']:.2f} monoid {tres['monoid']}", flush=True)
        for s in seeds:
            d = torch.load(out / f"student_s{s}" / "checkpoints.pt", weights_only=False)
            model = GRUWorldModel(cfg0.V, cfg0.V, d=cfg0.d_student)
            sub = GRUSubstrate(model)
            entry = {"log": d["log"], "trajectory": []}
            steps = sorted(d["checkpoints"])
            traj_steps = sorted(set([steps[0], steps[-1]] + [steps[i] for i in np.unique(np.round(np.geomspace(1, len(steps) - 1, 8)).astype(int))]))
            for step in traj_steps:
                model.load_state_dict(d["checkpoints"][step]); model.eval()
                r = run_extraction(sub, cfg0.V, prefixes, suffixes, a.eps, np.random.default_rng(3))
                tl = {x["step"]: x for x in d["log"]}[step]
                sm = summarize(r); sm["step"] = step; sm["test_loss"] = tl["test_loss"]
                sm["ari_to_teacher"] = adjusted_rand_index(r["partition"], tres["partition"])
                entry["trajectory"].append(sm)
                print(f"[s{s}] step {step:5d} test {tl['test_loss']:.3f} classes {r['n_classes']:4d} reach {r['n_reachable_classes']:3d} "
                      f"meta {r['metastability']:.2f} clos {r['closure']:.2f} coh {r['action_coherence']:.2f} monoid {r['monoid'].get('monoid_order','n/a')} "
                      f"ARI(teacher) {sm['ari_to_teacher']:.3f}", flush=True)
            # final checkpoint: full battery of tests
            model.load_state_dict(d["checkpoints"][steps[-1]]); model.eval()
            r = run_extraction(sub, cfg0.V, prefixes, suffixes, a.eps, np.random.default_rng(3))
            entry["final"] = summarize(r); entry["final"]["partition"] = r["partition"]
            entry["final"]["sufficiency_unseen"] = sufficiency_unseen(sub, r, a.eps, Aheld, np.random.default_rng(5))
            entry["final"]["patching"] = patching_test(sub, r, a.eps, Aheld, np.random.default_rng(6))
            entry["final"]["eps_sweep"] = {}
            for e in (0.02, 0.1, 0.2):
                re_ = run_extraction(sub, cfg0.V, prefixes, suffixes, e, np.random.default_rng(3))
                entry["final"]["eps_sweep"][str(e)] = {"n_classes": re_["n_classes"], "n_reachable": re_["n_reachable_classes"],
                                                        "closure": re_["closure"], "monoid_order": re_["monoid"].get("monoid_order"),
                                                        "ari_to_eps": adjusted_rand_index(re_["partition"], r["partition"])}
            # interface search on data-distribution pools
            tsb = trace_substrate("gru", model)
            L = 6
            don = torch.tensor(np.stack([Aheld[i, :L] for i in np.random.default_rng(8).integers(0, len(Aheld), 48)]), dtype=torch.long)
            rec = torch.tensor(np.stack([Aheld[i, :L] for i in np.random.default_rng(9).integers(0, len(Aheld), 48)]), dtype=torch.long)
            isr = search_interface(tsb, cfg0.V, L, np.random.default_rng(10), donors=don, recips=rec, suffixes=suffixes[:8])
            entry["final"]["interface"] = {"best": isr["best"], "full_complexity": isr["full_complexity"], "baseline": isr["baseline_sufficiency"]}
            print(f"[s{s}] final: unseen-sufficiency {entry['final']['sufficiency_unseen']} patching {entry['final']['patching']} "
                  f"interface {isr['best']['cells']} ({isr['best']['complexity']})", flush=True)
            # controls: untrained (step 0) handled in the trajectory; weight-shuffled model
            model.load_state_dict(d["checkpoints"][steps[-1]])
            with torch.no_grad():
                for p in model.parameters():
                    flat = p.view(-1); perm = torch.randperm(flat.numel()); flat.copy_(flat[perm])
            rsh = run_extraction(sub, cfg0.V, prefixes, suffixes, a.eps, np.random.default_rng(3))
            entry["shuffled_control"] = summarize(rsh)
            print(f"[s{s}] shuffled-weights control: classes {rsh['n_classes']} reach {rsh['n_reachable_classes']} meta {rsh['metastability']:.2f} closure {rsh['closure']:.2f}", flush=True)
            report["students"][s] = entry
        # cross-seed
        names = list(report["students"])
        cross = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pi, pj = report["students"][names[i]]["final"]["partition"], report["students"][names[j]]["final"]["partition"]
                Ci, Cj = np.array(report["students"][names[i]]["final"]["canonical_table"]), np.array(report["students"][names[j]]["final"]["canonical_table"])
                cross[f"s{names[i]} vs s{names[j]}"] = {"ari": adjusted_rand_index(pi, pj),
                                                        "automata_identical": bool(Ci.shape == Cj.shape and np.array_equal(Ci, Cj))}
        report["cross_seed"] = cross
        print("cross-seed:", cross, flush=True)
        json.dump(report, open(out / "report.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
