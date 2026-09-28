"""Full-batch AdamW training with checkpoints.

Checkpoints are kept as CPU state dicts (small models) and written to disk
so extraction can be re-run without retraining.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .models import build
from .task import all_pairs, labels, split


@dataclass
class TrainConfig:
    arch: str = "transformer"
    p: int = 97
    train_frac: float = 0.3
    seed: int = 0
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple = (0.9, 0.98)
    max_steps: int = 25000
    ckpt_every: int = 250
    stop_after_grok: int = 3000  # keep training this many steps after test acc >= grok_acc
    grok_acc: float = 0.99
    threads: int = 1


def checkpoint_schedule(max_steps: int, every: int) -> list[int]:
    """Dense early (log-spaced) then every ``every`` steps."""
    s = set([0, 10, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000])
    s.update(range(every, max_steps + 1, every))
    return sorted(x for x in s if x <= max_steps)


def train(cfg: TrainConfig, out_dir: Path, verbose: bool = True) -> dict:
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = all_pairs(cfg.p)
    Y = labels(cfg.p)
    tr, te = split(cfg.p, cfg.train_frac, cfg.seed)
    model = build(cfg.arch, cfg.p)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    sched = set(checkpoint_schedule(cfg.max_steps, cfg.ckpt_every))
    log = []
    ckpts = {}
    t0 = time.time()
    grok_step = None

    def evaluate():
        model.eval()
        with torch.no_grad():
            logits, _ = model(X)
            loss_all = F.cross_entropy(logits, Y, reduction="none")
            acc_all = (logits.argmax(-1) == Y).float()
        model.train()
        return {
            "train_loss": float(loss_all[tr].mean()),
            "test_loss": float(loss_all[te].mean()),
            "train_acc": float(acc_all[tr].mean()),
            "test_acc": float(acc_all[te].mean()),
        }

    for step in range(cfg.max_steps + 1):
        if step in sched:
            ev = evaluate()
            ev["step"] = step
            ev["elapsed_s"] = time.time() - t0
            log.append(ev)
            ckpts[step] = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
            if verbose:
                print(f"[{cfg.arch} s{cfg.seed}] step {step:6d} train {ev['train_acc']:.3f}/{ev['train_loss']:.3f} "
                      f"test {ev['test_acc']:.3f}/{ev['test_loss']:.3f} ({ev['elapsed_s']:.0f}s)", flush=True)
            if grok_step is None and ev["test_acc"] >= cfg.grok_acc:
                grok_step = step
            if grok_step is not None and step >= grok_step + cfg.stop_after_grok:
                break
        if step == cfg.max_steps:
            break
        logits, _ = model(X[tr])
        loss = F.cross_entropy(logits, Y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()

    torch.save({"config": asdict(cfg), "checkpoints": ckpts, "log": log}, out_dir / "checkpoints.pt")
    with open(out_dir / "train_log.json", "w") as f:
        json.dump({"config": asdict(cfg), "log": log, "grok_step": grok_step}, f, indent=1)
    return {"config": asdict(cfg), "log": log, "grok_step": grok_step, "checkpoints": ckpts}
