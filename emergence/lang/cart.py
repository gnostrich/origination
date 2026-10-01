"""Milestone 3 system: a one-dimensional kinematic cart under discrete
control, and the GRU world model trained to predict its quantised position.

The simulator is SEALED: it generates training data and is opened again
only in the interpretation step of the protocol.  Discovery never imports
anything from it except the sizes of the action and observation alphabets.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from emergence.grok.rnn import GRUWorldModel

N_ACTIONS = 4
N_OBS = 8
VMAX = 0.15
PUSH = 0.05
DECAY = 0.95
BRAKE = 0.5
X0, V0 = 0.5, 0.0


def simulate(actions):
    """actions (B, L) ints -> observations (B, L) bins, final states (B, 2)."""
    B, L = actions.shape
    x = np.full(B, X0)
    v = np.full(B, V0)
    obs = np.empty((B, L), dtype=int)
    for t in range(L):
        a = actions[:, t]
        v = np.where(a == 3, BRAKE * v, DECAY * v + PUSH * ((a == 1).astype(float) - (a == 0).astype(float)))
        v = np.clip(v, -VMAX, VMAX)
        x = x + v
        lo, hi = x < 0, x > 1
        x = np.where(lo, -x, np.where(hi, 2 - x, x))
        v = np.where(lo | hi, -v, v)
        x = np.clip(x, 0, 1)
        obs[:, t] = np.clip((x * N_OBS).astype(int), 0, N_OBS - 1)
    return obs, np.stack([x, v], 1)


def state_after(prefix):
    if len(prefix) == 0:
        return np.array([X0, V0])
    return simulate(np.asarray(prefix, dtype=int)[None])[1][0]


def make_data(n_train=20000, n_test=2000, L=16, L_long=24, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, N_ACTIONS, (n_train, L))
    At = rng.integers(0, N_ACTIONS, (n_test, L))
    Al = rng.integers(0, N_ACTIONS, (n_test, L_long))
    return (A, simulate(A)[0]), (At, simulate(At)[0]), (Al, simulate(Al)[0])


def train(out_dir, steps=4000, batch=512, lr=2e-3, wd=0.1, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    (A, O), (At, Ot), (Al, Ol) = make_data(seed=seed)
    A, O = torch.as_tensor(A), torch.as_tensor(O)
    model = GRUWorldModel(N_ACTIONS, N_OBS)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    log = []
    t0 = time.time()

    def evaluate(Ax, Ox):
        model.eval()
        with torch.no_grad():
            lg, _ = model(torch.as_tensor(Ax))
            acc = (lg.argmax(-1) == torch.as_tensor(Ox)).float()
            loss = F.cross_entropy(lg.reshape(-1, N_OBS), torch.as_tensor(Ox).reshape(-1))
        model.train()
        return float(acc.mean()), float(loss), [float(a) for a in acc.mean(0)]

    for step in range(steps + 1):
        if step % 250 == 0 or step == steps:
            acc16, loss16, _ = evaluate(At, Ot)
            acc24, loss24, per_t = evaluate(Al, Ol)
            rec = dict(step=step, test_acc_16=acc16, test_loss_16=loss16, test_acc_24=acc24, test_loss_24=loss24,
                       acc_24_last8=float(np.mean(per_t[16:])), elapsed=time.time() - t0)
            log.append(rec)
            if verbose:
                print(f"step {step:5d} test acc L16 {acc16:.3f} L24 {acc24:.3f} (steps 17-24: {rec['acc_24_last8']:.3f})", flush=True)
        if step == steps:
            break
        idx = torch.randint(0, len(A), (batch,))
        lg, _ = model(A[idx])
        loss = F.cross_entropy(lg.reshape(-1, N_OBS), O[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save({"state": model.state_dict(), "log": log, "config": dict(steps=steps, batch=batch, lr=lr, wd=wd, seed=seed)},
               os.path.join(out_dir, "cart_gru.pt"))
    json.dump(log, open(os.path.join(out_dir, "train_log.json"), "w"))
    return model, log


def load(out_dir):
    d = torch.load(os.path.join(out_dir, "cart_gru.pt"), weights_only=False)
    m = GRUWorldModel(N_ACTIONS, N_OBS)
    m.load_state_dict(d["state"])
    m.eval()
    return m, d["log"]


if __name__ == "__main__":
    train("results/lang/m3")
