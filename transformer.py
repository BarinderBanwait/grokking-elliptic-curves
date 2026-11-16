"""One-layer transformer of Nanda et al. (2023), trained on one of two tasks.

  task "modadd": input  a b =          label (a + b) mod p          (Nanda et al.)
  task "ec":     input  x1 y1 x2 y2 =  label log(P + Q) = log P + log Q mod p,
                 where P, Q are points of E(F_p) with log P = a, log Q = b

For "ec", E is the reduction of 61a1 mod 113 in a short Weierstrass model; E(F_113) is
cyclic of order 113 and log is the discrete log with respect to a fixed generator
(data/e61a1_f113_points.csv). The pair (a, b) is the same group
element pair in both tasks, so the train/test split and the labels are identical; only the
input encoding differs.

Run from the repository root:
    python transformer.py modadd
    python transformer.py ec mps     # optional device argument
"""

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
CKPT = HERE / "checkpoints"
POINTS_CSV = HERE / "data" / "e61a1_f113_points.csv"

CONFIG = dict(
    p=113,
    d_model=128,
    n_heads=4,
    d_head=32,
    d_mlp=512,
    lr=1e-3,
    weight_decay=1.0,
    betas=(0.9, 0.98),
    frac_train=0.3,
    num_epochs=40_000,
    warmup_steps=10,
    seed=0,
    ckpt_every=100,
)


# ---------------------------------------------------------------- model
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return self.W_E[:, x].permute(1, 2, 0)  # (batch, pos, d_model)


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, n_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[: x.shape[-2]]


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, n_ctx):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(n_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * n_heads) / np.sqrt(d_model))
        self.register_buffer("mask", torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)
        scores = torch.einsum("biph,biqh->biqp", k, q)
        n = x.shape[-2]
        mask = self.mask[:n, :n]
        scores = scores * mask - 1e10 * (1 - mask)
        attn = torch.softmax(scores / np.sqrt(self.d_head), dim=-1)
        z = torch.einsum("biph,biqp->biqh", v, attn)
        z = z.permute(0, 2, 1, 3).reshape(x.shape[0], n, -1)  # (batch, pos, n_heads * d_head)
        return z @ self.W_O.T


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, return_hidden=False):
        h = torch.relu(x @ self.W_in.T + self.b_in)
        out = h @ self.W_out.T + self.b_out
        return (out, h) if return_hidden else out


class Transformer(nn.Module):
    def __init__(self, d_vocab, n_ctx, d_vocab_out, d_model=128, n_heads=4, d_head=32, d_mlp=512, **_):
        super().__init__()
        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.attn = Attention(d_model, n_heads, d_head, n_ctx)
        self.mlp = MLP(d_model, d_mlp)
        self.unembed = Unembed(d_vocab_out, d_model)

    def forward(self, tokens, return_cache=False):
        resid0 = self.pos_embed(self.embed(tokens))
        resid1 = resid0 + self.attn(resid0)
        mlp_out, hidden = self.mlp(resid1, return_hidden=True)
        resid2 = resid1 + mlp_out
        logits = self.unembed(resid2)
        if return_cache:
            return logits, dict(resid0=resid0, resid1=resid1, hidden=hidden, resid2=resid2)
        return logits


# ---------------------------------------------------------------- data
def load_points(path=POINTS_CSV):
    """List of (x, y) indexed by log; x = y = None for the point at infinity."""
    pts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            assert int(row["log"]) == len(pts)
            pts.append((None, None) if row["x"] == "inf" else (int(row["x"]), int(row["y"])))
    return pts


def task_spec(task, p):
    """Vocabulary size, context length and a function (a, b) -> token list."""
    eq = p  # "=" is token p in both tasks
    if task == "modadd":
        return p + 1, 3, lambda a, b: [a, b, eq]
    if task == "ec":
        inf = p + 1
        pts = load_points()
        assert len(pts) == p
        enc = [(inf, inf) if x is None else (x, y) for x, y in pts]
        return p + 2, 5, lambda a, b: [*enc[a], *enc[b], eq]
    raise ValueError(task)


def make_data(task, p, frac_train, seed):
    """All p^2 pairs (a, b), shuffled with random.seed(seed); first frac_train is train.

    Returns (train_tokens, train_labels, test_tokens, test_labels, train_pairs, test_pairs).
    """
    _, _, encode = task_spec(task, p)
    pairs = [(a, b) for a in range(p) for b in range(p)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train * len(pairs))

    def to_tensors(ps):
        tokens = torch.tensor([encode(a, b) for a, b in ps])
        labels = torch.tensor([(a + b) % p for a, b in ps])
        return tokens, labels, torch.tensor(ps)

    tr_tok, tr_lab, tr_pairs = to_tensors(pairs[:div])
    te_tok, te_lab, te_pairs = to_tensors(pairs[div:])
    return tr_tok, tr_lab, te_tok, te_lab, tr_pairs, te_pairs


def init_model(task, cfg=CONFIG):
    """Nanda et al.'s initialisation under torch.manual_seed(seed), shared by both tasks.

    The modadd model is built exactly as in Nanda et al. (d_vocab = d_vocab_out = p + 1,
    n_ctx = 3). The ec model copies every one of its parameters and draws only what it has
    in addition: the embedding column of the token "inf", and the positional embeddings of
    the y1 and y2 positions. Positions x1, x2, "=" take the positional embeddings of a, b, "=".
    The unembedding has p + 1 outputs in both tasks.
    """
    p, d_model = cfg["p"], cfg["d_model"]
    torch.manual_seed(cfg["seed"])
    base = Transformer(p + 1, 3, p + 1, **cfg)
    if task == "modadd":
        return base
    inf_col = torch.randn(d_model, 1) / np.sqrt(d_model)
    y_pos = torch.randn(2, d_model) / np.sqrt(d_model)
    d_vocab, n_ctx, _ = task_spec(task, p)
    model = Transformer(d_vocab, n_ctx, p + 1, **cfg)
    state = base.state_dict()
    state["embed.W_E"] = torch.cat([state["embed.W_E"], inf_col], dim=1)
    a_pos, b_pos, eq_pos = state["pos_embed.W_pos"]
    state["pos_embed.W_pos"] = torch.stack([a_pos, y_pos[0], b_pos, y_pos[1], eq_pos])
    state["attn.mask"] = model.attn.mask
    model.load_state_dict(state)
    return model


def loss_fn(logits, labels):
    """Cross-entropy at the last position, in float64, softmax over all p + 1 outputs.

    Nanda et al.'s notebook casts to float64; their helpers.py casts to float32. MPS has no
    float64, so the logits are moved to the CPU first; labels live on the CPU."""
    logits = logits[:, -1].cpu().to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    return -log_probs.gather(dim=-1, index=labels[:, None])[:, 0].mean()


def accuracy(logits, labels):
    return (logits[:, -1].argmax(dim=-1).cpu() == labels).float().mean().item()


# ---------------------------------------------------------------- training
def train(task, cfg=CONFIG, device="cpu"):
    p = cfg["p"]
    tr_tok, tr_lab, te_tok, te_lab, _, _ = make_data(task, p, cfg["frac_train"], cfg["seed"])
    tr_tok, te_tok = tr_tok.to(device), te_tok.to(device)

    model = init_model(task, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                            betas=cfg["betas"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / cfg["warmup_steps"], 1))

    run_dir = CKPT / f"{task}_seed{cfg['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = dict(epoch=[], train_loss=[], test_loss=[], train_acc=[], test_acc=[])
    t0 = time.time()
    # As in Nanda et al.: losses for epoch e are measured before the step of epoch e, and the
    # checkpoint "epoch_e" is saved after it. The step of epoch 0 has learning rate 0.
    for epoch in range(cfg["num_epochs"]):
        logits = model(tr_tok)
        train_loss = loss_fn(logits, tr_lab)
        if epoch % cfg["ckpt_every"] == 0:
            with torch.no_grad():
                test_logits = model(te_tok)
                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss.item())
                history["test_loss"].append(loss_fn(test_logits, te_lab).item())
                history["train_acc"].append(accuracy(logits, tr_lab))
                history["test_acc"].append(accuracy(test_logits, te_lab))
            if epoch % 1000 == 0:
                print(f"{task} epoch {epoch:6d}  train {history['train_loss'][-1]:.3e}  "
                      f"test {history['test_loss'][-1]:.3e}  test acc {history['test_acc'][-1]:.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        train_loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        if epoch % cfg["ckpt_every"] == 0:
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, run_dir / f"epoch_{epoch}.pt")
    torch.save(dict(config=cfg, task=task, device=device, history=history), run_dir / "history.pt")
    return model, history


def load_model(task, epoch, seed=0, cfg=CONFIG):
    d_vocab, n_ctx, _ = task_spec(task, cfg["p"])
    model = Transformer(d_vocab, n_ctx, cfg["p"] + 1, **cfg)
    model.load_state_dict(torch.load(CKPT / f"{task}_seed{seed}" / f"epoch_{epoch}.pt"))
    return model.eval()


if __name__ == "__main__":
    train(sys.argv[1], device=sys.argv[2] if len(sys.argv) > 2 else "cpu")
