"""Progress measures of Nanda et al. (2023), Figure 7, on the checkpoints of transformer.py.

Definitions follow their Grokking_Analysis.ipynb:
  - logits: last position, the first p outputs (the "=" output is dropped), on all p^2
    pairs in the order a * p + b. For the task "ec", a = log P and b = log Q.
  - key frequencies: from the final checkpoint. Each MLP neuron's centred activation on all
    pairs is Fourier transformed over (a, b); a neuron's frequency is the f maximising the
    fraction of its squared norm in the 8 components (1, cos f), (1, sin f), (cos f, 1),
    (sin f, 1), (cos f, cos f), (cos f, sin f), (sin f, cos f), (sin f, sin f). The key
    frequencies are the distinct neuron frequencies.
  - excluded loss: subtract from each logit column its projection onto cos(w_k(a+b)) and
    sin(w_k(a+b)), k key; cross-entropy on the train pairs.
  - restricted loss: keep only those projections plus each column's mean over pairs;
    cross-entropy on all pairs.
  - Gini coefficients (Hurley and Rickard 2009) of the p Fourier component norms of the
    embedding and of W_L = W_U W_out (Fourier over the output index). Nanda et al. do not
    publish the code for these; this is the definition in their section 5.1.
    For "ec" the embedding is not indexed by group elements. The "point embedding" is used
    instead: the matrix whose column n is (W_E[:, x(P)], W_E[:, y(P)]) for P = nG.
  - total sum of squared weights: all parameters.

Run from the repository root, after both training runs:
    python progress_measures.py
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

from transformer import CKPT, CONFIG, load_model, make_data, task_spec

P = CONFIG["p"]
EPOCHS = list(range(0, CONFIG["num_epochs"], CONFIG["ckpt_every"]))


def fourier_basis(p):
    """(p, p): constant, then cos(w_k x), sin(w_k x) for k = 1..(p-1)/2; rows of unit norm."""
    x = torch.arange(p, dtype=torch.float64)
    rows = [torch.ones(p, dtype=torch.float64)]
    for k in range(1, p // 2 + 1):
        rows += [torch.cos(2 * np.pi * k * x / p), torch.sin(2 * np.pi * k * x / p)]
    B = torch.stack(rows)
    return B / B.norm(dim=1, keepdim=True)


FB = fourier_basis(P)


def all_tokens(task):
    _, _, encode = task_spec(task, P)
    return torch.tensor([encode(a, b) for a in range(P) for b in range(P)])


def train_mask(task):
    *_, tr_pairs, _ = make_data(task, P, CONFIG["frac_train"], CONFIG["seed"])
    mask = torch.zeros(P * P, dtype=torch.bool)
    mask[tr_pairs[:, 0] * P + tr_pairs[:, 1]] = True
    return mask


def xpy_directions(freq):
    """Unit vectors cos(w(a+b)), sin(w(a+b)) in R^{p^2}, as in Nanda et al."""
    c, s = FB[2 * freq - 1], FB[2 * freq]
    term = lambda u, v: (u[:, None] * v[None, :]).flatten()
    cos_dir = (term(c, c) - term(s, s)) / np.sqrt(2)
    sin_dir = (term(s, c) + term(c, s)) / np.sqrt(2)
    return cos_dir, sin_dir


def key_projection(logits, key_freqs):
    """Sum over key frequencies of the projections of each logit column on cos/sin(w(a+b))."""
    out = torch.zeros_like(logits)
    for f in key_freqs:
        for d in xpy_directions(f):
            out += d[:, None] @ (d[None, :] @ logits)
    return out


def key_frequencies(model, tokens):
    with torch.no_grad():
        _, cache = model(tokens, return_cache=True)
    acts = cache["hidden"][:, -1].double()                      # (p^2, d_mlp)
    acts = acts - acts.mean(0)
    grid = acts.reshape(P, P, -1)
    four = torch.einsum("ia,jb,abn->ijn", FB, FB, grid)          # (p, p, d_mlp)
    total = four.pow(2).sum((0, 1))
    fracs = []
    for f in range(1, P // 2 + 1):
        idx = [0, 2 * f - 1, 2 * f]
        block = four[idx][:, idx].pow(2).sum((0, 1)) - four[0, 0].pow(2)
        fracs.append(block / total)
    fracs = torch.stack(fracs)                                  # (56, d_mlp)
    alive = total > 0                                           # dead neurons are left out
    best, freq = fracs[:, alive].max(0)
    return sorted(set((freq + 1).tolist())), freq + 1, best


def gini(c):
    """Hurley and Rickard (2009), eq. 5."""
    c = torch.sort(c.abs()).values
    n = len(c)
    k = torch.arange(1, n + 1, dtype=c.dtype)
    return (1 - 2 * ((c / c.sum()) * (n - k + 0.5) / n).sum()).item()


def embedding_matrix(model, task):
    """(dim, p) matrix whose column n is the embedding of group element n."""
    W_E = model.embed.W_E.detach().double()
    if task == "modadd":
        return W_E[:, :P]
    toks = all_tokens(task)[::P][:, :2]                         # (x, y) tokens of P = nG, n = 0..p-1
    return torch.cat([W_E[:, toks[:, 0]], W_E[:, toks[:, 1]]], dim=0)


def neuron_logit_map(model):
    W_U = model.unembed.W_U.detach().double()[:, :P]             # (d_model, p)
    return W_U.T @ model.mlp.W_out.detach().double()            # (p, d_mlp)


def measures(task, key_freqs, tokens, mask, labels):
    rows = []
    for epoch in EPOCHS:
        model = load_model(task, epoch)
        with torch.no_grad():
            logits = model(tokens)[:, -1, :P].double()
        proj = key_projection(logits, key_freqs)
        restricted = proj + (logits - proj).mean(0)
        excl = logits - proj
        rows.append(dict(
            epoch=epoch,
            train=F.cross_entropy(logits[mask], labels[mask]).item(),
            test=F.cross_entropy(logits[~mask], labels[~mask]).item(),
            excluded=F.cross_entropy(excl[mask], labels[mask]).item(),
            restricted=F.cross_entropy(restricted, labels).item(),
            gini_E=gini((embedding_matrix(model, task) @ FB.T).norm(dim=0)),
            gini_L=gini((FB @ neuron_logit_map(model)).norm(dim=1)),
            sumsq=sum((w.detach().double() ** 2).sum().item() for w in model.parameters()),
        ))
        if epoch % 5000 == 0:
            r = rows[-1]
            print(f"{task} {epoch:6d}  train {r['train']:.2e}  test {r['test']:.2e}  excl {r['excluded']:.2e}  "
                  f"restr {r['restricted']:.2e}  giniE {r['gini_E']:.2f}  giniL {r['gini_L']:.2f}  "
                  f"sumsq {r['sumsq']:.0f}", flush=True)
    return {k: [r[k] for r in rows] for k in rows[0]}


def run(task):
    tokens = all_tokens(task)
    labels = torch.tensor([(a + b) % P for a in range(P) for b in range(P)])
    mask = train_mask(task)
    final = load_model(task, EPOCHS[-1])
    key_freqs, neuron_freq, neuron_frac = key_frequencies(final, tokens)
    print(f"{task}: key frequencies {key_freqs}")
    for f in key_freqs:
        sel = neuron_freq == f
        print(f"  freq {f:3d}: {sel.sum().item():3d} neurons, "
              f"median fraction explained {neuron_frac[sel].median().item():.2f}")
    m = measures(task, key_freqs, tokens, mask, labels)
    m["key_freqs"] = key_freqs
    torch.save(m, CKPT / f"{task}_seed{CONFIG['seed']}" / "measures.pt")
    return m


if __name__ == "__main__":
    for task in sys.argv[1:] or ["modadd", "ec"]:
        run(task)
