#!/usr/bin/env python3
"""
Quantum Transformer Demo (Ehands v2 + QCrank )
This could potentially improved by QFT 
==========================================================
Run with:
    python hybrid_quantum_transformer.py

What it does
------------
* Embeds tokens and positions with PyTorch.
* One attention head uses **2‑qubit e‑hands** (weighted sum on qubit‑0).
* Optional second head applies **QCrank + 3‑qubit QFT** (auto‑disabled if
  QCrank is not installed).
* Causal masking for autoregressive tasks.
* Bottom‑of‑file `__main__` block runs a quick forward smoke test.  Set
  `RUN_TRAIN_DEMO = True` to see one classical training epoch then quantum
  inference.

Device: cuda, QCrank: True
Logits tensor : (1, 6, 100)
e-hands runs  : 36            # seq_len²  circuits
QCrank runs   : 18            # seq_len × 3 circuits (Q,K,V)
Raw ⟨Z0⟩ scores from e-hands:
+0.042 -0.015 +0.061 -0.010 +0.035 -0.007
-0.028 +0.039 +0.018 +0.005 +0.011 -0.022
Dependencies: `torch`, `numpy`, `qiskit-aer>=0.14`.  QCrank is optional.
"""

# ─────────────────────────── imports ───────────────────────────
import math
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

import sys,os
sys.path.append(str(Path("/qcrank_light")))
from datacircuits.ParametricQCrankV2 import ParametricQCrankV2
from datacircuits.qm import QuantumMultiplier
from datacircuits.qc import QCrankEncoder
# ───────────────────── optional QCrank support ─────────────────────
EHANDS_AVAILABLE = True
QCRANK_AVAILABLE = True

def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Lower‑triangular mask (True keeps)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return mask if device is None else mask.to(device)


def apply_causal_mask(scores: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
    causal = create_causal_mask(scores.size(-1), scores.device)
    keep = causal if extra is None else (causal & extra)
    return scores.masked_fill(~keep, float("-inf"))

# ------------------------------------------------------------------
#   QuantumAttention block (2 heads)
# ------------------------------------------------------------------
def causal_mask(L: int, device) -> torch.Tensor:
    return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

class AngleMLP(nn.Module):
    """d → hidden → D  and squash to (-1,1) for arccos."""
    def __init__(self, d_in: int, D_out: int, hidden: int | None = None):
        super().__init__()
        h = hidden or 4 * d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Linear(h, D_out),
            nn.Tanh()                  # guarantees angles inside (-1,1)
        )

    def forward(self, x):
        return self.net(x)

# ───────────────────── autograd wrapper ──────────────────────
class QMScore(torch.autograd.Function):
    """
    Forward  : identical to _scores_qm_batch (or _scores_qm)
               -> returns S  (B, L, L)
    Backward : sends a classical-dot-product style gradient
               so that training does not crash.
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, qm):
        """
        Q, K : (B, L, F)
        qm   : QuantumMultiplier instance (opaque to PyTorch)
        out  : (B, L, L)
        """
        ctx.save_for_backward(Q, K)
        ctx.qm = qm

        B, L, F = Q.shape
        S = torch.empty(B, L, L, device=Q.device)

        # --------- identical numerical path ----------
        for b in range(B):
            for i in range(L):
                # flatten all                 ↓ (L·F,)
                x1 = Q[b, i].repeat(L, 1).flatten().cpu().numpy()
                x2 = K[b].reshape(-1).cpu().numpy()
                dots = qm.evaluate_batch(x1, x2)          # (L·F,)
                S[b, i] = torch.from_numpy(
                    dots.reshape(L, F).sum(1)
                ).to(Q.device)
        return S

    @staticmethod
    def backward(ctx, grad_S):
        """
        grad_S : d Loss / d S      shape (B, L, L)
        Return  : gradients wrt (Q, K, qm) – last one is None
        """
        Q, K = ctx.saved_tensors          # (B, L, F)

        # classical dot-product surrogate:
        #   S = Q Kᵀ  ⇒  dS/dQ = K , dS/dK = Q
        grad_Q = torch.matmul(grad_S, K)                    # (B, L, F)
        grad_K = torch.matmul(grad_S.transpose(-1, -2), Q)  # (B, L, F)

        return grad_Q, grad_K, None     # qm is not a tensor ⇒ None

class QuantumAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        shots: int = 2_048,
        use_qcrank: bool = True,
        nq_addr=2, nq_data=2,
    ):
        super().__init__()
        self.nq_addr = nq_addr
        self.nq_data = nq_data
        self.dim_qc = (2 ** nq_addr) * nq_data # qcrank dimension
        self.embed_dim = embed_dim
        self.qm = QuantumMultiplier(shots) # type: ignore
        self.use_qcrank = use_qcrank and QCRANK_AVAILABLE
        self.use_qm = True            # whether to call the e-hands circuits
        self.shots = shots
        self.encoder = None
        if self.use_qcrank and self.encoder is None:
            self.encoder = QCrankEncoder(nq_addr=self.nq_addr, nq_data=self.nq_data, shots=self.shots)
        
        # projections for head 0 (2-d)
        self.q0 = nn.Linear(embed_dim, 2)
        self.k0 = nn.Linear(embed_dim, 2)
        self.v0 = nn.Linear(embed_dim, 2)
        self.out0 = nn.Linear(2, embed_dim)

        # head 1
        # classical layer
        self.q1_lin = nn.Linear(embed_dim, self.dim_qc)
        self.k1_lin = nn.Linear(embed_dim, self.dim_qc)
        self.v1_lin = nn.Linear(embed_dim, self.dim_qc)
        if self.use_qcrank:                         # add angle-MLPs
            self.q1_ang = AngleMLP(embed_dim, self.dim_qc)
            self.k1_ang = AngleMLP(embed_dim, self.dim_qc)
            self.v1_ang = AngleMLP(embed_dim, self.dim_qc)
            
        self.out1 = nn.Linear(self.dim_qc, embed_dim)
        self.fuse = nn.Linear(embed_dim * 2, embed_dim)

    # --------------------------------------------------
    def _scores_qm(self, Q: torch.Tensor, K: torch.Tensor):
        B, L, F = Q.shape
        S = torch.zeros(B, L, L, device=Q.device)
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    dot = 0.0
                    for d in range(F):
                        dot += self.qm.evaluate(float(Q[b, i, d]), float(K[b, j, d]))  # w ignored for tag=1
                        # classical way
                        # dot += float(Q[b, i, d])*float(K[b, j, d])
                    S[b, i, j] = dot
        return S
    
    def _scores_qm_batch(self, Q: torch.Tensor, K: torch.Tensor):
        B, L, F = Q.shape
        S = torch.empty(B, L, L, device=Q.device)
        for b in range(B):
            for i in range(L):
                # —— flatten (L·F) feature pairs in one go ——
                x1 = Q[b, i].repeat(L, 1).flatten().cpu().numpy()   # (L·F,)
                x2 = K[b]               .reshape(-1).cpu().numpy()  # (L·F,)
                dots = self.qm.evaluate_batch(x1, x2)               # (L·F,)
                S[b, i] = torch.from_numpy(dots.reshape(L, F).sum(1)).to(Q.device)
        return S
    

    # --------------------------------------------------
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # ----- head 0 (e-hands) -----
        Q0 = torch.tanh(self.q0(x))
        K0 = torch.tanh(self.k0(x))
        V0 = torch.tanh(self.v0(x))
        if self.use_qm:
            # enable vectorized batch
            S0 = QMScore.apply(Q0, K0, self.qm)
        else:
            S0 = torch.matmul(Q0, K0.transpose(-2, -1))
        L = x.size(1)
        full_mask = causal_mask(L, x.device)
        if mask is not None:
            full_mask &= mask
        S0 = S0.masked_fill(~full_mask, float("-inf"))
        A0 = torch.softmax(S0 / math.sqrt(2.0), dim=-1)
        O0 = self.out0(torch.matmul(A0, V0))

        # ----- head 1 (QCrank or classical identity) -----
        # non-linear mapping
        if self.use_qcrank:
            # learn angles → encode → detach measurement
            angles_Q = self.q1_ang(x)
            angles_K = self.k1_ang(x)
            angles_V = self.v1_ang(x)

            with torch.no_grad():
                Q1 = self.encoder.encode(angles_Q)
                K1 = self.encoder.encode(angles_K)
                V1 = self.encoder.encode(angles_V)
        else:
            # pure classical baseline (linear identity)
            Q1 = self.q1_lin(x)
            K1 = self.k1_lin(x)
            V1 = self.v1_lin(x)
        # else: classical identity, no quantum encoding
        if self.use_qm:
            S1 = QMScore.apply(Q1, K1, self.qm)
        else:
            S1 = torch.matmul(Q1, K1.transpose(-2, -1))
        S1 = S1.masked_fill(~full_mask, float("-inf"))
        A1 = torch.softmax(S1 / math.sqrt(Q1.size(-1)), dim=-1)
        O1 = self.out1(torch.matmul(A1, V1))

        # Always fuse both heads
        return self.fuse(torch.cat([O0, O1], dim=-1))

# ------------------------------------------------------------------
#   Transformer block & model
# ------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d: int = 32, ff: int = 128, shots: int = 2048):
        super().__init__()
        self.att = QuantumAttention(d, shots=shots)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.att(x, mask))
        x = self.ln2(x + self.ff(x))
        return x

class HybridQCrankTransformer(nn.Module):
    def __init__(self, vocab: int = 100, seq_len: int = 8, d: int = 32, shots: int = 2048):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.blk = nn.ModuleList([TransformerBlock(d, shots=shots) for _ in range(1)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        B, L = idx.shape
        dev = idx.device
        x = self.tok(idx) + self.pos(torch.arange(L, device=dev))
        for b in self.blk:
            x = b(x)
        return self.head(self.ln(x))

# ───────────────────── quantum sanity-check ─────────────────────
def _counter_wrap(counter, fn):
    """Return a wrapper that increments counter[0] then calls fn."""
    def wrapped(*a, **kw):
        counter[0] += 1
        return fn(*a, **kw)
    return wrapped

def logits_test():
# ------------------------------------------------------------------
    # 2.  forward pass
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    batch, seq_len = 1, 6
    dummy = torch.randint(0, 100, (batch, seq_len), device=dev)
    with torch.no_grad():
        logits = model(dummy)

    print("Logits tensor :", tuple(logits.shape))
    print("e-hands runs  :", e_calls[0])
    if QCRANK_AVAILABLE:
        print("QCrank runs   :", qcr_calls[0])

    # 1. Run quantum/hybrid transformer
    with torch.no_grad():
        logits_quantum = model(dummy)

    # 2. Disable quantum logic for classical baseline
    for blk in model.blk:
        blk.att.use_qcrank = False
        blk.att.use_qm = False

    with torch.no_grad():
        logits_classical = model(dummy)

    # 3. Compare outputs
    delta = (logits_quantum - logits_classical).abs().mean().item()
    print("Mean |Δlogits| quantum vs classical:", f"{delta:.4f}")
    predicted_token_quantum = logits_quantum[0, 3].argmax().item()
    predicted_token_classical = logits_classical[0, 3].argmax().item()

    print("Quantum prediction at pos 3:", predicted_token_quantum)
    print("Classical prediction at pos 3:", predicted_token_classical)
    print("Match:", predicted_token_quantum == predicted_token_classical)

# ──────────────────────────  toy I/O helpers  ─────────────────────────
# 100-token alphabet: ASCII codes 32-131  (space, !, …, ˜) 
CHARS = ''.join(chr(i) for i in range(32, 132))
assert len(CHARS) == 100                    # 0-based indices

stoi = {ch: i for i, ch in enumerate(CHARS)}
itos = {i: ch for ch, i in stoi.items()}

def encode(text: str) -> torch.Tensor:
    """string → (1, L) integer tensor in model’s vocab range 0-99"""
    idxs = [(ord(c) - 32) % 100 for c in text]      # cheap modulo map
    return torch.tensor([idxs], dtype=torch.long, device=model.tok.weight.device)

def decode(toks: List[int]) -> str:
    return ''.join(itos[i] for i in toks)

# ──────────────────────────  sampling core  ───────────────────────────
@torch.no_grad()
def generate(model,
             prompt: str,
             max_new_tokens: int = 40,
             temperature: float = 1.0,
             top_k: int | None = None) -> str:
    """
    Autoregressively sample *max_new_tokens* after *prompt*.
    Returns the full prompt+continuation string.
    """
    model.eval()
    idx = encode(prompt)                                # (1,L₀)

    for _ in range(max_new_tokens):
        # forward only last seq_len positions to respect model positional dim
        idx_cond = idx[:, -model.pos.num_embeddings:]   # (1,≤8) for default
        logits = model(idx_cond)                        # (1,L,V)
        logits = logits[0, -1] / temperature            # last step, (V,)

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float('inf')

        probs = logits.softmax(dim=-1)                  # (V,)
        next_tok = torch.multinomial(probs, num_samples=1)  # (1,)
        idx = torch.cat([idx, next_tok.unsqueeze(0)], dim=1)  # append

    continuation = decode(idx[0].tolist())
    return continuation

@torch.no_grad()
def evaluate_perplexity(model: nn.Module,
                        data_iter,
                        vocab: int,
                        max_batches: int | None = None) -> float:
    """
    Returns average perplexity over the batches drawn from *data_iter*.
    *data_iter* must yield (input, target) tensors like the training loop.
    """
    model.eval()
    total_nll = 0.0       # negative log-likelihood
    total_tok = 0

    for b, (x, y) in enumerate(data_iter):
        if max_batches is not None and b >= max_batches:
            break
        logits = model(x)                               # (B,L,V)
        nll  = F.cross_entropy(
                  logits.view(-1, vocab),
                  y.view(-1),
                  reduction="sum")                      # sum over tokens
        total_nll += nll.item()
        total_tok += x.numel()

    return math.exp(total_nll / total_tok)              # perplexity = e^{H}

# ────────────────────── metric tracker  ────────────────────────
class MetricTracker:
    """
    Collect (step, loss, ppl) triples so we can plot them later.
    """
    def __init__(self):
        self.steps: list[int]   = []
        self.losses: list[float] = []
        self.ppls:   list[float] = []

    def add(self, step: int, loss: float):
        self.steps.append(step)
        self.losses.append(loss)
        self.ppls.append(math.exp(loss))

    # helper to convert to numpy if desired
    def as_arrays(self):
        import numpy as np
        return np.asarray(self.steps), np.asarray(self.losses), np.asarray(self.ppls)

# ──────────────────── pretty plotting  ─────────────────────────
def plot_metrics(tracker: MetricTracker,
                 window: int = 5,
                 title: str = "Training curve"):

    import numpy as np, matplotlib.pyplot as plt, seaborn as sns
    sns.set_style("whitegrid")

    steps, losses, ppls = tracker.as_arrays()
    if len(steps) == 0:                    # nothing recorded
        print("plot_metrics: tracker is empty")
        return

    def smooth(arr, k):
        if k <= 1 or len(arr) < k:
            return arr
        return np.convolve(arr, np.ones(k)/k, mode="valid")

    # prepare smoothed data and their x-positions
    loss_sm = smooth(losses, window)
    ppl_sm  = smooth(ppls,   window)
    x_sm    = steps[len(steps) - len(loss_sm):]   # last len(loss_sm) steps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    # raw curves
    ax1.plot(steps, losses, color="#1f77b4", alpha=.3, label="raw")
    ax2.plot(steps, ppls,   color="#ff7f0e", alpha=.3, label="raw")

    # smoothed curves (only if they exist)
    if len(loss_sm):
        ax1.plot(x_sm, loss_sm, lw=2.2, color="#1f77b4",
                 label=f"smoothed (w={window})")
    if len(ppl_sm):
        ax2.plot(x_sm, ppl_sm,  lw=2.2, color="#ff7f0e",
                 label=f"smoothed (w={window})")

    # cosmetics
    ax1.set_ylabel("cross-entropy loss"); ax1.legend()
    ax2.set_ylabel("perplexity");         ax2.set_xlabel("training step")
    ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig("perplexity.png", dpi=300)

# ─────────────────────────  demo  main  ──────────────────────────
if __name__ == "__main__":
    import time
    EVAL = False
    # ── tqdm (optional) ───────────────────────────────────────────
    try:
        from tqdm.auto import tqdm
        TQDM = True
    except ImportError:                     # graceful fallback
        print("tqdm not found → no progress bar")
        TQDM = False
        def tqdm(x, *a, **k):               # type: ignore
            return x                        # dummy wrapper

    # reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # ── device & model ────────────────────────────────────────────
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shots = 1024
    print(f"Device: {dev},  QCrank: {QCRANK_AVAILABLE},  "
          f"EHands: {EHANDS_AVAILABLE},  shots: {shots}")

    model = HybridQCrankTransformer(shots=shots).to(dev)
    # for blk in model.blk:
    #     blk.att.use_qcrank = False
    #     blk.att.use_qm = False
    model.train()                       # enable grad / dropout etc.

    # ── quantum-call counters (optional) ──────────────────────────
    e_calls = [0]
    QuantumMultiplier.evaluate_batch = _counter_wrap(e_calls,
                                                 QuantumMultiplier.evaluate_batch)

    if QCRANK_AVAILABLE:
        qcr_calls = [0]
        QCrankEncoder.encode = _counter_wrap(qcr_calls,
                                             QCrankEncoder.encode)

    # ── synthetic data ------------------------------------------------
    vocab     = 100
    seq_len   = 6
    batch_sz  = 5
    n_steps   = 1                    # bump this if you like

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── training loop -------------------------------------------------
    bar = tqdm(range(1, n_steps + 1),
               desc="training",
               unit="step",
               disable=not TQDM)

    start = time.perf_counter()
    tracker   = MetricTracker()           #  ← NEW
    eval_every = 1
    for step in bar:
        # random token batch; target = next token (shift left)
        x = torch.randint(0, vocab, (batch_sz, seq_len), device=dev)
        y = x.roll(shifts=-1, dims=1)
        opt.zero_grad(set_to_none=True)
        logits = model(x)                         # (B, L, V)
        loss   = F.cross_entropy(
                     logits.view(-1, vocab),
                     y.view(-1)
                 )
        if step % eval_every == 0:          # or simply “always”
            tracker.add(step, loss.item())
        loss.backward()

        # example gradient: weight before quantum multiplier
        gnorm = model.blk[0].att.q0.weight.grad.norm().item()

        opt.step()

        if TQDM:
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            grad=f"{gnorm:.4f}")
        else:
            # simple print if tqdm unavailable
            print(f"step {step:4d}/{n_steps} | "
                  f"loss {loss.item():.4f} | "
                  f"‖∇ q0.weight‖ {gnorm:.4f}")

    elapsed = time.perf_counter() - start

    # ── summary -------------------------------------------------------
    print("\n──────── training finished ────────")
    print(f"elapsed time          : {elapsed:5.2f} s")
    print(f"e-hands circuit calls : {e_calls[0]}")
    if QCRANK_AVAILABLE:
        print(f"QCrank circuit calls : {qcr_calls[0]}")
    # if EVAL: 
    #     model.to(dev).eval()
    #     prompt = "quantum "
    #     out = generate(model,
    #                 prompt,
    #                 max_new_tokens=2,
    #                 temperature=0.9,
    #                 top_k=1)          # restrict to 20 best tokens
    #     print("\n▁prompt:", repr(prompt))
    #     print("▁output:", repr(out))

    # for step in bar:
    #     x = torch.randint(0, vocab, (batch_sz, seq_len), device=dev)
    #     y = x.roll(shifts=-1, dims=1)

    #     opt.zero_grad(set_to_none=True)
    #     logits = model(x)
    #     loss   = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
    #     loss.backward()
    #     opt.step()

    #     gnorm = model.blk[0].att.q0.weight.grad.norm().item()

    #     if TQDM:
    #         bar.set_postfix(loss=f"{loss.item():.4f}",
    #                         ppl=f"{math.exp(loss.item()):.2f}",   # ← NEW
    #                         grad=f"{gnorm:.4f}")
    #     else:
    #         print(f"step {step:4d}/{n_steps} | "
    #             f"loss {loss.item():.4f} | "
    #             f"ppl {math.exp(loss.item()):.2f} | "          # ← NEW
    #             f"‖∇ q0.weight‖ {gnorm:.4f}")
    #     # quick validation on 100 random batches
    # val_iter = ((torch.randint(0, vocab, (batch_sz, seq_len), device=dev),
    #             torch.randint(0, vocab, (batch_sz, seq_len), device=dev))
    #             for _ in range(5))

    # val_ppl = evaluate_perplexity(model, val_iter, vocab)
    # print(f"\nvalidation perplexity : {val_ppl:.2f}")

    # plot_metrics(tracker, window=7, title="Hybrid transformer – training")