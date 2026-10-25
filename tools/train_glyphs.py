#!/usr/bin/env python3
"""
Train a ternary 16-8-4 glyph classifier for tt_um_pcm_ternary_mlp.

The tile is not a general MLP. Its forward pass is exactly:

    acc1[j] = sum_i x[i] * w1[i][j]          x in {0,1},  w in {-1,0,+1}
    hid[j]  = 1 if acc1[j] > 0 else 0        hard step, no bias
    acc2[m] = sum_k hid[k] * w2[k][m]
    class   = argmax(acc2), ties to lowest index

No biases, no scaling, binary inputs, binary hidden units. Training therefore
happens in three stages:

  1. float surrogate   - sigmoid hidden unit stands in for the hard step so
                         gradients exist at all.
  2. ternarization     - TWN magnitude threshold, delta = 0.7 * mean|W|.
  3. integer repair    - coordinate descent against the EXACT integer model.
                         This is the stage that matters: stages 1-2 give a
                         good starting point, but only stage 3 optimizes the
                         thing the silicon actually computes, so there is no
                         train/deploy mismatch left to discover later.

Writes tools/trained_weights.py for golden.py to import.

    python3 tools/train_glyphs.py
"""

import json
import random
import sys

import numpy as np

N_IN, N_HID, N_OUT = 16, 8, 4
SEED = 20261122          # the submit-by date
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ----------------------------------------------------------------- glyphs --
# 4x4 binary grids, index i = row*4 + col. Three of the four carry exactly
# 8 lit pixels, so pixel count alone cannot separate them - the network has
# to learn position, not brightness.

GLYPHS = {
    0: ("BAR", ["0000",
                "1111",
                "1111",
                "0000"]),
    1: ("COL", ["0110",
                "0110",
                "0110",
                "0110"]),
    2: ("X",   ["1001",
                "0110",
                "0110",
                "1001"]),
    3: ("BOX", ["1111",
                "1001",
                "1001",
                "1111"]),
}


def glyph_vector(rows):
    return np.array([int(c) for r in rows for c in r], dtype=np.int8)


PROTOTYPES = np.stack([glyph_vector(GLYPHS[c][1]) for c in range(N_OUT)])


def make_set(n_per_class, max_flips):
    """Noisy variants: flip 0..max_flips pixels uniformly at random."""
    X, y = [], []
    for c in range(N_OUT):
        for _ in range(n_per_class):
            v = PROTOTYPES[c].copy()
            k = rng.integers(0, max_flips + 1)
            if k:
                idx = rng.choice(N_IN, size=int(k), replace=False)
                v[idx] ^= 1
            X.append(v)
            y.append(c)
    return np.array(X, dtype=np.int8), np.array(y, dtype=np.int64)


# ------------------------------------------------- stage 1: float surrogate --

def train_float(X, y, epochs=4000, lr=0.08, beta=1.0):
    Xf = X.astype(np.float64)
    Y = np.eye(N_OUT)[y]
    W1 = rng.normal(0, 0.5, (N_IN, N_HID))
    W2 = rng.normal(0, 0.5, (N_HID, N_OUT))

    for ep in range(epochs):
        a1 = Xf @ W1
        h = 1.0 / (1.0 + np.exp(-beta * a1))      # stands in for step(a1 > 0)
        a2 = h @ W2
        a2 -= a2.max(axis=1, keepdims=True)
        p = np.exp(a2)
        p /= p.sum(axis=1, keepdims=True)

        d2 = (p - Y) / len(Xf)
        gW2 = h.T @ d2
        dh = d2 @ W2.T
        d1 = dh * beta * h * (1.0 - h)
        gW1 = Xf.T @ d1

        W1 -= lr * gW1
        W2 -= lr * gW2

    return W1, W2


# -------------------------------------------------- stage 2: ternarization --

def ternarize(W, scale=0.7):
    """TWN threshold: delta = scale * mean(|W|), everything below goes to 0."""
    delta = scale * np.abs(W).mean()
    T = np.zeros_like(W, dtype=np.int8)
    T[W > delta] = 1
    T[W < -delta] = -1
    return T


# --------------------------------------- the exact integer model, vectorized --

def predict(T1, T2, X):
    """Vectorized twin of golden.infer. Verified against it in main()."""
    acc1 = X.astype(np.int32) @ T1.astype(np.int32)
    hid = (acc1 > 0).astype(np.int32)              # strict >0, as in the RTL
    acc2 = hid @ T2.astype(np.int32)
    return acc2.argmax(axis=1)                     # ties -> lowest index


def accuracy(T1, T2, X, y):
    return float((predict(T1, T2, X) == y).mean())


# ------------------------------------------------ stage 3: integer repair --

def repair(T1, T2, X, y, passes=6, zero_bias=1e-4):
    """
    Coordinate descent on the integer model. For each weight, try all three
    ternary values and keep the best training accuracy. Ties prefer 0, which
    buys sparsity for free - a zero weight is a cell that never toggles.
    """
    best = accuracy(T1, T2, X, y)
    for p in range(passes):
        improved = False
        coords = ([("1", i, j) for i in range(N_IN) for j in range(N_HID)] +
                  [("2", k, m) for k in range(N_HID) for m in range(N_OUT)])
        random.shuffle(coords)
        for layer, a, b in coords:
            T = T1 if layer == "1" else T2
            cur = T[a, b]
            best_v, best_s = cur, best + (zero_bias if cur == 0 else 0.0)
            for v in (-1, 0, 1):
                if v == cur:
                    continue
                T[a, b] = v
                s = accuracy(T1, T2, X, y) + (zero_bias if v == 0 else 0.0)
                if s > best_s:
                    best_v, best_s = v, s
            T[a, b] = best_v
            if best_v != cur:
                improved = True
                best = accuracy(T1, T2, X, y)
        print(f"  repair pass {p + 1}: train acc {best:.4f}")
        if not improved:
            break
    return T1, T2


# ----------------------------------------------------------------- output --

def as_literal(T):
    return "[\n" + "".join(
        "    [" + ", ".join(f"{v:2d}" for v in row) + "],\n" for row in T
    ) + "]"


def main():
    sys.path.insert(0, "test")
    import golden as G

    Xtr, ytr = make_set(400, max_flips=2)
    Xte, yte = make_set(250, max_flips=3)
    print(f"train {len(Xtr)} samples, test {len(Xte)} samples "
          f"(up to 2 and 3 flipped pixels)")

    print("\nstage 1 - float surrogate")
    W1, W2 = train_float(Xtr, ytr)

    print("stage 2 - ternarize")
    T1, T2 = ternarize(W1), ternarize(W2)
    a_tr, a_te = accuracy(T1, T2, Xtr, ytr), accuracy(T1, T2, Xte, yte)
    print(f"  after quantization: train {a_tr:.4f}  test {a_te:.4f}")

    print("stage 3 - integer repair")
    T1, T2 = repair(T1, T2, Xtr, ytr)

    # The fast path must agree with the reference, element for element.
    w1l = [[int(v) for v in row] for row in T1]
    w2l = [[int(v) for v in row] for row in T2]
    for v in list(Xte[:200]) + list(PROTOTYPES):
        x = [int(b) for b in v]
        ref, _, _ = G.infer(w1l, w2l, x)
        fast = int(predict(T1, T2, v[None, :])[0])
        assert ref == fast, f"golden.infer={ref} vs predict={fast} on {x}"
    print("  vectorized model agrees with golden.infer on 204 cases")

    a_tr, a_te = accuracy(T1, T2, Xtr, ytr), accuracy(T1, T2, Xte, yte)
    clean = accuracy(T1, T2, PROTOTYPES, np.arange(N_OUT))
    sparsity = float((np.concatenate([T1.ravel(), T2.ravel()]) == 0).mean())

    print(f"\nFINAL  train {a_tr:.4f}   test {a_te:.4f}   "
          f"clean prototypes {clean:.4f}   sparsity {sparsity:.3f}")

    # Per-class accuracy, and robustness as noise grows.
    pred = predict(T1, T2, Xte)
    for c in range(N_OUT):
        sel = yte == c
        print(f"  class {c} {GLYPHS[c][0]:4s}: {float((pred[sel] == c).mean()):.4f}")
    print("  noise sweep (test accuracy vs flipped pixels):")
    for k in range(0, 5):
        Xk, yk = make_set(250, max_flips=k)
        print(f"    up to {k} flips: {accuracy(T1, T2, Xk, yk):.4f}")

    with open("tools/trained_weights.py", "w") as f:
        f.write('"""Generated by tools/train_glyphs.py - do not edit by hand."""\n\n')
        f.write(f"SEED = {SEED}\n")
        f.write(f"GLYPHS = {json.dumps({k: v for k, v in GLYPHS.items()}, indent=4)}\n\n")
        f.write(f"TRAIN_ACC = {a_tr:.4f}\nTEST_ACC = {a_te:.4f}\n")
        f.write(f"SPARSITY = {sparsity:.4f}\n\n")
        f.write("W1 = " + as_literal(T1) + "\n")
        f.write("W2 = " + as_literal(T2) + "\n")
    print("\nwrote tools/trained_weights.py")


if __name__ == "__main__":
    main()
