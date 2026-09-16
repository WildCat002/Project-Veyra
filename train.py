import pickle
import time

import numpy as np

# ============================================================
# TRonX - small character-level RNN
# Pure NumPy, no PyTorch required
# ============================================================

# ---------- Load data ----------
text = open("data.txt", "r", encoding="utf-8", errors="ignore").read().lower()

chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
V = len(chars)

data = np.array([stoi[c] for c in text], dtype=np.int32)

print(f"Vocabulary: {V}")
print(f"Characters: {len(data)}")

# ---------- Hyperparameters ----------
SEQ_LEN = 128
EMB = 64
HID = 192

LR = 0.01
EPOCHS = 100
BATCH = 32

# Number of training positions per epoch.
# Using all 200k characters is possible, but this keeps training reasonable.
STEPS_PER_EPOCH = 1000


# ---------- Initialize ----------
def init_weights():
    rng = np.random.default_rng(42)

    return {
        "E": rng.normal(0, 0.05, (V, EMB)).astype(np.float32),
        "Wx": rng.normal(0, 0.08, (EMB, 4 * HID)).astype(np.float32),
        "Wh": rng.normal(0, 0.08, (HID, 4 * HID)).astype(np.float32),
        "b": np.zeros(4 * HID, dtype=np.float32),
        "Wy": rng.normal(0, 0.08, (HID, V)).astype(np.float32),
        "by": np.zeros(V, dtype=np.float32),
    }


# Start fresh if an old incompatible weights.pkl exists.
W = init_weights()


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-9)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------- One training batch ----------
def train_batch(starts):
    B = len(starts)

    # Inputs and targets
    X = np.empty((B, SEQ_LEN), dtype=np.int32)
    Y = np.empty((B, SEQ_LEN), dtype=np.int32)

    for b, start in enumerate(starts):
        X[b] = data[start : start + SEQ_LEN]
        Y[b] = data[start + 1 : start + SEQ_LEN + 1]

    # Forward caches
    hs = [np.zeros((B, HID), dtype=np.float32)]
    cache = []

    loss = 0.0

    for t in range(SEQ_LEN):
        x = W["E"][X[:, t]]

        gates = x @ W["Wx"] + hs[-1] @ W["Wh"] + W["b"]

        i = sigmoid(gates[:, 0:HID])
        f = sigmoid(gates[:, HID : 2 * HID])
        o = sigmoid(gates[:, 2 * HID : 3 * HID])
        g = np.tanh(gates[:, 3 * HID : 4 * HID])

        # Cell state
        if t == 0:
            c_prev = np.zeros((B, HID), dtype=np.float32)
        else:
            c_prev = cache[-1][1]

        c = f * c_prev + i * g
        h = o * np.tanh(c)

        logits = h @ W["Wy"] + W["by"]
        probs = softmax(logits)

        loss -= np.log(probs[np.arange(B), Y[:, t]] + 1e-9).mean()

        hs.append(h)
        cache.append((i, f, o, g, c, c_prev, x, gates, probs))

    loss /= SEQ_LEN

    # ---------- Backward ----------
    dE = np.zeros_like(W["E"])
    dWx = np.zeros_like(W["Wx"])
    dWh = np.zeros_like(W["Wh"])
    db = np.zeros_like(W["b"])
    dWy = np.zeros_like(W["Wy"])
    dby = np.zeros_like(W["by"])

    dh_next = np.zeros((B, HID), dtype=np.float32)
    dc_next = np.zeros((B, HID), dtype=np.float32)

    for t in reversed(range(SEQ_LEN)):
        i, f, o, g, c, c_prev, x, gates, probs = cache[t]
        h = hs[t + 1]

        dlogits = probs.copy()
        dlogits[np.arange(B), Y[:, t]] -= 1
        dlogits /= B * SEQ_LEN

        dWy += h.T @ dlogits
        dby += dlogits.sum(axis=0)

        dh = dlogits @ W["Wy"].T + dh_next

        tanh_c = np.tanh(c)

        do = dh * tanh_c
        dc = dh * o * (1 - tanh_c * tanh_c) + dc_next

        di = dc * g
        dg = dc * i
        df = dc * c_prev

        dc_next = dc * f

        di_raw = di * i * (1 - i)
        df_raw = df * f * (1 - f)
        do_raw = do * o * (1 - o)
        dg_raw = dg * (1 - g * g)

        dgate = np.concatenate([di_raw, df_raw, do_raw, dg_raw], axis=1)

        dWx += x.T @ dgate
        dWh += hs[t].T @ dgate
        db += dgate.sum(axis=0)

        dx = dgate @ W["Wx"].T
        dh_next = dgate @ W["Wh"].T

        np.add.at(dE, X[:, t], dx)

    # Gradient clipping
    for grad in [dE, dWx, dWh, db, dWy, dby]:
        np.clip(grad, -5.0, 5.0, out=grad)

    # ---------- Update ----------
    W["E"] -= LR * dE
    W["Wx"] -= LR * dWx
    W["Wh"] -= LR * dWh
    W["b"] -= LR * db
    W["Wy"] -= LR * dWy
    W["by"] -= LR * dby

    return loss


# ---------- Training ----------
rng = np.random.default_rng(123)

max_start = len(data) - SEQ_LEN - 1

print("\nTraining TRonX...")
print("Old weights are intentionally NOT loaded.")
print()

for epoch in range(EPOCHS):
    start_time = time.time()
    total_loss = 0.0

    for step in range(STEPS_PER_EPOCH):
        starts = rng.integers(0, max_start, size=BATCH)

        loss = train_batch(starts)
        total_loss += loss

    avg = total_loss / STEPS_PER_EPOCH

    print(
        f"epoch {epoch + 1:02d}/{EPOCHS} "
        f"loss {avg:.4f} "
        f"time {time.time() - start_time:.1f}s"
    )

    # Save after every epoch
    with open("weights.pkl", "wb") as f:
        pickle.dump(
            {
                "W": W,
                "stoi": stoi,
                "itos": itos,
                "V": V,
                "SEQ_LEN": SEQ_LEN,
                "EMB": EMB,
                "HID": HID,
            },
            f,
        )

print("\nTraining done.")
print("New weights saved to weights.pkl")
