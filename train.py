import pickle
import time

import numpy as np

# ============================================================
# Veyra v0.2.0
# Character-level LSTM language model
# Pure NumPy
#
# Upgrades from v0.1.0:
# - Adam optimizer
# - Global gradient clipping
# - Forget-gate bias initialization
# - Checkpoint/resume support
# ============================================================


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "data.txt"
WEIGHTS_FILE = "weights.pkl"

# Model
SEQ_LEN = 128
EMB = 64
HID = 192

# Training
LR = 0.001
EPOCHS = 100
BATCH = 32
STEPS_PER_EPOCH = 1000

GRAD_CLIP = 5.0

# Random seeds
MODEL_SEED = 42
TRAIN_SEED = 123

# False = train a fresh model
# True  = continue from weights.pkl
RESUME = False


# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")

with open(
    DATA_FILE,
    "r",
    encoding="utf-8",
    errors="ignore",
) as f:
    text = f.read().lower()


chars = sorted(set(text))

stoi = {c: i for i, c in enumerate(chars)}

itos = {i: c for i, c in enumerate(chars)}

V = len(chars)

data = np.array(
    [stoi[c] for c in text],
    dtype=np.int32,
)


print(f"Vocabulary: {V}")
print(f"Characters: {len(data):,}")


if len(data) <= SEQ_LEN + 1:
    raise ValueError("Dataset is too small for the selected sequence length.")


# ============================================================
# Initialize model
# ============================================================


def init_weights():
    rng = np.random.default_rng(MODEL_SEED)

    W = {
        "E": rng.normal(
            0,
            0.05,
            (V, EMB),
        ).astype(np.float32),
        "Wx": rng.normal(
            0,
            0.08,
            (EMB, 4 * HID),
        ).astype(np.float32),
        "Wh": rng.normal(
            0,
            0.08,
            (HID, 4 * HID),
        ).astype(np.float32),
        "b": np.zeros(
            4 * HID,
            dtype=np.float32,
        ),
        "Wy": rng.normal(
            0,
            0.08,
            (HID, V),
        ).astype(np.float32),
        "by": np.zeros(
            V,
            dtype=np.float32,
        ),
    }

    # Gate order:
    # input | forget | output | candidate
    #
    # Positive forget bias helps the LSTM retain information.
    W["b"][HID : 2 * HID] = 1.0

    return W


# ============================================================
# Adam state
# ============================================================


def init_adam_state(W):
    return {
        "m": {name: np.zeros_like(value) for name, value in W.items()},
        "v": {name: np.zeros_like(value) for name, value in W.items()},
        "step": 0,
    }


# ============================================================
# Math
# ============================================================


def softmax(x):
    x = x - np.max(
        x,
        axis=1,
        keepdims=True,
    )

    e = np.exp(x)

    return e / (
        np.sum(
            e,
            axis=1,
            keepdims=True,
        )
        + 1e-9
    )


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ============================================================
# One training batch
# ============================================================


def train_batch(starts):
    B = len(starts)

    # --------------------------------------------------------
    # Inputs / targets
    # --------------------------------------------------------

    X = np.empty(
        (B, SEQ_LEN),
        dtype=np.int32,
    )

    Y = np.empty(
        (B, SEQ_LEN),
        dtype=np.int32,
    )

    for b, start in enumerate(starts):
        X[b] = data[start : start + SEQ_LEN]

        Y[b] = data[start + 1 : start + SEQ_LEN + 1]

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    hs = [
        np.zeros(
            (B, HID),
            dtype=np.float32,
        )
    ]

    cache = []

    loss = 0.0

    for t in range(SEQ_LEN):

        x = W["E"][X[:, t]]

        gates = x @ W["Wx"] + hs[-1] @ W["Wh"] + W["b"]

        i = sigmoid(gates[:, 0:HID])

        f = sigmoid(gates[:, HID : 2 * HID])

        o = sigmoid(gates[:, 2 * HID : 3 * HID])

        g = np.tanh(gates[:, 3 * HID : 4 * HID])

        if t == 0:
            c_prev = np.zeros(
                (B, HID),
                dtype=np.float32,
            )
        else:
            c_prev = cache[-1][4]

        c = f * c_prev + i * g

        h = o * np.tanh(c)

        logits = h @ W["Wy"] + W["by"]

        probs = softmax(logits)

        loss -= np.log(probs[np.arange(B), Y[:, t]] + 1e-9).mean()

        hs.append(h)

        cache.append(
            (
                i,
                f,
                o,
                g,
                c,
                c_prev,
                x,
                probs,
            )
        )

    loss /= SEQ_LEN

    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    dW = {
        "E": np.zeros_like(W["E"]),
        "Wx": np.zeros_like(W["Wx"]),
        "Wh": np.zeros_like(W["Wh"]),
        "b": np.zeros_like(W["b"]),
        "Wy": np.zeros_like(W["Wy"]),
        "by": np.zeros_like(W["by"]),
    }

    dh_next = np.zeros(
        (B, HID),
        dtype=np.float32,
    )

    dc_next = np.zeros(
        (B, HID),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Backpropagation through time
    # --------------------------------------------------------

    for t in reversed(range(SEQ_LEN)):

        (
            i,
            f,
            o,
            g,
            c,
            c_prev,
            x,
            probs,
        ) = cache[t]

        h = hs[t + 1]

        # Output gradient
        dlogits = probs.copy()

        dlogits[np.arange(B), Y[:, t]] -= 1.0

        dlogits /= B * SEQ_LEN

        dW["Wy"] += h.T @ dlogits

        dW["by"] += dlogits.sum(axis=0)

        # Hidden gradient
        dh = dlogits @ W["Wy"].T + dh_next

        tanh_c = np.tanh(c)

        # Cell gradient
        do = dh * tanh_c

        dc = dh * o * (1.0 - tanh_c * tanh_c) + dc_next

        # Gate gradients
        di = dc * g
        dg = dc * i
        df = dc * c_prev

        dc_next = dc * f

        # Activation derivatives
        di_raw = di * i * (1.0 - i)

        df_raw = df * f * (1.0 - f)

        do_raw = do * o * (1.0 - o)

        dg_raw = dg * (1.0 - g * g)

        # Combine gates
        dgate = np.concatenate(
            [
                di_raw,
                df_raw,
                do_raw,
                dg_raw,
            ],
            axis=1,
        )

        # LSTM parameters
        dW["Wx"] += x.T @ dgate

        dW["Wh"] += hs[t].T @ dgate

        dW["b"] += dgate.sum(axis=0)

        # Previous hidden state
        dh_next = dgate @ W["Wh"].T

        # Embedding gradient
        dx = dgate @ W["Wx"].T

        np.add.at(
            dW["E"],
            X[:, t],
            dx,
        )

    # --------------------------------------------------------
    # Global gradient norm
    # --------------------------------------------------------

    total_norm_sq = 0.0

    for grad in dW.values():
        total_norm_sq += float(np.sum(grad.astype(np.float64) ** 2))

    total_norm = np.sqrt(total_norm_sq)

    # --------------------------------------------------------
    # Global gradient clipping
    # --------------------------------------------------------

    if total_norm > GRAD_CLIP:

        scale = GRAD_CLIP / (total_norm + 1e-8)

        for grad in dW.values():
            grad *= scale

    return loss, total_norm, dW


# ============================================================
# Adam update
# ============================================================


def adam_update(dW):
    adam["step"] += 1

    step = adam["step"]

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    for name in W:

        grad = dW[name]

        m = adam["m"][name]
        v = adam["v"][name]

        # First moment
        m *= beta1
        m += (1.0 - beta1) * grad

        # Second moment
        v *= beta2
        v += (1.0 - beta2) * grad * grad

        # Bias correction
        m_hat = m / (1.0 - beta1**step)

        v_hat = v / (1.0 - beta2**step)

        # Update
        W[name] -= LR * m_hat / (np.sqrt(v_hat) + epsilon)


# ============================================================
# Save checkpoint
# ============================================================


def save_checkpoint(
    epoch,
    best_loss,
):

    checkpoint = {
        "W": W,
        "stoi": stoi,
        "itos": itos,
        "V": V,
        "SEQ_LEN": SEQ_LEN,
        "EMB": EMB,
        "HID": HID,
        "optimizer": "adam",
        "adam": adam,
        "epoch": epoch,
        "best_loss": best_loss,
        "learning_rate": LR,
        "version": "0.2.0",
    }

    with open(
        WEIGHTS_FILE,
        "wb",
    ) as f:

        pickle.dump(
            checkpoint,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# ============================================================
# Initialize / resume
# ============================================================

adam = None
start_epoch = 0
best_loss = float("inf")


if RESUME:

    try:

        print()
        print("Loading existing checkpoint...")

        with open(
            WEIGHTS_FILE,
            "rb",
        ) as f:

            saved = pickle.load(f)

        # Check architecture compatibility
        if (
            saved["V"] != V
            or saved["SEQ_LEN"] != SEQ_LEN
            or saved["EMB"] != EMB
            or saved["HID"] != HID
        ):
            raise ValueError(
                "Checkpoint architecture does not "
                "match the current model configuration."
            )

        W = saved["W"]

        adam = saved.get(
            "adam",
            init_adam_state(W),
        )

        start_epoch = saved.get(
            "epoch",
            0,
        )

        best_loss = saved.get(
            "best_loss",
            float("inf"),
        )

        print(f"Resuming from epoch {start_epoch}")

        print(f"Previous best loss: {best_loss:.4f}")

    except FileNotFoundError:

        print("No weights.pkl found.")

        print("Starting a new model.")

        W = init_weights()
        adam = init_adam_state(W)

else:

    print()
    print("Starting a completely new model.")

    W = init_weights()
    adam = init_adam_state(W)


# ============================================================
# Training setup
# ============================================================

rng = np.random.default_rng(TRAIN_SEED)

max_start = len(data) - SEQ_LEN - 1


print()
print("=" * 60)
print("Veyra v0.2.0")
print("=" * 60)

print(f"Dataset:       {len(data):,} characters")

print(f"Vocabulary:    {V}")

print(f"Sequence:      {SEQ_LEN}")

print(f"Embedding:     {EMB}")

print(f"Hidden:        {HID}")

print(f"Batch:         {BATCH}")

print(f"Steps/epoch:   {STEPS_PER_EPOCH}")

print(f"Learning rate: {LR}")

print("Optimizer:     Adam")

print(f"Epochs:        {EPOCHS}")

print("=" * 60)
print()


# ============================================================
# Training loop
# ============================================================

for epoch in range(
    start_epoch,
    EPOCHS,
):

    epoch_start = time.time()

    total_loss = 0.0
    total_grad_norm = 0.0

    for step in range(
        STEPS_PER_EPOCH,
    ):

        starts = rng.integers(
            0,
            max_start,
            size=BATCH,
        )

        loss, grad_norm, dW = train_batch(starts)

        # Adam
        adam_update(dW)

        total_loss += loss
        total_grad_norm += grad_norm

    avg_loss = total_loss / STEPS_PER_EPOCH

    avg_grad_norm = total_grad_norm / STEPS_PER_EPOCH

    elapsed = time.time() - epoch_start

    best_loss = min(best_loss, avg_loss)

    print(
        f"epoch {epoch + 1:03d}/{EPOCHS} "
        f"loss {avg_loss:.4f} "
        f"grad {avg_grad_norm:.3f} "
        f"time {elapsed:.1f}s"
    )

    # Save after every epoch
    save_checkpoint(
        epoch + 1,
        best_loss,
    )


print()
print("=" * 60)
print("Training done.")
print(f"Final loss: {avg_loss:.4f}")
print(f"Best loss:  {best_loss:.4f}")
print("Weights saved to weights.pkl")
print("=" * 60)
