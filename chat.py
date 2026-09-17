import pickle

import numpy as np

# ============================================================
# Veyra v0.2.0
# Character-level text generation
# ============================================================


WEIGHTS_FILE = "weights.pkl"


# ============================================================
# Load model
# ============================================================

with open(
    WEIGHTS_FILE,
    "rb",
) as f:

    saved = pickle.load(f)


W = saved["W"]

stoi = saved["stoi"]
itos = saved["itos"]

V = saved["V"]

SEQ_LEN = saved["SEQ_LEN"]
HID = saved["HID"]

VERSION = saved.get(
    "version",
    "unknown",
)


# ============================================================
# Random generator
# ============================================================

rng = np.random.default_rng()


# ============================================================
# Math
# ============================================================


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    x = x - np.max(x)

    e = np.exp(x)

    return e / (np.sum(e) + 1e-9)


# ============================================================
# One character step
# ============================================================


def next_char(
    char_id,
    h,
    c,
):

    x = W["E"][char_id]

    gates = x @ W["Wx"] + h @ W["Wh"] + W["b"]

    i = sigmoid(gates[0:HID])

    f = sigmoid(gates[HID : 2 * HID])

    o = sigmoid(gates[2 * HID : 3 * HID])

    g = np.tanh(gates[3 * HID : 4 * HID])

    c = f * c + i * g

    h = o * np.tanh(c)

    logits = h @ W["Wy"] + W["by"]

    return logits, h, c


# ============================================================
# Sampling
# ============================================================


def sample_next(
    logits,
    temperature,
):

    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    logits = logits / temperature

    p = softmax(logits)

    return rng.choice(
        V,
        p=p,
    )


# ============================================================
# Generate text
# ============================================================


def generate(
    prompt,
    length=500,
    temperature=0.7,
):

    prompt = prompt.lower()

    # Keep only known characters
    prompt = "".join(c for c in prompt if c in stoi)

    if not prompt:
        prompt = "alice"

    # Initial hidden/cell states
    h = np.zeros(
        HID,
        dtype=np.float32,
    )

    c = np.zeros(
        HID,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Warm up using prompt
    # --------------------------------------------------------

    for ch in prompt:

        _, h, c = next_char(
            stoi[ch],
            h,
            c,
        )

    out = list(prompt)

    last_id = stoi[prompt[-1]]

    # --------------------------------------------------------
    # Generate continuation
    # --------------------------------------------------------

    for _ in range(length):

        logits, h, c = next_char(
            last_id,
            h,
            c,
        )

        nxt = sample_next(
            logits,
            temperature,
        )

        ch = itos[nxt]

        out.append(ch)

        last_id = nxt

    return "".join(out)


# ============================================================
# CLI
# ============================================================

print("=" * 50)
print(f"Veyra {VERSION}")
print("=" * 50)

print("Character-level language model")

print("Type 'quit' to exit.")

print()


while True:

    try:

        s = input("> ").strip()

    except (
        KeyboardInterrupt,
        EOFError,
    ):

        print("\nBye!")

        break

    if s.lower() == "quit":

        print("Bye!")

        break

    if not s:
        continue

    print()

    print(
        generate(
            s,
            length=500,
            temperature=0.7,
        )
    )

    print()
