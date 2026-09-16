import pickle

import numpy as np

# ============================================================
# Veyra - chat / text generation
# Compatible with the new RNN train.py
# ============================================================

with open("weights.pkl", "rb") as f:
    saved = pickle.load(f)

W = saved["W"]
stoi = saved["stoi"]
itos = saved["itos"]
V = saved["V"]

SEQ_LEN = saved["SEQ_LEN"]
HID = saved["HID"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def next_char(char_id, h, c, temp=0.7):
    x = W["E"][char_id]

    gates = x @ W["Wx"] + h @ W["Wh"] + W["b"]

    i = sigmoid(gates[0:HID])
    f = sigmoid(gates[HID : 2 * HID])
    o = sigmoid(gates[2 * HID : 3 * HID])
    g = np.tanh(gates[3 * HID : 4 * HID])

    c = f * c + i * g
    h = o * np.tanh(c)

    logits = (h @ W["Wy"] + W["by"]) / temp
    p = softmax(logits)

    return p, h, c


def generate(prompt, length=500, temp=0.7):
    prompt = prompt.lower()

    # Keep only characters our tokenizer knows.
    prompt = "".join(c for c in prompt if c in stoi)

    if not prompt:
        prompt = "alice"

    # Warm up the RNN using the entire prompt.
    h = np.zeros(HID, dtype=np.float32)
    c = np.zeros(HID, dtype=np.float32)

    for ch in prompt:
        _, h, c = next_char(stoi[ch], h, c, temp)

    out = list(prompt)

    # Generate continuation.
    last_id = stoi[prompt[-1]]

    for _ in range(length):
        p, h, c = next_char(last_id, h, c, temp)

        nxt = np.random.choice(V, p=p)

        ch = itos[nxt]
        out.append(ch)

        last_id = nxt

    return "".join(out)


print("Veyra")
print("Type a prompt (or 'quit'):")
print()

while True:
    try:
        s = input("> ").strip()

    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        break

    if s.lower() == "quit":
        print("Bye!")
        break

    if not s:
        continue

    print(generate(s, length=500, temp=0.7))

    print()
