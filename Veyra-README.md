# Veyra

A small character-level text generation model built with Python and NumPy.

## Requirements

- Windows 10/11
- Python 3
- NumPy
- Internet connection for downloading the training data

---

## 1. Open the Veyra folder

Open PowerShell or Windows Terminal and navigate to the project:

```powershell
cd path\to\Veyra
```

For example:

```powershell
cd E:\Projects\Veyra
```

---

## 2. Create a virtual environment

Recommended:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install NumPy

```powershell
python -m pip install --upgrade pip
python -m pip install numpy
```

---

## 4. Prepare the training data

Veyra is currently trained on multiple plain-text public-domain books from Project Gutenberg.

Create a `data` folder:

```powershell
mkdir data
```

Download the training books with PowerShell:

```powershell
Invoke-WebRequest "https://www.gutenberg.org/files/11/11-0.txt" -OutFile "data/alice.txt"
Invoke-WebRequest "https://www.gutenberg.org/files/55/55-0.txt" -OutFile "data/oz.txt"
Invoke-WebRequest "https://www.gutenberg.org/files/1661/1661-0.txt" -OutFile "data/sherlock.txt"
Invoke-WebRequest "https://www.gutenberg.org/files/35/35-0.txt" -OutFile "data/time_machine.txt"
```

Then combine the books:

```powershell
python combine_data.py
```

This creates `data.txt` from all `.txt` files in the `data` folder.

The current dataset contains approximately **1.11 million characters**.

### Dataset backup

If you already have an older `data.txt`, keep a backup before replacing it:

```powershell
Rename-Item data.txt data_old.txt
```

`data_old.txt` is only a backup and is not used by training.

---

## 5. Training data size

The training script uses the full contents of `data.txt`.

The current dataset contains approximately **1,114,179 characters**. Do not manually trim `data.txt` to 200,000 characters.

## 6. Train the model

Run:

```powershell
python train.py
```

The current training configuration uses:

- Character-level language modeling
- NumPy
- Sequence length: 64
- Embedding size: 64
- Hidden size: 192
- Batch size: 32
- 100 training epochs

The training process will:

- Load `data.txt`
- Build the character vocabulary
- Train the recurrent neural network
- Save `weights.pkl`

The training script intentionally starts from scratch rather than loading previous model weights.

You should eventually see:

```text
Training done.
New weights saved to weights.pkl
```

### Backing up an older model

Before starting a new training run, you can preserve the previous weights:

```powershell
Copy-Item weights.pkl weights_old.pkl
```

## 7. Run the text generator

After training has completed:

```powershell
python chat.py
```

You should see:

```text
Type a prompt (or 'quit'):
>
```

Enter a prompt, for example:

```text
Alice was
```

To exit:

```text
quit
```

---

## Project files

```text
Veyra/
├── README.md
├── train.py
├── chat.py
├── combine_data.py
├── requirements.txt
├── data/
│   ├── alice.txt
│   ├── oz.txt
│   ├── sherlock.txt
│   └── time_machine.txt
├── data.txt
├── data_old.txt
├── weights.pkl
└── weights_old.pkl
```

### Main files

- `train.py` — trains the Veyra neural language model.
- `chat.py` — loads `weights.pkl` and generates text.
- `combine_data.py` — combines the `.txt` files in `data/` into `data.txt`.
- `requirements.txt` — Python dependencies.
- `data.txt` — current combined training dataset.
- `data_old.txt` — optional backup of the previous dataset.
- `weights.pkl` — current trained model weights.
- `weights_old.pkl` — optional backup of previous weights.

## Quick Start

If Python is already installed:

```powershell
cd path\to\Veyra

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir data

Invoke-WebRequest "https://www.gutenberg.org/files/11/11-0.txt" -OutFile "data/alice.txt"
Invoke-WebRequest "https://www.gutenberg.org/files/55/55-0.txt" -OutFile "data/oz.txt"
Invoke-WebRequest "https://www.gutenberg.org/files/1661/1661-0.txt" -OutFile "data/sherlock.txt"
Invoke-WebRequest "https://www.gutenberg.org/files/35/35-0.txt" -OutFile "data/time_machine.txt"

python combine_data.py
python train.py
python chat.py
```

## Troubleshooting

### `python is not recognized`

Install Python and make sure **Add Python to PATH** is enabled during installation.

You can check the installation with:

```powershell
python --version
```

### `No module named 'numpy'`

Make sure the virtual environment is activated, then run:

```powershell
python -m pip install numpy
```

### `FileNotFoundError: data.txt`

Make sure you are running the commands from the Veyra folder:

```powershell
dir
```

You should see `data.txt`.

### `FileNotFoundError: weights.pkl`

Run the training script first:

```powershell
python train.py
```

### PowerShell says scripts are disabled

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```