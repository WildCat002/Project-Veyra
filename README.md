# Veyra

A small character-level recurrent language model built from scratch with **Python + NumPy**.

Veyra is a learning-focused project designed to explore how a language model can learn character patterns, words, punctuation, and sentence structure without relying on a deep-learning framework.

---

## Current Version

**Veyra v0.2.0**

v0.2.0 is the first major training-system upgrade after the initial baseline.

### v0.2.0 highlights

- Adam optimizer
- Global gradient clipping
- Forget-gate bias initialization
- Checkpoint saving after every epoch
- Checkpoint/resume support
- Longer training sequences
- Full multi-book training dataset
- Model metadata stored with `weights.pkl`
- Improved generation quality compared with v0.1.0

---

## Model

Veyra uses a small character-level recurrent architecture implemented entirely with NumPy.

### Current configuration

| Setting | Value |
|---|---:|
| Dataset | 1,114,179 characters |
| Vocabulary | 77 characters |
| Sequence length | 128 |
| Embedding size | 64 |
| Hidden size | 192 |
| Batch size | 32 |
| Steps per epoch | 1,000 |
| Epochs | 100 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Gradient clipping | 5.0 |

No PyTorch or TensorFlow is required.

---

## Training Data

Veyra is trained on four public-domain English books:

- **Alice's Adventures in Wonderland** — Lewis Carroll
- **The Wonderful Wizard of Oz** — L. Frank Baum
- **The Adventures of Sherlock Holmes** — Arthur Conan Doyle
- **The Time Machine** — H. G. Wells

The books are stored individually inside `data/` and combined into `data.txt` using `combine_data.py`.

Current combined dataset:

```text
Characters: 1,114,179
Vocabulary: 77
```

The dataset files are included in the repository, so an existing checkout does not need to download them again.

---

## Training Results

### v0.1.0 baseline

The original training run reached approximately:

```text
Final loss: ~2.24
```

Generation was often highly fragmented and repetitive.

### v0.2.0

The upgraded training system reached:

```text
Final loss: 1.0989
Best loss:  1.0989
```

The loss decreased steadily throughout training, while the gradient norm remained finite during the completed run.

Generation also showed a clear improvement in learned language patterns, including:

- More recognizable English words
- Better punctuation
- More natural-looking sentence structure
- Dialogue formatting
- Longer coherent-looking sequences
- Stronger patterns from the books in the training corpus

Veyra is still a very small character-level model, so semantic coherence is limited and generated text can contain malformed words, mixed contexts, and nonsensical sentences. That is expected for the current architecture and scale.

---

## Requirements

- Windows 10/11 or a compatible Python environment
- Python 3
- NumPy

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

---

## Installation

Open PowerShell or Windows Terminal and navigate to the project:

```powershell
cd path\to\Veyra
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the requirements:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Prepare the Dataset

The repository already contains the current dataset.

If you want to rebuild `data.txt` from the individual books:

```powershell
python combine_data.py
```

This reads every `.txt` file in `data/` and combines them into `data.txt`.

Example output:

```text
Adding: alice.txt
Adding: oz.txt
Adding: sherlock.txt
Adding: time_machine.txt

Books: 4
Characters: 1,114,179
Saved: data.txt
```

### Replacing the Dataset

If you experiment with a new dataset, back up the current one first:

```powershell
Copy-Item data.txt data_old.txt
```

`data_old.txt` is a local backup and is not required for training.

---

## Train Veyra

To start a completely new training run:

```powershell
python train.py
```

The script will:

1. Load `data.txt`
2. Build the character vocabulary
3. Initialize the model
4. Train using Adam
5. Apply global gradient clipping
6. Save a checkpoint after every epoch
7. Store the final model in `weights.pkl`

A completed run ends with output similar to:

```text
============================================================
Training done.
Final loss: 1.0989
Best loss:  1.0989
Weights saved to weights.pkl
============================================================
```

### Checkpoints

`weights.pkl` contains:

- Model weights
- Character vocabulary
- Model configuration
- Optimizer state
- Current epoch
- Best loss
- Training metadata
- Veyra version

The checkpoint system also supports resuming training.

In `train.py`:

```python
RESUME = True
```

will attempt to continue from the existing `weights.pkl`.

---

## Generate Text

After training:

```powershell
python chat.py
```

You will see:

```text
==================================================
Veyra 0.2.0
==================================================
Character-level language model
Type 'quit' to exit.
```

Enter a prompt such as:

```text
Alice
```

or:

```text
The rabbit
```

or:

```text
Once upon a time
```

Veyra will generate a continuation based on the learned character patterns.

To exit:

```text
quit
```

---

## Example

A v0.2.0 generation can produce text with patterns such as:

```text
Alice waselong in our perfectrops. an eldergration, and then
the name was more and we cannot do in his face of the window...
```

The text is not expected to be grammatically or semantically perfect. The important improvement in v0.2.0 is that the model has learned much stronger character, word, punctuation, and sentence-level patterns than the original baseline.

---

## Project Structure

```text
Veyra/
├── .gitignore
├── Veyra-README.md
├── train.py
├── chat.py
├── combine_data.py
├── requirements.txt
├── data.txt
└── data/
    ├── alice.txt
    ├── oz.txt
    ├── sherlock.txt
    └── time_machine.txt
```

### Main Files

**`train.py`**

The complete NumPy training implementation.

**`chat.py`**

Loads the trained checkpoint and generates text interactively.

**`combine_data.py`**

Combines all `.txt` files in `data/` into `data.txt`.

**`data/`**

Contains the individual training books.

**`data.txt`**

The combined training corpus.

**`weights.pkl`**

The trained model checkpoint. It is intentionally ignored by Git because it is a generated binary file.

**`requirements.txt`**

Contains the project's Python dependency:

```text
numpy
```

---

## Quick Start

If Python is already installed:

```powershell
cd path\to\Veyra

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

python train.py
python chat.py
```

---

## Version History

### v0.2.0

**Training and stability upgrade**

- Replaced the previous optimizer with Adam
- Added global gradient clipping
- Added forget-gate bias initialization
- Increased sequence length from 64 to 128
- Added checkpoint saving
- Added resume support
- Added optimizer state persistence
- Trained on the full 1.11M-character dataset
- Reached a final loss of **1.0989**
- Improved generated text quality over v0.1.0

### v0.1.0

**Initial Veyra baseline**

- First complete character-level recurrent language model
- Pure NumPy implementation
- Multi-book public-domain dataset
- Interactive text generation
- Initial Git release

---

## Project Goal

Veyra is primarily a **from-scratch learning project**.

The goal is not to compete with modern large language models. Instead, the project is an experiment in understanding the fundamentals behind language modeling:

```text
Text
 ↓
Characters
 ↓
Vocabulary
 ↓
Embeddings
 ↓
Recurrent Network
 ↓
Predicted Next Character
 ↓
Generated Text
```

Future versions may experiment with larger models, improved training, better sampling, GPU acceleration, and other architectures while keeping the project understandable and reproducible.

---

## License / Source Material

The model code is part of the Veyra project.

The training corpus consists of public-domain works obtained from Project Gutenberg. The individual source texts retain their respective original public-domain status and source information.
