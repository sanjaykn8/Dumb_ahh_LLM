# 🧠 TinyGPT — Build Your Own LLM From Scratch

> A minimalist Large Language Model built **from the ground up** using PyTorch.  
> No pre-trained weights. No shortcuts. Just pure learning.

---

## 🚀 Overview

TinyGPT is a lightweight transformer-based language model trained **character-by-character**,  
inspired by the core ideas behind GPT and modern attention mechanisms.

Built for those who want to **understand** how an LLM actually works — not just use one.

---

## 🧩 Features

- ⚙️ **From Scratch:** No Hugging Face, no pre-training. Everything coded manually.  
- 🧠 **Transformer Core:** Implements self-attention, feed-forward, and positional encoding.  
- 💾 **Offline Training:** Train locally on your own dataset.  
- 🔥 **Supports GPU (CUDA):** Accelerated by your RTX 4050 or equivalent GPU.  
- 💬 **Simple Inference:** Generate text directly from the command line.  
- 🪶 **Modular Design:** Easily swap tokenizer, model depth, or vocab system.

---

## 🧱 Project Structure

TinyGPT/
├── model.py # Transformer model (TinyGPT)
├── train.py # Training pipeline
├── infer.py # Inference script
├── token_utils.py # Encoding / decoding utilities
├── train.txt # Training data
└── opfiles/ # Model checkpoints

yaml
Copy code

---

## ⚡ Quick Start

### 1️⃣ Train Your Model

```bash
python train.py
This will:

Load train.txt

Train for 2000 steps (configurable)

Save checkpoints in /opfiles/

2️⃣ Run Inference
bash
Copy code
python infer.py --prompt "love"
Parameters you can tweak:

bash
Copy code
--ckpt opfiles/model_final.pt
--prompt "hello"
--max_new_tokens 120
--top_k 50
--temp 1.0
Example Output:

rust
Copy code
love ca'tr   u  tifu, csaa nbod cu p s   ,  g ta  r  g    s    k u t tarAnnfhndu...
(raw but improving as it trains — your model learns character patterns over time.)

🧠 How It Works
Component	Description
Tokenizer	Converts each byte into a token (0–255).
Transformer	Core of the model, using multi-head self-attention.
Training Loop	Learns next-token prediction over text data.
Sampling	Predicts next character using top-k sampling.

🔧 Configuration
Modify constants in train.py:

Parameter	Default	Description
SEQ_LEN	128	Sequence length
BATCH	4	Batch size
STEPS	2000	Total training steps
LR	3e-4	Learning rate
FP16	True	Enable mixed precision
GRAD_ACCUM	8	Gradient accumulation steps

🎨 Why “Tiny” GPT?
Because the goal isn’t to compete — it’s to comprehend.

You’ll understand:

How attention masks work

Why positional embeddings matter

How logits turn into probabilities

How transformers generate text one token at a time

🌌 Vision
TinyGPT is a seed — a way to see the machinery of intelligence without the noise.
Train it on your poetry, code, or scientific notes.
Let it echo back fragments of your dataset like dreams forming in silicon.

🛠️ Requirements
Python ≥ 3.10

PyTorch ≥ 2.0

tqdm

Install dependencies:

bash
Copy code
pip install torch tqdm