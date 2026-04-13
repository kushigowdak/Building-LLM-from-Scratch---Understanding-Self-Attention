# 🧠 Building LLM from Scratch — Understanding Self-Attention (PyTorch)

This notebook demonstrates **Self-Attention** from scratch using **PyTorch**, one of the core building blocks behind **Large Language Models (LLMs)** like GPT and Transformers.

---

# 📌 Overview

This lab walks through:

* Dot-Product Attention
* Attention Scores Calculation
* Softmax Normalization
* Context Vector Creation
* Full Self-Attention Matrix
* Scaled Dot-Product Attention

The implementation is built **step-by-step** to help understand how LLMs actually work internally.

---

# 🎯 Objectives

After completing this notebook, you will understand:

* What Self-Attention is
* How tokens attend to each other
* Query, Key, Value intuition
* Attention weights calculation
* Context vector generation
* Matrix-based Self-Attention

---

# 📚 Input Example

The notebook starts with token embeddings:

```
"Your journey starts with one step"
```

Each word is converted into a vector representation:

```
Your
Journey
Starts
With
One
Step
```

These vectors are used to compute **attention relationships** between words.

---

# 🏗️ Implementation Steps

## Step 1 — Import PyTorch

```python
import torch
```

---

## Step 2 — Create Input Embeddings

Token embeddings are defined manually:

```python
inputs = torch.tensor([
 [0.43, 0.15, 0.89],
 [0.55, 0.87, 0.66],
 [0.57, 0.85, 0.64],
 [0.22, 0.58, 0.33],
 [0.77, 0.25, 0.10],
 [0.05, 0.80, 0.55]
])
```

---

## Step 3 — Compute Attention Scores

Using dot product:

```python
attn_scores = torch.dot(x_i, query)
```

This measures similarity between tokens.

---

# 📐 Attention Formula

Self-Attention is computed as:

```
Attention(Q, K, V) = softmax(QKᵀ)V
```

Where:

* Q → Query
* K → Key
* V → Value

---

# 🔢 Step 4 — Normalize Using Softmax

Softmax converts scores into probabilities:

```python
attn_weights = torch.softmax(attn_scores, dim=0)
```

Properties:

* Values between 0 and 1
* Sum equals 1

---

# 🎯 Step 5 — Compute Context Vector

Weighted sum of tokens:

```python
context_vec = attention_weights * inputs
```

This produces contextual representation.

---

# 🔁 Step 6 — Full Self-Attention Matrix

Compute attention for all tokens:

```python
attn_scores = inputs @ inputs.T
```

This creates:

```
6 × 6 Attention Matrix
```

Each token attends to every other token.

---

# ⚡ Matrix Multiplication Optimization

Instead of loops:

```
inputs @ inputs.T
```

This speeds up computation significantly.

---

# 🚀 Key Concepts Covered

* Dot Product Attention
* Softmax Attention
* Context Vector
* Self-Attention Matrix
* Vector Similarity
* PyTorch Tensor Operations

---

# 📦 Requirements

Install dependencies:

```bash
pip install torch
```

---

# ▶️ Run Notebook

```bash
jupyter notebook LAB-6_1rvu23cse232.ipynb
```

---

# 📈 Learning Outcomes

You will learn:

* How attention works internally
* How GPT-style models process tokens
* How to implement Self-Attention from scratch

---

# 🔜 Next Steps

After this lab:

* Multi-Head Attention
* Transformer Encoder
* Transformer Decoder
* Build Mini GPT

---

# 🎓 Educational Purpose

This notebook is designed for:

* Students learning Transformers
* Deep Learning beginners
* NLP practitioners
* LLM enthusiasts

---

# 📄 License

Educational Use Only

---

# 👨‍💻 Lab

**Building LLM from Scratch — Understanding Self-Attention**
