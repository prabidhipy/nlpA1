# Word Embeddings from Scratch

**Skip-gram, Skip-gram with Negative Sampling, and GloVe**

**Author:** Prabidhi Pyakurel, st126380

---

## Overview

This project implements **word embedding models completely from scratch** and evaluates them using standard NLP benchmarks.
A simple **web application** is also provided to search for similar contexts using the trained embeddings.

---

## Implemented Models

The following models are trained **without using any pretrained embeddings**:

1. **Skip-gram with Full Softmax**
2. **Skip-gram with Negative Sampling**
3. **GloVe (Global Vectors)**

All models are trained on the **Reuters-21578** corpus.

---

## Datasets

* **Reuters-21578 Corpus** (NLTK)
  David D. Lewis, 1997
* **Word Analogy Dataset**
  Mikolov et al., 2013
* **WordSim-353**
  Finkelstein et al., 2001

---

## Task Breakdown

### Task 1 — Training Word Embeddings

* Text preprocessing: lowercasing, tokenization, cleaning
* Vocabulary construction with `<UNK>` token
* Dynamic context window sampling
* Training:

  * Skip-gram (full softmax)
  * Skip-gram with negative sampling (unigram distribution^0.75)
  * GloVe using co-occurrence matrix

---

### Task 2 — Evaluation

Models are evaluated using:

1. **Training Loss and Time**
2. **Word Analogy Tasks**

   * Semantic (e.g., capital-common-countries)
   * Syntactic (e.g., past tense)
3. **Word Similarity (WordSim-353)**

   * Cosine similarity
   * Spearman correlation
   * Mean Squared Error (MSE)

A **Gensim GloVe model** is used **only as a reference baseline**.

> Note: Lower absolute scores are expected due to the limited size and domain-specific nature of the Reuters corpus.

---

### Task 3 — Web Application

The web application allows users to:

* Enter a query word
* Retrieve the **top-10 most similar contexts**
* Similarity is computed using **cosine similarity** on averaged word vectors

---
---

## How to Run

### 1. Train Models

Run all cells in the notebook file named A1.ipynb

This will generate:

* `model_sg.pth`
* `model_neg.pth`
* `model_glove.pth`
* `model_data.pkl`

---

### 2. Run the Web Application

```bash
python3 app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## Implementation Notes

* Dynamic context window is used for Skip-gram training
* Sentence vectors are computed by **averaging word embeddings**
* Vectors are normalized so dot product corresponds to cosine similarity
* Out-of-vocabulary words are safely handled using `<UNK>`