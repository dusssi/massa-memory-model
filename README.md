# 🧠 MASSA Prototype

### Memory-Augmented Sequence Modeling

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat\&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat\&logo=pytorch)
![Status](https://img.shields.io/badge/Project-Research%20Prototype-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project explores how **memory mechanisms improve sequence learning**.

Traditional models like LSTM compress entire sequences into hidden states, often losing important information.
This project introduces a **Memory-Augmented Model with attention**, allowing the model to focus on key parts of the sequence.

---

## 🎯 Objective

* Compare **LSTM vs Memory-Augmented Model**
* Understand **long-term dependency learning**
* Show impact of **attention-based memory**
* Build a simple, explainable research prototype

---

## 🧪 Problem Setup

### Input Sequence

```
[3, 8, 1, 5, 9, ...]
```

### Target Output

```
(first_element + last_element) % 10
```

👉 Requires:

* remembering the **start**
* tracking the **end**
* combining both → real memory task

---

## 🏗️ Project Structure

```
massa-memory-model/
│
├── data/
│   └── dataset.py
│
├── models/
│   ├── lstm_model.py
│   └── memory_model.py
│
├── results/
│   ├── loss.png
│   └── accuracy.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/massa-memory-model.git
cd massa-memory-model

python -m venv .venv
.venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python main.py
```

---

## 📊 Results

### 🔹 Accuracy Comparison

![Accuracy](results/accuracy.png)

### 🔹 Training Loss

![Loss](results/loss.png)

---

## 📈 Sample Performance

| Model        | Accuracy |
| ------------ | -------- |
| LSTM         | 0.36     |
| Memory Model | 0.53     |

👉 Memory model shows clear improvement using attention.

---

## 🧠 Key Insight

> LSTM compresses sequence → loses information
> Memory model attends → keeps what matters

---

## 🔬 Explanation

* **LSTM Model**

  * Uses sequence summarization
  * Limited ability to capture long-range dependencies

* **Memory Model**

  * Uses attention mechanism
  * Dynamically focuses on important sequence elements
  * Better performance on dependency-heavy tasks

---

## 🚀 Future Work

* Add Transformer comparison
* Use real datasets (NLP / time-series)
* Improve attention mechanism
* Tune hyperparameters

---

## 📄 Research Context

Inspired by:

* Memory-Augmented Models
* State Space Models (Mamba)
* Attention-based architectures

---

## 👤 Author

**Dushyant**
MCA (AI & ML)
SGT University

---

## ⚠️ Disclaimer

This is a **research prototype** built for understanding concepts.
Accuracy is secondary to **learning behavior and comparison**.

---
