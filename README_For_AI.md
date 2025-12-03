#  Malicious Backlink Prediction — **Kaggle Edition (.ipynb)**

This repo uses **Jupyter notebooks (.ipynb)** on **Kaggle** for both **training** and **inference**.  

> **Training notebook:** `training.ipynb`  
> **Inference notebook:** `inference.ipynb`  
> **Author:** Huỳnh Ngọc Như Quỳnh  
> **Last updated:** 2025-10-17 08:23

---

## ✨ Highlights

- 15 classes: `gambling, movies, ecommerce, government, education, technology, tourism, health, finance, media, nonprofit, realestate, services, industries, agriculture`
- Vietnamese word segmentation via **py_vncorenlp**
- Hugging Face **Trainer** + **custom Focal Loss (γ=2.0)** + **label smoothing (0.2)** + **class weights**
- Token length diagnostics, stratified split, eval metrics (accuracy / precision / recall / weighted F1)
- Misclassified samples exported to CSV
- Inference notebook supports **single text** and **batch CSV**

---

## 📁 Repository Structure

```text
repo/
├─ README.md
├─ notebooks/
│  ├─ training.ipynb
│  └─ inference.ipynb
├─ model/                                        
├─ dataset/
└─  └─ dataset.csv
```

---

## 🗃️ Data & Labels

- **Columns required:**  
  - `text` — raw Vietnamese string  
  - `label` — one of 15 class names

- **Label mapping (0–14):**
```python
{
  "gambling":0, "movies":1, "ecommerce":2, "government":3, "education":4,
  "technology":5, "tourism":6, "health":7, "finance":8, "media":9,
  "nonprofit":10, "realestate":11, "services":12, "industries":13, "agriculture":14
}
```

---

## 📓 Training Notebook (`training.ipynb`)
1. **Install & imports** (`%pip install ...`, `import torch`, `transformers`, `py_vncorenlp`, etc.)
2. **Reproducibility** — set global seed `42`
3. **VnCoreNLP** — download or load the model, init segmenter
4. **Load CSV to HF Dataset** — map labels → IDs
5. **Preprocess** — word-seg → PhoBERT tokenize (`max_length=64`, truncation & pad)
6. **Train/val split** — stratified ~90/10 (quick sanity-check eval)
7. **Class weights** — `compute_class_weight('balanced')`
8. **Loss/Trainer** — custom **Focal Loss (γ=2.0, label_smoothing=0.2)**; `TrainingArguments`
9. **Train** — epochs=20, batch=32, lr=2e-5, wd=0.01, warmup=500, early stopping (patience=5, threshold=0.001)
10. **Save artifacts** — model & tokenizer to: `/kaggle/working/fine_tuned_phobert/`
11. **Evaluate** — print metrics; export `misclassified_samples.csv` to `/kaggle/working/`
12. *(Optional)* **Plots** — loss/metrics curves (cells can be toggled on)

---

## 🧪 “Data‑Max” Mode (optional)

**Goal:** train on **100%** labeled data; evaluate on a **small stratified subset** from the same data for a quick health check.

- Use **only** for the final fit after hyperparameters are chosen on a proper split or k‑fold CV.
- Metrics are **optimistic** (data leakage) — document clearly.

**Notebook hint:** set
```python
train_dataset = tokenized_dataset
eval_dataset  = tokenized_dataset.select(eval_indices)  # small stratified subset
```
---

## 📓 Inference Notebook (`inference.ipynb`)

---
