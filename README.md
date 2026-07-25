# Contrastive Learning for Text Classification

<div align="center">

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)

*A **two-phase training pipeline** for text classification: **supervised contrastive pre-training**, then classification with **asymmetric loss**. Built with **imbalanced datasets** in mind.*

</div>

## At a glance

| Capability | What it does |
|---|---|
| **Contrastive pre-training** | Reshapes the base model's embeddings with a supervised contrastive loss before any classifier is attached |
| **Asymmetric loss** | Counters class imbalance directly in the objective — no resampling hacks required |
| **Task coverage** | Binary · multiclass · multilabel, out of the box |
| **Any HF transformer** | Point it at any model on the Hub; built on the `Trainer` API |
| **Layer control** | Prune to specific layers or freeze layers during training |
| **Two modes** | Full two-phase run for maximum lift, or Phase 2 alone for a one-command baseline |
| **Introspection** | A notebook to visualize learned projections and see what the contrastive stage did |
| **Tracking** | Optional Weights & Biases logging for comparing runs |

## Why two phases?

The usual recipe is to grab a pre-trained transformer and fine-tune it straight away. That's a fine starting point, but you tend to leave performance on the table when your classes are heavily imbalanced or your domain sits far from whatever the base model was trained on.

This repo splits training into **two stages** so you can tackle both problems:

**Phase 1 (optional)** continues pre-training your base model with *supervised contrastive learning*, using a loss taken directly implemented from [Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning](https://arxiv.org/abs/2011.01403). The goal is a base model with cleaner, more separable representations before you ever attach a classifier.

**Phase 2** trains the classifier itself using *asymmetric loss*, which is designed to cope with class imbalance across binary, multiclass, and multilabel tasks. The loss comes from [Asymmetric Loss for Multi-Label Classification](https://arxiv.org/abs/2009.14119); the implementation here is based on the [Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL) repo.

You can run both stages back to back, or **skip Phase 1 entirely** and go straight to a solid Phase 2 baseline.

## Key Features

- A **direct implementation of the supervised contrastive loss** from the paper above, ready to drop into pre-training.
- **Asymmetric loss functions for imbalanced data**, with binary, multiclass, and multilabel variants.
- **Works with any pre-trained transformer on the Hub**, and builds on top of the HuggingFace `Trainer` API rather than reinventing the training loop.
- **Layer pruning and freezing** — trim the model down to specific layers, or hold layers fixed during training.
- **Run the full two-phase pipeline or just Phase 2**.
- Optional **Weights & Biases tracking**.
- A notebook for **visualizing the learned projections**, so you can actually see what the contrastive stage did.

## Architecture Overview

```text
Pre-trained model  (e.g. nlpaueb/bert-base-greek-uncased-v1)
        │
        ▼
Phase 1 — Contrastive pre-training  (optional)
        │   supervised contrastive loss
        ▼
Enhanced base model
        │
        ▼
Phase 2 — Classification training
        │   asymmetric loss (handles class imbalance)
        ▼
Final classification model
```

## Technical Details

### Phase 1 — Contrastive pre-training

Supervised contrastive learning shapes the embedding space by pulling samples of the same class together and pushing different classes apart. In practice this gives you representations that a downstream classifier can separate more easily, and it tends to beat plain fine-tuning as a starting point.

Reference: [Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning](https://arxiv.org/abs/2011.01403)

### Phase 2 — Classification with asymmetric loss

Asymmetric loss treats positive and negative examples differently, which is exactly what you want when some classes are badly underrepresented. It handles binary, multiclass, and multilabel setups and generally trains more stably on skewed data.

Reference: [Asymmetric Loss for Multi-Label Classification](https://arxiv.org/abs/2009.14119)

## Built with

- **PyTorch** — the underlying deep learning framework
- **HuggingFace Transformers** — pre-trained models and the `Trainer` API
- **Weights & Biases** — experiment tracking (optional)

## Getting started

### Install

```bash
pip install -r requirements.txt
```

### 1. Prepare your data

```bash
mkdir -p data/raw
# drop your dataset into data/raw, then open the prep notebook
jupyter notebook notebooks/prepare_dataset.ipynb
```

The notebook walks you through pre-processing your text, building train/validation/test splits, and getting everything into the expected format. Note that it creates *two* validation sets — one for classification and one for contrastive learning — so the two phases don't leak into each other.

### 2. (Optional) Phase 1 — Contrastive pre-training

```bash
python scripts/cl_training.py
```

Settings live in `scripts/cl_config.json`: learning rate, batch size, epochs, the base model path, task-specific parameters, and WandB logging. For example:

```json
{
    "base_model_path": "nlpaueb/bert-base-greek-uncased-v1",
    "lr": 2e-5,
    "batch_size": 32,
    "epochs": 200,
    "enable_wandb_logging": false
}
```

### 3. Phase 2 — Classification training

```bash
python scripts/task_training.py
```

Edit `scripts/task_config.json` to point at the right base model. If you ran Phase 1, feed it that model and flag it accordingly:

```json
{
    "base_model_name": "your_phase1_model_name",
    "comes_from_phase1": true,
    "lr": 2e-4,
    "epochs": 40
}
```

If you skipped Phase 1, just point it at a model from the Hub:

```json
{
    "base_model_name": "nlpaueb/bert-base-greek-uncased-v1",
    "comes_from_phase1": false,
    "lr": 2e-4,
    "epochs": 40
}
```

### 4. (Optional) Visualize the results

```bash
jupyter notebook notebooks/visualize_projections.ipynb
```

This lets you inspect the representations the model learned — handy for sanity-checking what Phase 1 actually did.

## Weights & Biases (optional)

If you want experiment tracking and run comparisons:

1. Grab an account at [wandb.ai](https://wandb.ai/site).
2. Create `scripts/wandb.json`:

   ```json
   {
       "key": "YOUR_API_KEY",
       "project": "your-project-name",
       "entity": "your-entity-name"
   }
   ```

3. Set `"enable_wandb_logging": true` in `cl_config.json`, `task_config.json`, or both.

## Multilabel Classification Limitation

The repository currently doesn't *directly* support Phase 1 (contrastive pre-training) for multilabel classification tasks, as the contrastive loss function works with binary and multiclass targets.

**Workarounds:**

- **Option 1:** Skip Phase 1 and train directly with Phase 2 (multilabel is fully supported here)
- **Option 2:** Convert multilabel targets to multiclass for Phase 1, then use original multilabel targets in Phase 2 (see [Label Powerset](http://scikit.ml/api/skmultilearn.problem_transform.lp.html))

## Project layout

```text
Contrastive-Learning-Based-Text-Classification/
│
├── notebooks/
│   ├── prepare_dataset.ipynb      # Data preprocessing and splitting
│   └── visualize_projections.ipynb # Model visualization
│
├── scripts/
│   ├── cl_training.py              # Phase 1: Contrastive pre-training
│   ├── task_training.py            # Phase 2: Classification training
│   ├── losses.py                   # Custom loss implementations
│   ├── training_utils.py           # Training utilities
│   ├── cl_config.json              # Phase 1 configuration
│   └── task_config.json            # Phase 2 configuration
│
├── data/
│   └── raw/                        # Place your dataset here
│
└── requirements.txt
```

## Where this is useful

- **Imbalanced text classification**, where a handful of classes dominate and the rest are starved for examples.
- **Domain adaptation**, when you're pushing a general-purpose model into a specialized field.
- **Low-resource languages**, where pre-trained coverage is thin and every bit of representation quality helps.
- **Research and experimentation** with contrastive approaches to classification.

## References

- [Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning](https://arxiv.org/abs/2011.01403)
- [Asymmetric Loss for Multi-Label Classification](https://arxiv.org/abs/2009.14119)
- [A guide to contrastive learning](https://encord.com/blog/guide-to-contrastive-learning/)
