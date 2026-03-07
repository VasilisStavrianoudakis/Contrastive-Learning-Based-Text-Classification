# 🚀 Contrastive Learning for Text Classification

<div align="center">

[![Python 3.12](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)

*A Supercharged two-phase text classification training pipeline with: contrastive pre-training + asymmetric loss. Perfect for imbalanced datasets!*

</div>

---

## ✨ What Makes This Different?

Most text classification projects jump straight to fine-tuning pre-trained models. **This repository takes it a step further** by offering a two-phase training approach that can significantly improve your model's performance:

1. **🎯 Phase 1 (Optional):** Further pre-train your base model using **Contrastive Learning** with a custom loss function directly implemented from the [Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning](https://arxiv.org/abs/2011.01403) paper
2. **📊 Phase 2:** Train your classification model using **Asymmetric Loss** functions designed to combat class imbalance - supporting binary, multiclass, and multilabel tasks. The loss functions are described in the [Asymmetric Loss for Multi-Label Classification](https://arxiv.org/abs/2009.14119) paper and were implemented by [this](https://github.com/Alibaba-MIIL/ASL) repository.

### 🔑 Key Features

- **Custom Contrastive Loss Implementation** - Direct implementation of the supervised contrastive learning loss from research papers
- **Asymmetric Loss Functions** - Built-in support for handling imbalanced datasets with binary, multiclass, and multilabel variants
- **HuggingFace Integration** - Seamlessly works with any pre-trained transformer model. It also builds on the HuggingFace Trainer API.
- **Model Optimizations** - Prune the model to keep only the specific layers. Freeze specific layers to prevent them from being updated during training.
- **Flexible Training Pipeline** - Skip Phase 1 if you want to train directly, or use both phases for maximum performance
- **WandB Support** - Optional experiment tracking and visualization
- **Visualization Tools** - Notebook included for visualizing model projections

---

## 🏗️ Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    Pre-trained Model                        │
│          (e.g., nlpaueb/bert-base-greek-uncased-v1)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Phase 1: Contrastive       │
        │   Pre-training (Optional)    │
        │                              │
        │  Custom Supervised           │
        │  Contrastive Loss            │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Enhanced Base Model        │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Phase 2: Classification    │
        │   Training                   │
        │                              │
        │  Asymmetric Loss             │
        │  (Handles Class Imbalance)   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Final Classification Model │
        └──────────────────────────────┘
```

---

## 📚 Technical Details

### Phase 1: Contrastive Pre-training

We implement the **Supervised Contrastive Learning** approach, which learns better representations by:

- Pulling samples from the same class closer together in embedding space
- Pushing samples from different classes further apart
- Using a custom loss function that outperforms traditional fine-tuning approaches

**Paper:** [Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning](https://arxiv.org/abs/2011.01403)

### Phase 2: Classification with Asymmetric Loss

The downstream classification phase uses **Asymmetric Loss** functions specifically designed to handle:

- **Class imbalance** - Automatically adjusts for underrepresented classes
- **Multiple task types** - Binary, multiclass, and multilabel classification
- **Better convergence** - More stable training on imbalanced datasets

**Paper:** [Asymmetric Loss](https://arxiv.org/abs/2009.14119)

---

## 🛠️ Built With

- **PyTorch** - Deep learning framework
- **HuggingFace Transformers** - Pre-trained models and Trainer API
- **Weights & Biases** - Experiment tracking (optional)

---

## 🚦 Quick Start

### Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step-by-Step Guide

#### 1. Prepare Your Data

```bash
# Create data directory
mkdir -p data/raw

# Place your dataset in data/raw
# Then use the preprocessing notebook
jupyter notebook notebooks/prepare_dataset.ipynb
```

This notebook will help you:

- Pre-process your text data
- Create train/validation/test splits
  - Create two validation sets, one for classification and one for contrastive learning, to avoid data leakage.
- Prepare data in the required format

#### 2. (Optional) Phase 1: Contrastive Pre-training

Train an enhanced base model using contrastive learning:

```bash
python scripts/cl_training.py
```

**Configuration:** Edit `scripts/cl_config.json` to customize:

- Learning rate, batch size, epochs
- Base model path
- Task-specific parameters
- WandB logging settings

**Example config:**

```json
{
    "base_model_path": "nlpaueb/bert-base-greek-uncased-v1",
    "lr": 2e-5,
    "batch_size": 32,
    "epochs": 200,
    "enable_wandb_logging": false
}
```

#### 3. Phase 2: Classification Training

Train your final classification model:

```bash
python scripts/task_training.py
```

**Configuration:** Edit `scripts/task_config.json`:

**If you completed Phase 1:**

```json
{
    "base_model_name": "your_phase1_model_name",
    "comes_from_phase1": true,
    "lr": 2e-4,
    "epochs": 40
}
```

**If skipping Phase 1 (using HuggingFace model directly):**

```json
{
    "base_model_name": "nlpaueb/bert-base-greek-uncased-v1",
    "comes_from_phase1": false,
    "lr": 2e-4,
    "epochs": 40
}
```

#### 4. Visualize Results (Optional)

Explore your model's learned representations:

```bash
jupyter notebook notebooks/visualize_projections.ipynb
```

---

## 📊 WandB Integration (Optional)

Track experiments, visualize metrics, and compare runs with Weights & Biases.

### Setup

1. Create an account at [wandb.ai](https://wandb.ai/site)
2. Create `scripts/wandb.json`:

```json
{
    "key": "YOUR_API_KEY",
    "project": "your-project-name",
    "entity": "your-entity-name"
}
```

1. Enable logging in config files:

   - Set `"enable_wandb_logging": true` in `cl_config.json` and/or `task_config.json`

---

## ⚠️ Important Notes

### Multilabel Classification Limitation

The repository currently doesn't **directly** support Phase 1 (contrastive pre-training) for multilabel classification tasks, as the contrastive loss function works with binary and multiclass targets.

**Workarounds:**

- **Option 1:** Skip Phase 1 and train directly with Phase 2 (multilabel is fully supported here)
- **Option 2:** Convert multilabel targets to multiclass for Phase 1, then use original multilabel targets in Phase 2 (see [Label Powerset](http://scikit.ml/api/skmultilearn.problem_transform.lp.html))

---

## 📁 Project Structure

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

---

## 🎯 Use Cases

Perfect for:

- **Imbalanced text classification** - When some classes have far fewer examples
- **Domain adaptation** - Adapting pre-trained models to specific domains
- **Low-resource languages** - Improving performance on languages with limited pre-trained models
- **Research** - Experimenting with contrastive learning approaches

---

## 📖 References

- [Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning](https://arxiv.org/abs/2011.01403)
- [Asymmetric Loss for Multi-Label Classification](https://arxiv.org/abs/2009.14119)
- [Contrastive Learning Guide](https://encord.com/blog/guide-to-contrastive-learning/)
