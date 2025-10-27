# DIME: Diffusion-based Multi-outcome Generation

PyTorch Implementation of [KDD2025] **A Diffusion-Based Method for Learning the Multi-Outcome Distribution of Medical Treatments**

## Overview

DIME is a diffusion-based generative model designed to learn conditional distributions of multiple continuous outcomes given input features. The model uses a masked transformer architecture combined with diffusion-based sampling to generate realistic multi-outcome predictions for tabular data, particularly suited for medical treatment scenarios.


## Installation

### Requirements

- Python >= 3.12
- PyTorch
- See [pyproject.toml](pyproject.toml) for complete dependency list

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd DIME

# Install dependencies (using uv or pip)
uv sync
# or
pip install -e .
```

## Quick Start

```
python create_syn.py
python preprocess.py
python run_example.py
```
