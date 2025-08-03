---
language:
- en
license: apache-2.0
tags:
- gan
- image-generation
pipeline_tag: image-generation
library_name: pytorch
---

# The coreDiffusor

A Diffusor trained on ~1,300 images of snow cores captured during the 2024-2025 ski season to understand features of the data. Long term use of this is for higher level embedding.

## Model Details
- **Framework**: TensorFlow
- **Input**: 100D latent vector (z)
- **Output**: 64x64 RGB snowpack image

## Exploratory Usage

```python
import tensorflow as tf
import snowGAN

# Figure out if GPU with CUDAs are available, else set ot CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model

z = torch.randn(1, 100, 1, 1)
image = gen(z)
```

# Traing Usage
