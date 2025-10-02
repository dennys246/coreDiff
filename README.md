---
language:
- en
license: apache-2.0
tags:
- diffusion
- image-generation
pipeline_tag: image-generation
library_name: keras
datasets:
- rmdig/rocky_mountain_snowpack
---

# The coreDiff

A Diffusion model trained on ~1,300 images of snow cores from the Rocky Mountain Snowpack dataset, captured during the 2024-2025 ski season to understand features of the data. Long term use of this is for higher level embedding
of feature's learned from the diffusion in other tasks.

## Model Details
- **Framework**: TensorFlow, Keras
- **Architecture:** Custom Keras Functional model
- **Input**: 1000 noise vector 
- **Output**: 100x50 RGB snowpack image


- **Framework:** TensorFlow 2.15 / Keras 3
- **Input shape:** (100, 50, 3)  # Noise seed
- **Output:** (100, 50, 3)  # Or whatever your task is
- **Training data:** [rmdig/rocky_mountain_snowpack]
- **Intended use:** [Research, demonstration, pre-trained]

## Exploratory Usage

```bash
# Install the package to run locally
pip install -e .

# Run the package to start training a corediff
corediff --mode generate --synthetics 25

```

```python
# Generate using the corediff library
import corediff

# Load pre-trained model from hugging face repo
model = corediff.load_model("rmdig/corediff")

# Generate images with the model
_ = corediff.generate(model, count = 10, save_dir = "path/to/output/dir/")

```

## Training Usage

```bash

python3 main.py --mode train --epochs 10 --lr 0.001

```

```python
# Import corediff trainer class...
import corediff as corediff

model = corediff.load_model(model_path)

trainer = corediff.trainer(model_path)

trainer.train(batch_size = 8, epochs = 10)

trainer.save_model()
```

## Limitations

- Trained on ~1,300 snowpack images — limited dataset size.
- Outputs small resolution images (100x50).
- Not intended for production avalanche forecasting or operational decision making.
