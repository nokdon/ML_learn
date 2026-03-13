# ML_learn

This repository is a personal machine learning playground with notebooks covering NumPy fundamentals, neural networks, CNNs, residual models, and JAX-based experiments.

The public-facing side of the repo is now curated: polished projects live in [`ML/`](./ML/README.md), while local study notebooks, datasets, and scratch files are kept out of version control where possible.

## Public Structure

- [`ML/`](./ML/README.md) - showcase projects with focused documentation
- root notebooks - older tracked experiments kept for continuity

## Featured Projects

### [Lorenz Transformer in JAX](./ML/lorenz-transformer-jax/README.md)

A decoder-only Transformer for autoregressive forecasting of the Lorenz system.

- Framework: `JAX`
- Task: chaotic time-series prediction
- Highlights: causal self-attention, multi-step rollout, attention-map visualization
- Main notebook: [`ML/lorenz-transformer-jax/lorenz_transformer_jax.ipynb`](./ML/lorenz-transformer-jax/lorenz_transformer_jax.ipynb)

### `CNN_A.ipynb`

A convolutional neural network implemented with `NumPy` and trained on MNIST.

- Reported final validation accuracy: `0.9658`
- Data source: `tensorflow.keras.datasets`
- Implementation: forward pass, backpropagation, and training loop written in NumPy

### `RN_A.ipynb`

A residual-style CNN implemented with `NumPy` and trained on MNIST.

- Uses residual blocks
- Trains with batch-level progress logs
- Data source: `tensorflow.keras.datasets`

## Notes

- The repository still contains legacy tracked notebooks from the learning process.
- The cleanest GitHub entry point right now is the Transformer project inside [`ML/`](./ML/README.md).
