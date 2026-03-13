# ML_learn

This repository is a personal machine learning playground with notebooks covering NumPy fundamentals, neural networks, CNNs, residual models, and JAX-based experiments.

Some files are exploratory, while the more presentable projects are being grouped under [`ML/`](./ML/README.md).

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

- The repository currently contains many study notebooks alongside the showcase projects.
- If you want a cleaner public-facing GitHub profile, the best entry point right now is the Transformer project inside [`ML/`](./ML/README.md).
