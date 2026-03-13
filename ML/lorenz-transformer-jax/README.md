<div align="center">
  <h1>Lorenz Transformer in JAX</h1>
  <p>
    A decoder-only Transformer for autoregressive forecasting of chaotic Lorenz trajectories.
  </p>
  <p>
    <img alt="JAX" src="https://img.shields.io/badge/Framework-JAX-black">
    <img alt="Task" src="https://img.shields.io/badge/Task-Chaotic%20Forecasting-blue">
    <img alt="Architecture" src="https://img.shields.io/badge/Architecture-Decoder--Only%20Transformer-teal">
    <img alt="Domain" src="https://img.shields.io/badge/Domain-Dynamical%20Systems-orange">
  </p>
</div>

## Overview

This project explores how a Transformer can model a nonlinear dynamical system instead of natural language.

The notebook trains a causal decoder-only Transformer on procedurally generated trajectories of the Lorenz system. Given a context window of past states, the model predicts the next state autoregressively and is then evaluated both with standard loss curves and with a free rollout into the future.

The result is a compact research-style notebook that combines:

- synthetic data generation with the Lorenz equations
- sinusoidal positional encoding
- masked multi-head self-attention
- stacked decoder blocks with residual connections and feed-forward layers
- autoregressive rollout in 3D phase space
- attention-map visualization from the final decoder block

## Project Snapshot

| Item | Value |
| --- | --- |
| Task | Next-step prediction on Lorenz trajectories |
| Framework | JAX + Matplotlib |
| Input per token | 3D state `(x, y, z)` |
| Context length | `128` |
| Layers | `4` decoder blocks |
| Attention heads | `8` |
| Model width | `128` |
| Feed-forward width | `512` |
| Batch size | `32` |
| Training schedule | `80` epochs, `100` steps per epoch |
| Learning rate | `3e-4` |
| Rollout horizon shown | `200` steps |
| Approx. parameter count | `793,987` |

## Results

The executed notebook reports:

- best observed validation loss: `0.000625`
- final epoch validation loss: `0.007021`
- GPU-backed JAX execution during the recorded run

In addition to the loss curves, the notebook includes:

- a 3D comparison between true and predicted Lorenz rollouts
- rollout error over time
- a causal self-attention heatmap from the last decoder block

Because the Lorenz system is chaotic, long-horizon divergence is expected even for a strong model. The rollout visualizations are therefore an important part of the story, not just the scalar validation loss.

## Why This Project Is Interesting

This notebook is a strong demonstration piece because it shows that Transformer mechanics are useful outside text generation.

Instead of language tokens, the model consumes continuous physical states. Instead of next-word prediction, it learns short-horizon dynamics in a chaotic system. That makes the project a clean bridge between deep learning engineering and scientific machine learning.

## Repository Layout

- [`lorenz_transformer_jax.ipynb`](./lorenz_transformer_jax.ipynb) - main experiment notebook
- [`README.md`](./README.md) - project overview and model-style description

## Running The Notebook

1. Install JAX for your hardware backend by following the official JAX installation guide.
2. Install plotting support:

```bash
pip install matplotlib
```

3. Open and run [`lorenz_transformer_jax.ipynb`](./lorenz_transformer_jax.ipynb).

The Lorenz trajectories are generated on the fly, so no external dataset download is required.

## Intended Use

This project is best suited for:

- learning how decoder-only Transformers work in continuous domains
- studying causal attention on time-series data
- showcasing JAX-based model building in a compact notebook
- using chaotic dynamics as a toy scientific ML benchmark

## Limitations

- The project is a research notebook, not a packaged training library.
- Training is demonstrated on synthetic Lorenz data only.
- Long-range rollout accuracy is fundamentally limited by chaotic dynamics.
- There is no checkpointing, experiment tracking, or baseline comparison yet.

## Next Improvements

- add checkpoint saving and reproducible experiment configs
- compare against MLP, RNN, or linear baselines
- test longer context windows and different rollout horizons
- export figures for a cleaner GitHub or portfolio presentation
