# CNN_A (NumPy CNN on MNIST)

`CNN_A.ipynb` contains a convolutional neural network implemented with **NumPy**.

The notebook reports a final validation accuracy of **0.9658** (about **0.96**) after training.

Note: MNIST is loaded via `tensorflow.keras.datasets`, while the model, forward/backward passes, and training logic are written in NumPy.

# RN_A (NumPy Residual CNN on MNIST)

`RN_A.ipynb` contains a residual-style convolutional neural network (ResBlock-based) implemented with **NumPy**.

The notebook trains and validates the model on MNIST, with batch-level progress logs during training.

Note: MNIST is loaded via `tensorflow.keras.datasets`, while the network layers, backprop, and optimizer update steps are written in NumPy.
