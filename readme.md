# Neural network
## Tested and trained on MNIST Digits dataset 

This is a neural network, built completely from scratch using cupy (numpy for GPUs) to solidify my understanding of backprop, optimisers and other fundemental Nerual Network concepts

## Features:
* Dense layers with L1/L2 regularization
* ReLU, Softmax, Sigmoid activations
* Dropout
* Categorical cross-entropy loss
* SGD (with momentum) and Adam optimizers
* Automatic architecture search (grows/widens layers, adds dropout based on validation performance)
* Automatic hyperparameter tuning
* Model save/load (pickle-based)

## Final Accuracy - 97.36%
Total Time taken ()

## Requirements
* Python 3+
* Built with cupy - requires a CUDA core GPU and toolkit
* * If want to use numpy instead, replace "import cupy as np" at the start of "main.py", "model.py" and "loader.py" with "import numpy as np"

## Usage
"loader.py" is specifically for the MNNIST Digits dataset. If using the same dataset, put the data files in the same directory and input the correct file names into
the appropriate functions in "main.py"
If you have wish to use it on a separate dataset, you will need your own script for loading which presents the same output structure as "loader.py"
Other tweaks can be made - especially to the hardcoded values inside the automatic optimisers, depending on the compute/accuracy tradeoff

## Limitations
Main limitations is the lack of variety - only Categorical Cross Entropy as loss function, Static Gradient Descent and Adam optimiser as Optimisers and only fully-
connected architecture available, however with the current framework, I believe integrating these components wouldn't be a specifically difficult task.

Another limitation is that layers dont have different Regularisation values, as the hyperparameter optimiser doesnt deal with them and they are not saved by the modelSave()
function. They are hardcoded at 5e-4.

Similarly, all dropout functions have a rate of 0.45.

## Credits
The mathematical section of this code was built following the book "Neural Networks from Scratch" by Harrison Kinsley and Daniel Kukieła.
The code is not identical as I made some additions to help it run with my framework, but the mathematical logic was taken from this book.

The dataset that I used can be found here: https://www.kaggle.com/datasets/hojjatk/mnist-dataset