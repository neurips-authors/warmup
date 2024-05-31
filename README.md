# Repository Title
This repository contains the source of code used in our paper. Here, we provide the scripts to replicate the experiments and results presented in the paper.

## Code Structure
The repository is structured to include implementations for various experiments using different network architectures and optimization algorithms:

- **vision folder**: Contains scripts for reproducing image classification experiments.
- **test folder**: Contains unit tests to check the correcteness of custom Adam implementation by comparing it to the optax implementation.

## Environment Setup
To ensure that the code runs smoothly, install the necessary libraries by running:

`pip install -r requirements.txt`

## Running Experiments
Description of contents in the `vision` folder:

1. **Warmup Mechanisms**: Demonstrates the sharpness evolution during warmup for different optimizers and architectures.
2. **Standard Training**: Used to reproduce the phase diagrams and includes experiments with GI-Adam.
3. **Initial Learning Rate Selection**: Reproduces the initial learning rate selection experiments.

### Additional Instructions for Google Colab
If you are using Google Colab, note that most libraries are pre-installed; however, you might still need to install some dependencies manually. Also, remember to set the path to your working directory in the notebooks appropriately.

## License
This project is distributed under the MIT License. See the `LICENSE.md` file for more details.
