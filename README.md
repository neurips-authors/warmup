# Repository Title
This repository contains the source of code used in our paper. Here, we provide the scripts to replicate the experiments and results presented in the paper.

## Code Structure
The repository is structured to include implementations for various experiments using different network architectures and optimization algorithms:

- **Vision Folder**: Contains scripts for image classification tasks.

## Environment Setup
To ensure that the code runs smoothly, you must install the necessary libraries. Use the following command to install dependencies from the provided `requirements.txt`:

`pip install -r requirements.txt`

## Running Experiments
Detailed steps to reproduce the experiments in the `vision` folder:

1. **Warmup Mechanisms**: Demonstrates the sharpness evolution during warmup.
2. **Standard Training**: Used to reproduce the phase diagrams and includes experiments with GI-Adam.
3. **Initial Learning Rate Selection**: Reproduces the initial learning rate selection experiments.

### Additional Instructions for Google Colab
If you are using Google Colab, note that most libraries are pre-installed; however, you might still need to install some dependencies manually. Also, remember to set the path to your working directory in the notebooks appropriately.

## License
This project is distributed under the MIT License. See the `LICENSE.md` file for more details.
