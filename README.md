# Why Warmup the Learning Rate? Underlying Mechanisms and Improvements

This repository contains the code and data supporting our paper on the benefits and underlying mechanisms of learning rate warmup in deep learning optimization. Our systematic experiments with SGD and Adam reveal the critical role of warmup in enabling larger target learning rates, thereby improving network performance and robustness in hyperparameter tuning.

## Abstract
It is common in deep learning to warm up the learning rate $\eta$, often by a linear schedule between $\eta_{\text{init}} = 0$ and a predetermined target $\eta_{\text{trgt}}$. In this paper, we show through systematic experiments with SGD and Adam that the overwhelming benefit of warmup arises from allowing the network to tolerate larger $\eta_{\text{trgt}}$ by forcing the network to more well-conditioned areas of the loss landscape. The ability to handle larger $\eta_{\text{trgt}}$ makes hyperparameter tuning more robust while improving the final performance. We uncover different regimes of operation during the warmup period, depending on whether training starts off in a progressive sharpening or sharpness reduction phase, which in turn depends on the initialization and parameterization. Using these insights, we show how $\eta_{\text{init}}$ can be properly chosen by utilizing the loss catapult mechanism, which saves on the number of warmup steps, in some cases completely eliminating the need for warmup. We also suggest an initialization for the variance in Adam which provides benefits similar to warmup. 

## Code
The repository includes implementation of the experiments conducted using various network architectures and optimizers. The code is structured to facilitate easy replication of our results and further experimentation.

## Usage
Instructions for setting up the environment, running the experiments, and analyzing the results are provided in the respective folders. Each experiment folder contains a README with detailed steps and configurations used.

## Citation
If you find our research useful, please consider citing:

@inproceedings{warmup2024,
  title={Why Warmup the Learning Rate? Underlying Mechanisms and Improvements},
  author={Anonymous},
  booktitle={38th Conference on Neural Information Processing Systems},
  year={2024}
}

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

