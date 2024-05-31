# Repository Title
This repository contains the source of code and datasets used in our paper. Here, you'll find everything needed to replicate the experiments and results we have discussed.

## Code Structure
The repository is structured to include implementations for various experiments using different network architectures and optimization algorithms:

- **Vision Folder**: Contains scripts for image classification tasks.
- **NLP Folder**: (if applicable) Includes code for natural language processing experiments.
- ⚙️ **Other Folders**: Describe other folders if present.

## Environment Setup
To ensure that the code runs smoothly, you must install the necessary libraries. Use the following command to install dependencies from the provided `requirements.txt`:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Additional Instructions for Google Colab
If you are using Google Colab, note that most libraries are pre-installed; however, you might still need to install some dependencies manually. Also, remember to set the path to your working directory in the notebooks appropriately.

## Running Experiments
Detailed steps to reproduce the experiments:

1. **Warmup Mechanisms**: Demonstrates the evolution of sharpness in the training process.
2. **Standard Training**: Helps in reproducing phase diagrams and includes experiments with GI-Adam optimizer.
3. **Initial Learning Rate Selection**: Guides through choosing the initial learning rate for training.

Each set of experiments can be found in the `vision` folder and are designed to be easy to replicate.

## License
This project is distributed under the MIT License. See the `LICENSE.md` file for more details.
