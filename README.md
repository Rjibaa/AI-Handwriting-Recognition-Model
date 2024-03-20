# Handwritten Digit Recognition AI

This repository contains the AI component of a professional personal project aimed at developing a mobile application for recognizing handwritten digits. The AI model is designed to recognize handwritten digits using Convolutional Neural Networks (CNNs) trained on the MNIST dataset.

## Contents

- **Flask.py:** This file contains the code for deploying the trained model and exposing it through a Flask API.
- **MNIST_CNN.h5:** The pre-trained CNN model for recognizing handwritten digits.
- **MNIST_CNN.ipynb:** Jupyter Notebook containing the architecture and code used for training the MNIST CNN model.
- **Preprocessing.py:** This script includes character segmentation and data preprocessing functions. It is utilized in the deployment of the model in Flask.py.

## Usage

1. **Model Deployment:** The Flask.py script can be used to deploy the trained model. It sets up a Flask API to expose the model's predictions.
2. **Training Model:** Use the MNIST_CNN.ipynb notebook to explore the architecture and train the MNIST CNN model. You can customize and experiment with different architectures and hyperparameters.
3. **Preprocessing:** The Preprocessing.py script provides functions for character segmentation and data preprocessing, which are essential for preparing input data for the model.

## Getting Started

To get started with using or contributing to this project:

1. Clone this repository to your local machine.
2. Set up the required dependencies by installing the necessary libraries (e.g., TensorFlow, Flask).
3. Explore the provided scripts and notebooks to understand the project structure and functionality.
4. Experiment with the model architecture, data preprocessing, or deployment settings to tailor it to your specific needs.

## Contributors

- [Rjibaa](https://github.com/Rjibaa)

Feel free to contribute by opening issues, suggesting enhancements, or submitting pull requests. Your contributions are highly appreciated!

## License

This project is licensed under the [MIT License](LICENSE).
