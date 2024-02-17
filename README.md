# Multi-Label Classification Model Evaluation

## Overview

This Python script is designed to evaluate the performance of multi-label classification models on various datasets using [scikit-multiflow](https://scikit-multiflow.github.io).

The script supports the evaluation of different base classifiers, prediction functions, and metric functions. The evaluated models include Logistic Regression, Stochastic Gradient Descent (SGD) Classifier, Random Forest Classifier, and AdaBoost Classifier.

## Key Features

- **Modular Code Structure:** The code is organized into classes and functions for improved readability and maintainability.

- **Flexible Dataset Handling:** Supports reading datasets from a specified folder and allows for train-test split. Handles both individual datasets and datasets stored in subfolders.

- **Multiple Evaluation Metrics:** Evaluates models based on metrics such as Accuracy, Hamming Loss, and Zero-One Loss.

- **Scikit-Multiflow Integration:** Utilizes the `scikit-multiflow` library for handling multi-label classification scenarios.

- **Dynamic Model Configuration:** Easily configure models, datasets, and evaluation parameters to suit your requirements.

- **Result Output:** Evaluation results are saved to a CSV file for further analysis and comparison.

## Usage

1. **Installation:** Ensure you have the required dependencies installed. You can install them using `pip install -r requirements.txt`.

2. **Dataset Preparation:** Place your datasets in the specified folder or create subfolders for individual datasets.

3. **Configuration:** Modify the script to specify the models, datasets, and evaluation metrics you want to use.

4. **Run the Script:** Execute the script (`python inference_evaluate_models.py`) to train and evaluate models on the specified datasets.

5. **Review Results:** Check the generated CSV file (`evaluation_results.csv`) for detailed evaluation metrics.

## Requirements

- Python 3.6+
- numpy
- scikit-multiflow

## Installation

**Note:** The installed version is really important!

```bash
conda create --name inference_prob_mlc python=3.10
conda activate inference_prob_mlc
pip install -r requirements.txt
python inference_evaluate_models.py
# OR
make eval

```

I still got issues with installing numpy < 1.20 to be compatible with scikit-multiflow due to MacM1 incompatible with older version of numpy.

## License

This script is provided under the [MIT License](LICENSE).

---
