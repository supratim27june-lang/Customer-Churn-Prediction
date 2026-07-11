# Customer Churn Prediction

Predicting customer churn for a telecom company using an Artificial Neural Network (ANN). The model learns from historical subscriber data to flag customers who are likely to leave, achieving approximately **79% accuracy** on the test set.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## Overview

Customer churn — when a subscriber stops doing business with a company — is one of the most expensive problems in the telecom industry, since acquiring a new customer costs significantly more than retaining an existing one. This project builds a deep learning classifier that predicts, from a customer's account and usage profile, whether that customer is likely to churn. Armed with these predictions, a business can target at-risk customers with retention offers before they leave.

The entire workflow — from data cleaning and exploratory analysis through model training and evaluation — is contained in a single, reproducible Jupyter notebook.

## Problem Statement

Given a set of customer attributes (demographics, subscribed services, contract type, billing information, and tenure), predict a binary outcome:

- **`Churn = Yes`** — the customer left the company
- **`Churn = No`** — the customer stayed

This is a **binary classification** task, tackled here with a feed-forward neural network.

## Dataset

The project uses the well-known **IBM Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), included in this repository.

| Property | Detail |
|---|---|
| Records | ~7,043 customers |
| Features | 20 predictor columns + 1 target (`Churn`) |
| Target | `Churn` (Yes / No) |
| Type | Tabular, mixed categorical & numerical |

**Feature groups include:**

- **Demographics** — gender, senior citizen status, partner, dependents
- **Account information** — tenure, contract type, paperless billing, payment method, monthly charges, total charges
- **Subscribed services** — phone service, multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV, streaming movies

> The dataset is moderately imbalanced (roughly 1 in 4 customers churn), which is worth keeping in mind when interpreting accuracy.

## Approach

The notebook follows a standard, end-to-end machine learning pipeline:

1. **Data loading & inspection** — read the CSV and review structure, types, and summary statistics.
2. **Data cleaning** — handle the `TotalCharges` column (which contains blank/whitespace values), convert it to numeric, and drop non-predictive identifiers such as `customerID`.
3. **Exploratory data analysis (EDA)** — visualize churn distribution and how features such as contract type, tenure, and monthly charges relate to churn.
4. **Preprocessing** — encode categorical variables (label / one-hot encoding) and scale numerical features so all inputs are on a comparable range.
5. **Train/test split** — partition the data to evaluate the model on unseen customers.
6. **Model building** — construct and compile an ANN using Keras.
7. **Training** — fit the network over multiple epochs, monitoring loss and accuracy.
8. **Evaluation** — assess performance on the test set using accuracy and a classification report / confusion matrix.

## Model Architecture

A feed-forward Artificial Neural Network built with the Keras Sequential API:

- **Input layer** sized to the number of processed features
- **Hidden layers** with ReLU activation
- **Output layer** with a single sigmoid unit for binary probability
- **Loss:** binary cross-entropy
- **Optimizer:** Adam
- **Metric:** accuracy

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **~79%** |

The model correctly classifies roughly four out of five customers. Because the classes are imbalanced, reviewing the confusion matrix and per-class precision/recall (included in the notebook) gives a fuller picture of performance than accuracy alone.

## Tech Stack

- **Python 3**
- **TensorFlow / Keras** — building and training the ANN
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — visualization
- **scikit-learn** — preprocessing, splitting, and evaluation metrics
- **Jupyter Notebook** — interactive development environment

## Project Structure

```
Customer-Churn-Prediction/
├── CustomerChurn.ipynb                    # Main notebook: EDA, preprocessing, model, evaluation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Telco customer churn dataset
├── LICENSE                                # MIT License
└── README.md                              # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` (or `conda`)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/supratim27june-lang/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **(Optional) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
   ```

## Usage

Launch Jupyter and open the notebook:

```bash
jupyter notebook CustomerChurn.ipynb
```

Run the cells from top to bottom to reproduce the full pipeline — data cleaning, EDA, preprocessing, model training, and evaluation. The dataset is already included, so no additional download is required.

## Future Improvements

- Address class imbalance with techniques such as SMOTE, class weights, or threshold tuning
- Benchmark the ANN against classical models (Logistic Regression, Random Forest, XGBoost)
- Add hyperparameter tuning (learning rate, layer sizes, dropout) via grid/random search
- Report ROC-AUC and F1-score alongside accuracy for a more balanced evaluation
- Add regularization (dropout, early stopping) to reduce overfitting
- Deploy the trained model as a REST API or Streamlit app for interactive predictions

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## Author

**Supratim**
GitHub: [@supratim27june-lang](https://github.com/supratim27june-lang)

---

⭐ If you found this project useful, consider giving it a star!
