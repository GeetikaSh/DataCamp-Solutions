# Logistic Regression

Welcome to the Logistic Regression project! This repository provides an in-depth overview of logistic regression, a fundamental statistical technique used for binary classification problems. Here, you'll find an implementation of logistic regression, explanations, and insights into its application for data science and machine learning. I have added DataCamp Solution for Logistic Regression, and will continue to attach my Kaggle projects where I will use logistic regression.

## Table of Contents
1. [Introduction](#introduction)
2. [Concepts](#concepts)
3. [Implementation](#implementation)
5. [Model Training](#model-training)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Example Workflow](#example-workflow)

## Introduction
Logistic regression is a type of regression analysis often used for binary classification tasks. It estimates the probability of a binary response based on one or more predictor variables, making it useful for applications such as fraud detection, medical diagnoses, and marketing.


## Concepts
- **Binary Classification**: Logistic regression is used when the target variable has two possible outcomes (0 and 1).
- **Sigmoid Function**: The logistic function (sigmoid) is used to squeeze the output of the linear equation between 0 and 1, representing probabilities.
- **Cost Function**: Cross-entropy loss is used to evaluate the model's predictions against actual labels.

### Mathematical Representation

The probability that the output is 1 given an input `x` is given by:

    P(y=1 | x) = 1 / (1 + e^-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ))

where:

- `P(y=1 | x)`: The probability that the target `y` is 1, given input `x`.
- `e`: The base of the natural logarithm.
- `β₀`: The intercept or bias term.
- `β₁, β₂, ..., βₙ`: Coefficients corresponding to each feature `x₁, x₂, ..., xₙ` in the dataset.

This logistic or "sigmoid" function maps the input to a value between 0 and 1, representing the predicted probability of the positive class (e.g., 1).


## Implementation
The project includes:
1. **Data Preprocessing**: Handling missing values, encoding categorical data, and feature scaling.
2. **Model Training**: Training logistic regression.
3. **Evaluation**: Using accuracy, precision, recall, and F1-score as evaluation metrics.

## Model Training
1. **Train-Test Split**: The data is divided into training and testing sets.
2. **Training**: Logistic regression is trained using gradient descent or by using library-based solvers.
3. **Hyperparameter Tuning**: Regularization parameters can be tuned for optimal performance.

## Evaluation Metrics
- **Accuracy**: Proportion of correctly classified samples.
- **Precision**: Correct positive predictions divided by total positive predictions.
- **Recall**: Correct positive predictions divided by actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

## Example Workflow

Here’s an example workflow to illustrate a typical use case:

```python
# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split your dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using the scikit-learn model
from sklearn.linear_model import LogisticRegression
sklearn_model = LogisticRegression()
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)
print("Scikit-learn Model Accuracy:", accuracy_score(y_test, sklearn_predictions))
