# Hyperparameter Tuning

This README covers the fundamentals of hyperparameter tuning, various techniques, code examples, and best practices. Hyperparameter tuning is essential for optimizing machine learning model performance, and this guide provides the tools and understanding needed to implement it effectively.

## Table of Contents
1. [Introduction](#introduction)
2. [What are Hyperparameters?](#what-are-hyperparameters)
3. [Why Tune Hyperparameters?](#why-tune-hyperparameters)
4. [Hyperparameter Tuning Techniques](#hyperparameter-tuning-techniques)
   - Grid Search
   - Random Search
   - Bayesian Optimization
   - Cross-Validation
5. [Implementation Examples](#implementation-examples)
6. [Best Practices for Hyperparameter Tuning](#best-practices-for-hyperparameter-tuning)
7. [References](#references)

## Introduction
Hyperparameter tuning is the process of optimizing the settings that control the training process of machine learning models. Proper tuning helps improve model accuracy, stability, and overall performance. This guide explores popular hyperparameter tuning techniques, including Grid Search, Random Search, and Bayesian Optimization.

## What are Hyperparameters?
Hyperparameters are values set before training a model and are not learned from the data. They influence the training process, model complexity, and performance. Examples of hyperparameters include:
- **Learning Rate**: Determines the step size at each iteration during optimization.
- **Number of Trees**: For ensemble models like Random Forest or Gradient Boosting.
- **Regularization Parameters**: Such as L1 or L2 regularization, which control model complexity.
- **Batch Size and Epochs**: For neural networks, impacting training duration and stability.

## Why Tune Hyperparameters?
Optimizing hyperparameters is crucial for:
- Improving model accuracy.
- Reducing overfitting or underfitting.
- Enhancing generalization to new data.
- Saving computational resources by choosing efficient settings.

## Hyperparameter Tuning Techniques

### 1. Grid Search
Grid Search is an exhaustive search that tests all combinations of hyperparameters in a predefined grid. This method is thorough but can be computationally expensive, especially with large parameter spaces.

**Example**:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 2. Random Search
Random Search randomly selects combinations of hyperparameters from a predefined range.
It’s faster than Grid Search and can yield good results, particularly when the parameter space is large or the model is complex.

**Example**:
```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11)
}
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_dist, cv=5, n_iter=10)
random_search.fit(X_train, y_train)
```

### 3. Bayesian Optimization
Bayesian Optimization is a probabilistic model-based optimization method, often implemented with libraries like Hyperopt or Optuna. 
It selects the next set of hyperparameters based on the performance of previous sets, making it more efficient for large and complex parameter spaces.

**Example**
```python
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def objective(params):
    clf = RandomForestClassifier(**params)
    return -cross_val_score(clf, X_train, y_train, cv=5).mean()

space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
    'max_depth': hp.choice('max_depth', [None, 10, 20, 30]),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)
}

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)
```
### 4. Cross-Validation
Cross-validation (typically K-Fold) is often paired with hyperparameter tuning to ensure that the model’s performance generalizes well across different subsets of the data. This technique helps mitigate overfitting by training and validating the model on multiple data splits.

## Implementation Examples
Will include jupyter notebooks with these example, right now I have just added the datacamp solutions:

- Hyperparameter tuning for popular classifiers like SVM, Random Forest, and XGBoost.
- Comparison of Grid Search, Random Search, and Bayesian Optimization.
- Models evaluated with cross-validation.

## Best Practices for Hyperparameter Tuning
- Start with Random Search: For high-dimensional parameter spaces, begin with Random Search to identify promising ranges before fine-tuning with Grid Search.
- Use Cross-Validation: Combining cross-validation with hyperparameter tuning provides more reliable performance estimates.
- Monitor for Overfitting: Regularly check training vs. validation scores to detect overfitting, especially when tuning complex models.
- Leverage Available Tools: Libraries like Hyperopt, Optuna, and scikit-optimize streamline hyperparameter optimization.

## References
- Hyperparameter Tuning - scikit-learn Documentation
- Hyperopt - Hyperparameter Optimization
- Optuna - A Hyperparameter Optimization Framework
