# SGD Regressor from Scratch

This project implements **Stochastic Gradient Descent (SGD) for Linear Regression from scratch using NumPy**.  
The goal is to understand how gradient-based optimization works internally instead of relying on prebuilt machine learning models.

The implementation includes manual training of model parameters and evaluation using common regression metrics.

---

## Overview

Stochastic Gradient Descent is an optimization algorithm used to minimize a loss function by updating model parameters iteratively using **one randomly selected training example at a time**.

Compared to standard Gradient Descent:

| Method | Update Strategy |
|------|------|
| Batch Gradient Descent | Uses entire dataset |
| Stochastic Gradient Descent | Uses one random sample |
| Mini-Batch Gradient Descent | Uses small batches |

SGD is computationally faster for large datasets and introduces noise that can help escape local minima.

---

## Implementation Details

The model is implemented in a custom class:

```python
class SGDRegressor:
```

### Parameters

| Parameter | Description |
|---|---|
| `epochs` | Number of passes through the dataset |
| `learning_rate` | Step size for gradient updates |

---

## Training Procedure

1. Initialize weights and bias.
2. For each epoch:
   - Randomly select a training example.
   - Compute prediction.
   - Compute gradients of loss with respect to parameters.
   - Update weights and bias using the learning rate.
3. Repeat for the specified number of epochs.

---

## Prediction

Predictions are computed using the linear regression equation:

\[
\hat{y} = Xw + b
\]

Where:

- \(X\) = feature matrix  
- \(w\) = model weights  
- \(b\) = bias term  

---

## Evaluation Metrics

The model performance is evaluated using several regression metrics.

### Mean Absolute Error (MAE)

Average absolute difference between predicted and actual values.

\[
MAE = \frac{1}{n}\sum |y - \hat{y}|
\]

---

### Mean Squared Error (MSE)

Average of squared prediction errors.

\[
MSE = \frac{1}{n}\sum (y - \hat{y})^2
\]

---

### Root Mean Squared Error (RMSE)

Square root of the Mean Squared Error.

\[
RMSE = \sqrt{MSE}
\]

---

### R² Score

Measures how well the model explains the variance in the target variable.

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

---

### Adjusted R² Score

Adjusted version of R² that accounts for the number of features.

\[
Adjusted\ R^2 =
1 - \frac{(1-R^2)(n-1)}{n-k-1}
\]

Where:

- \(n\) = number of samples  
- \(k\) = number of features  

---

## Example Results

```
Mean Absolute Error: 48.43
Mean Squared Error: 3319.13
Root Mean Squared Error: 57.61
R2 Score: 0.4668
Adjusted R2 Score: 0.4512
```

---

## Key Learning Outcomes

- Understanding how **Stochastic Gradient Descent works internally**
- Implementing **linear regression optimization manually**
- Computing **regression evaluation metrics**
- Learning the difference between **batch, stochastic, and mini-batch gradient descent**

---

## Technologies Used

- Python
- NumPy
- scikit-learn (for evaluation metrics)

---

## Author

**Kanisk Dasgupta**  
B.Tech in Artificial Intelligence & Machine Learning  
