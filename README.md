# SGD Regressor from Scratch

This project implements **Stochastic Gradient Descent (SGD) for Linear Regression from scratch using NumPy**.  
The goal is to understand how gradient-based optimization works internally instead of relying on built-in machine learning models.

Ghe implementation manually trains model parameters and evaluates performance using common regression metrics.

---

## Overview

Stochastic Gradient Descent is an optimization algorithm used to minimize a loss function by updating model parameters iteratively using **one randomly selected training sample at a time**.

Comparison of gradient descent methods:

| Method | Update Strategy |
|------|------|
| Batch Gradient Descent | Uses the entire dataset for each update |
| Stochastic Gradient Descent | Uses one randomly selected sample |
| Mini-Batch Gradient Descent | Uses small batches of samples |

SGD is computationally efficient and often converges faster for large datasets.

---

## Implementation Details

The model is implemented using a custom class:

```python
class SGDRegressor:
```

### Parameters

| Parameter | Description |
|---|---|
| `epochs` | Number of training iterations over the dataset |
| `learning_rate` | Step size used for parameter updates |

---

## Training Procedure

1. Initialize weights and bias.
2. For each epoch:
   - Randomly select a training sample.
   - Compute the prediction.
   - Compute gradients of the loss function.
   - Update weights and bias using the learning rate.
3. Repeat for the specified number of epochs.

---

## Prediction

Predictions follow the linear regression equation:

y_hat = Xw + b

Where:

- X = feature matrix  
- w = weight vector  
- b = bias (intercept)  
- y_hat = predicted value  

---

# Evaluation Metrics

## Mean Absolute Error (MAE)

Average absolute difference between predicted and actual values.

MAE = (1/n) * Σ |y_i - y_hat_i|

---

## Mean Squared Error (MSE)

Average of the squared differences between predicted and actual values.

MSE = (1/n) * Σ (y_i - y_hat_i)^2

---

## Root Mean Squared Error (RMSE)

Square root of the Mean Squared Error.

RMSE = sqrt(MSE)

---

## R² Score

Measures how well the model explains the variance in the target variable.

R² = 1 - ( Σ (y_i - y_hat_i)^2 / Σ (y_i - y_mean)^2 )

Where:

- y_i = actual value  
- y_hat_i = predicted value  
- y_mean = mean of actual values  

---

## Adjusted R² Score

Adjusted R² accounts for the number of predictors used in the model.

Adjusted R² = 1 - ((1 - R²) * (n - 1) / (n - k - 1))

Where:

- n = number of samples  
- k = number of features  

---

## Example Output

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
- Implementing **linear regression optimization from scratch**
- Computing common **regression evaluation metrics**
- Learning the differences between **batch, stochastic, and mini-batch gradient descent**

---

## Technologies Used

- Python  
- NumPy  
- scikit-learn (for evaluation metrics)
