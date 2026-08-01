# Customer Churn Prediction using Artificial Neural Networks (ANN)

This repository contains a Deep Learning project built with Keras and TensorFlow to predict customer churn (`Exited`) for a banking institution using tabular demographic and financial data.

---

## Project Overview

Customer retention is critical for subscription-based and banking services. This project implements an Artificial Neural Network (ANN) binary classifier to identify high-risk customers likely to churn. The project covers the end-to-end Machine Learning pipeline:
* Data Ingestion & Exploration
* Feature Selection & One-Hot Encoding
* Feature Scaling
* Neural Network Architecture Construction
* Model Training and Validation

---

## Tech Stack & Dependencies

* **Python 3.x**
* **Pandas**: Data manipulation and cleaning
* **NumPy**: Numerical computations
* **Matplotlib & Seaborn**: Data visualization
* **Scikit-Learn**: Dataset splitting (`train_test_split`) and feature scaling (`StandardScaler`)
* **TensorFlow / Keras**: Sequential neural network construction and training

---

## Dataset & Features

The dataset (`Churn_Modelling.csv`) contains 10,000 records of customer attributes:

* **Identifiers (Dropped during preprocessing)**:
  * `RowNumber`
  * `CustomerId`
  * `Surname`
* **Features**:
  * `CreditScore`: Customer's credit score
  * `Geography`: Customer country (`France`, `Germany`, `Spain`)
  * `Gender`: `Male` or `Female`
  * `Age`: Age of the customer
  * `Tenure`: Number of years the customer has been with the bank
  * `Balance`: Account balance
  * `NumOfProducts`: Number of bank products used
  * `HasCrCard`: Credit card status (0 = No, 1 = Yes)
  * `IsActiveMember`: Active status (0 = No, 1 = Yes)
  * `EstimatedSalary`: Estimated annual salary
* **Target Variable**:
  * `Exited`: Customer churn status (0 = Retained, 1 = Exited)

---

## Preprocessing Pipeline

1. **Identifier Removal**: Non-informative columns (`RowNumber`, `CustomerId`, `Surname`) are removed.
2. **Categorical Encoding**: One-Hot Encoding is applied to categorical columns (`Geography` and `Gender`), utilizing `drop_first=True` to prevent dummy variable multicollinearity.
3. **Train-Test Split**: Data is split into an 80% training set and a 20% test set (`test_size=0.2`, `random_state=1`).
4. **Feature Scaling**: Numerical features are standardized using Scikit-Learn's `StandardScaler` to ensure optimal gradient descent convergence.

---

## Model Architecture

The deep learning model is built using Keras `Sequential`:

* **Input Layer**: 11 features (`input_dim=11`)
* **Hidden Layer 1**: Fully Connected (`Dense`), 11 neurons, `ReLU` activation
* **Hidden Layer 2**: Fully Connected (`Dense`), 11 neurons, `ReLU` activation
* **Output Layer**: Fully Connected (`Dense`), 1 neuron, `Sigmoid` activation for binary classification

### Model Summary
* Total Parameters: 276
* Loss Function: Binary Cross-Entropy (`binary_crossentropy`)
* Optimizer: Adam (`Adam`)
* Evaluation Metrics: Accuracy (`accuracy`)

---

## Model Training & Performance

* **Batch Size / Validation**: Trained over 100 epochs with validation split monitoring (`val_loss`, `val_accuracy`).
* **Performance Overview**:
  * Training Accuracy: ~86.8%
  * Validation Accuracy: ~84.5% - 85.0%

---

## Install Required Dependencies
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
  ```

## How to Run

1. Clone this repository:
```bash
git clone https://github.com/rehan-rafique/Customer_Churn_Prediction.git
