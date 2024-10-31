import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Step 1: Generate Data
np.random.seed(42)
X = np.random.uniform(0, 1000, 300).reshape(-1, 1)  # Generate 300 random variables
Y = np.where((X > 500) & (X < 800), 1, 0).flatten()  # Create labels based on conditions

# Step 2: Train Models with Polynomial Features
def train_models(X, Y):
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Train models
    model_log_reg = LogisticRegression(max_iter=1000)
    model_svm = SVC(kernel='rbf', gamma='scale')  # Use RBF kernel for SVM

    model_log_reg.fit(X_poly, Y)
    model_svm.fit(X_poly, Y)

    y_pred_log_reg = model_log_reg.predict(X_poly)
    y_pred_svm = model_svm.predict(X_poly)

    return y_pred_log_reg, y_pred_svm, model_log_reg, model_svm

# Step 3: Streamlit Interface
st.title("Logistic Regression and SVM on Random Data")

# Button to train models
if st.button("Train Models"):
    y_pred_log_reg, y_pred_svm, model_log_reg, model_svm = train_models(X, Y)
    
    # Calculate accuracies
    accuracy_log_reg = accuracy_score(Y, y_pred_log_reg)
    accuracy_svm = accuracy_score(Y, y_pred_svm)
    
    st.write(f"Logistic Regression Accuracy: {accuracy_log_reg:.2f}")
    st.write(f"SVM Accuracy: {accuracy_svm:.2f}")

    # Step 4: Visualization
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    # Create a range of values for X for decision boundary
    x_range = np.linspace(0, 1000, 300).reshape(-1, 1)
    x_range_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x_range)

    # Plot for Logistic Regression
    ax[0].scatter(X, Y, label='Actual', color='blue', alpha=0.5)
    ax[0].scatter(X, y_pred_log_reg, label='Predicted', color='orange', alpha=0.5)

    # Decision boundary
    Z_log_reg = model_log_reg.predict(x_range_poly)
    ax[0].plot(x_range, Z_log_reg, color='green', label='Decision Boundary', linewidth=2)
    ax[0].set_title("Logistic Regression: Actual vs Predicted")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].legend()

    # Plot for SVM
    ax[1].scatter(X, Y, label='Actual', color='blue', alpha=0.5)
    ax[1].scatter(X, y_pred_svm, label='Predicted', color='orange', alpha=0.5)

    # Decision boundary
    Z_svm = model_svm.predict(x_range_poly)
    ax[1].plot(x_range, Z_svm, color='green', label='Decision Boundary', linewidth=2)
    ax[1].set_title("SVM: Actual vs Predicted")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].legend()

    # Show plots
    st.pyplot(fig)
