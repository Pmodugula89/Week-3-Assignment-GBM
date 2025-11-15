import os

# Create 'figures' folder in the project root (not inside src)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_path = os.path.join(project_root, "figures")

if not os.path.exists(figures_path):
    os.makedirs(figures_path)

from sklearn.datasets import load_breast_cancer
from preprocessing import get_preprocessed_data
from model_gbm import train_and_evaluate_model

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Preprocess
X_train, X_test, y_train, y_test = get_preprocessed_data(X, y)

# Train and evaluate
train_and_evaluate_model(X_train, X_test, y_train, y_test, data.feature_names)
