# CST600 Week 03 – GBM Model for Disease Diagnosis

This project applies **Gradient Boosting Machines (GBM)** using `scikit-learn` to predict breast cancer diagnosis. It demonstrates a full machine learning pipeline including data loading, preprocessing, model training, evaluation, and responsible feature interpretation.

---

## 📁 Project Structure
cst600-week03-gbm-pavan/ ├── .venv/                  # Virtual environment (excluded via .gitignore) ├── figures/                # Saved plots: confusion matrix & ROC curve ├── src/                    # Source code │   ├── main.py             # Entry point │   ├── model_gbm.py        # Model training and evaluation │   ├── preprocessing.py    # Train/test split ├── .gitignore              # Excludes .venv/, pycache/ ├── README.md               # Project documentation ├── requirements.txt        # Python dependencies

---

## 📊 Dataset

We use the built-in `load_breast_cancer()` dataset from `scikit-learn`, which contains:
- **569 samples**
- **30 numeric features** (e.g., radius, texture, perimeter)
- **Target:** Binary classification (malignant vs benign)

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Manishakittu/cst600-week03-gbm-pavan.git
cd cst600-week03-gbm-pavan
**2. Create and activate virtual environment
**
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # For PowerShell

**3. Install dependencies
**pip install -r requirements.txt

**4. Run the model

**python src/main.py

Outputs
After running the script, you’ll see:
📌 Printed Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Top 5 most important features


Saved Plots
- figures/confusion_matrix.png
- figures/roc_curve.png

📈 Model Details
We use GradientBoostingClassifier with:
- n_estimators=100
- learning_rate=0.1
- max_depth=3
- subsample=0.8
- random_state=42

Evaluation is done using:
- Stratified train/test split (80/20)
- Confusion matrix
- ROC curve
- Feature importance ranking

🧠 Responsible Modeling
- Stratification ensures balanced class distribution
- Feature importance helps interpret model decisions
- No overfitting observed — metrics are high but realistic
- Ethical consideration: Model is for educational use only, not clinical deployment


Next Steps
- Extend to real-world medical datasets
- Add cross-validation and hyperparameter tuning
- Explore SHAP or LIME for deeper interpretability


Author
Pavan
CST600 – Week 03 Assignment

📜 License
This project is for academic purposes under the CST600 module. No commercial or clinical use permitted.



