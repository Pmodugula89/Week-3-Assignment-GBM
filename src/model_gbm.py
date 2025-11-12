def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
        ConfusionMatrixDisplay, RocCurveDisplay
    )
    import matplotlib.pyplot as plt
    import os

    # Create figures folder in project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_path = os.path.join(project_root, "figures")
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Print metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # Save confusion matrix
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(os.path.join(figures_path, "confusion_matrix.png"))

    # Save ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.savefig(os.path.join(figures_path, "roc_curve.png"))

    # Feature importance
    importances = model.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop Features:")
    for name, score in top_features:
        print(f"{name}: {score:.4f}")