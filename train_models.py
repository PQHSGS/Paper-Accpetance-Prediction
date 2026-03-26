import os
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import classification_report, accuracy_score

from Models import (
    LogisticRegression,
    RandomForestClassifier,
    SVMClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    EnsembleClassifier
)

def load_data(data_dir, max_vocab="False", encoder="w2v", hand="True"):
    train_file = os.path.join(data_dir, "train", "dataset", f"features.svmlite_{max_vocab}_{encoder}_{hand}.txt")
    dev_file = os.path.join(data_dir, "dev", "dataset", f"features.svmlite_{max_vocab}_{encoder}_{hand}.txt")
    test_file = os.path.join(data_dir, "test", "dataset", f"features.svmlite_{max_vocab}_{encoder}_{hand}.txt")

    print(f"Loading training data from {train_file}...")
    X_train_sp, y_train = load_svmlight_file(train_file)
    n_features = X_train_sp.shape[1]

    print(f"Loading dev/test data...")
    X_dev_sp, y_dev = load_svmlight_file(dev_file, n_features=n_features)
    X_test_sp, y_test = load_svmlight_file(test_file, n_features=n_features)

    # Convert sparse matrices to dense arrays for our custom Models/
    X_train = X_train_sp.toarray()
    X_dev = X_dev_sp.toarray()
    X_test = X_test_sp.toarray()

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)

def main():
    data_dir = os.path.join("Dataset", "all_combined")
    
    # Load dataset
    try:
        (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = load_data(data_dir)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print(f"\nDataset loaded. Train shape: {X_train.shape}, Dev shape: {X_dev.shape}, Test shape: {X_test.shape}")
    print("-" * 50)

    # Initialize a few custom models to compare
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=10, max_depth=5),
        "SVM (Pegasos)": SVMClassifier(kernel="linear"),
        "AdaBoost": AdaBoostClassifier(n_estimators=10),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=10)
    }

    # Add the EnsembleClassifier using the first three base models (hard voting)
    models["Ensemble Model"] = EnsembleClassifier(
        models=[
            LogisticRegression(),
            RandomForestClassifier(n_estimators=10, max_depth=5),
            SVMClassifier(kernel="linear")
        ],
        voting="hard"
    )

    # Train and evaluate
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            
            # Predict and evaluate on Dev Set
            preds_dev = model.predict(X_dev)
            acc_dev = accuracy_score(y_dev, preds_dev)
            
            # Predict and evaluate on Test Set
            preds_test = model.predict(X_test)
            acc_test = accuracy_score(y_test, preds_test)
            
            print(f"[{name}] Dev Accuracy: {acc_dev*100:.2f}% | Test Accuracy: {acc_test*100:.2f}%")
        except Exception as e:
            import traceback
            print(f"Error training/evaluating {name}:\n")
            traceback.print_exc()
        print("-" * 50)

if __name__ == "__main__":
    main()
