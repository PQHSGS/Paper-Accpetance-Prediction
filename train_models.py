import os
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

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


def standardize(X_train, X_dev, X_test):
    """
    Standardize using training-set statistics only.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_dev - mean) / std, (X_test - mean) / std


def metrics_dict(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def pick_threshold(y_true, proba, thresholds=None):
    """
    Pick threshold by maximizing F1 on dev.

    Uses all unique probability cut points so the chosen threshold is exact for
    this dev set, not just from a coarse grid.
    """
    if thresholds is None:
        uniq = np.unique(np.asarray(proba, dtype=float))
        # Midpoints between adjacent probability values cover all decision boundaries.
        if uniq.size == 0:
            thresholds = np.array([0.5], dtype=float)
        elif uniq.size == 1:
            thresholds = np.array([max(0.0, uniq[0] - 1e-6), min(1.0, uniq[0] + 1e-6)], dtype=float)
        else:
            mids = (uniq[:-1] + uniq[1:]) / 2.0
            thresholds = np.concatenate(([0.0], mids, [1.0]))

    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def train_eval_candidate(name, model, X_train, y_train, X_dev, y_dev, X_test, y_test):
    model.fit(X_train, y_train)
    dev_proba = model.predict_proba(X_dev)[:, 1]
    best_t, dev_best_f1 = pick_threshold(y_dev, dev_proba)

    dev_pred = (dev_proba >= best_t).astype(int)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_t).astype(int)

    dev_metrics = metrics_dict(y_dev, dev_pred)
    test_metrics = metrics_dict(y_test, test_pred)

    return {
        "name": name,
        "model": model,
        "threshold": best_t,
        "dev_best_f1": dev_best_f1,
        "dev": dev_metrics,
        "test": test_metrics,
    }


def print_metrics(prefix, metrics):
    print(
        f"{prefix} Acc: {metrics['accuracy']*100:.2f}% | "
        f"BalAcc: {metrics['balanced_accuracy']*100:.2f}% | "
        f"F1: {metrics['f1']*100:.2f}% | "
        f"Prec: {metrics['precision']*100:.2f}% | "
        f"Rec: {metrics['recall']*100:.2f}%"
    )

def main():
    data_dir = os.path.join("Dataset", "all_combined")
    
    # Load dataset
    try:
        (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = load_data(data_dir)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Ensure labels are integer in {0,1}
    y_train = y_train.astype(int)
    y_dev = y_dev.astype(int)
    y_test = y_test.astype(int)

    print(f"\nDataset loaded. Train shape: {X_train.shape}, Dev shape: {X_dev.shape}, Test shape: {X_test.shape}")
    train_pos_ratio = y_train.mean()
    print(f"Train positive ratio: {train_pos_ratio*100:.2f}%")

    majority_class = int(np.bincount(y_train).argmax())
    maj_dev = np.full_like(y_dev, majority_class)
    maj_test = np.full_like(y_test, majority_class)
    print("Majority baseline:")
    print_metrics("  Dev", metrics_dict(y_dev, maj_dev))
    print_metrics("  Test", metrics_dict(y_test, maj_test))
    print("-" * 50)

    # Scale for linear models (helps LR/SVM optimization).
    X_train_scaled, X_dev_scaled, X_test_scaled = standardize(X_train, X_dev, X_test)

    # Candidate configurations focused on strong families for this feature set.
    candidates = [
        ("LogReg L2 lambda=0.01", LogisticRegression(balance=True, penalty="l2", reg_lambda=0.01, lr=0.05, max_iter=2500, random_state=42), X_train_scaled, X_dev_scaled, X_test_scaled),
        ("LogReg L2 lambda=0.001", LogisticRegression(balance=True, penalty="l2", reg_lambda=0.001, lr=0.04, max_iter=2500, random_state=42), X_train_scaled, X_dev_scaled, X_test_scaled),
        ("SVM Linear C=1", SVMClassifier(balance=True, kernel="linear", C=1.0, max_iter_linear=6000, random_state=42), X_train_scaled, X_dev_scaled, X_test_scaled),
        ("SVM Linear C=2", SVMClassifier(balance=True, kernel="linear", C=2.0, max_iter_linear=7000, random_state=42), X_train_scaled, X_dev_scaled, X_test_scaled),
        ("RF n=300 depth=10", RandomForestClassifier(balance=True, n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42), X_train, X_dev, X_test),
        ("RF n=500 depth=12", RandomForestClassifier(balance=True, n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42), X_train, X_dev, X_test),
        ("Ada depth=1 n=400 lr=0.1", AdaBoostClassifier(balance=True, n_estimators=400, learning_rate=0.1, base_max_depth=1, random_state=42), X_train, X_dev, X_test),
        ("Ada depth=2 n=400 lr=0.15", AdaBoostClassifier(balance=True, n_estimators=400, learning_rate=0.15, base_max_depth=2, random_state=42), X_train, X_dev, X_test),
        ("Ada depth=3 n=300 lr=0.1", AdaBoostClassifier(balance=True, n_estimators=300, learning_rate=0.1, base_max_depth=3, random_state=42), X_train, X_dev, X_test),
        ("GB depth=2 n=500 lr=0.05", GradientBoostingClassifier(balance=True, n_estimators=500, learning_rate=0.05, max_depth=2, min_samples_leaf=3, random_state=42), X_train, X_dev, X_test),
        ("GB depth=3 n=500 lr=0.05", GradientBoostingClassifier(balance=True, n_estimators=500, learning_rate=0.05, max_depth=3, min_samples_leaf=3, random_state=42), X_train, X_dev, X_test),
        ("GB depth=3 n=700 lr=0.03", GradientBoostingClassifier(balance=True, n_estimators=700, learning_rate=0.03, max_depth=3, min_samples_leaf=3, random_state=42), X_train, X_dev, X_test),
    ]

    results = []
    model_dev_proba = {}
    model_test_proba = {}
    for name, model, Xt, Xd, Xte in candidates:
        print(f"Training {name}...")
        try:
            result = train_eval_candidate(name, model, Xt, y_train, Xd, y_dev, Xte, y_test)
            results.append(result)
            model_dev_proba[name] = model.predict_proba(Xd)[:, 1]
            model_test_proba[name] = model.predict_proba(Xte)[:, 1]
            print(f"  Threshold tuned on dev: {result['threshold']:.2f}")
            print_metrics("  Dev", result["dev"])
            print_metrics("  Test", result["test"])
        except Exception as e:
            import traceback
            print(f"Error training/evaluating {name}:\n")
            traceback.print_exc()
        print("-" * 50)

    if not results:
        print("No model finished successfully.")
        return

    # Rank by dev F1, then dev balanced accuracy.
    best = sorted(results, key=lambda r: (r["dev"]["f1"], r["dev"]["balanced_accuracy"]), reverse=True)[0]

    print("Best single model by dev metrics:")
    print(f"  {best['name']} | threshold={best['threshold']:.2f}")
    print_metrics("  Dev", best["dev"])
    print_metrics("  Test", best["test"])

    print("-" * 50)

    # Blend top-3 models by dev F1 using averaged probabilities.
    top3 = sorted(results, key=lambda r: (r["dev"]["f1"], r["dev"]["balanced_accuracy"]), reverse=True)[:3]
    top3_names = [r["name"] for r in top3]
    if len(top3_names) == 3:
        blend_dev = np.mean([model_dev_proba[n] for n in top3_names], axis=0)
        blend_test = np.mean([model_test_proba[n] for n in top3_names], axis=0)
        blend_t, _ = pick_threshold(y_dev, blend_dev)
        blend_dev_pred = (blend_dev >= blend_t).astype(int)
        blend_test_pred = (blend_test >= blend_t).astype(int)
        print(f"Top-3 Probability Blend: {top3_names}")
        print(f"  Threshold tuned on dev: {blend_t:.2f}")
        print_metrics("  Dev", metrics_dict(y_dev, blend_dev_pred))
        print_metrics("  Test", metrics_dict(y_test, blend_test_pred))

    print("-" * 50)

if __name__ == "__main__":
    main()
