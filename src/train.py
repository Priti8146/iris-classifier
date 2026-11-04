import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def train_and_evaluate(test_size=0.2, random_state=42):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size,
        random_state=random_state, stratify=iris.target)

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return model, acc, cm, iris.target_names


def main():
    parser = argparse.ArgumentParser(description="Train a decision tree on Iris.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    model, acc, cm, labels = train_and_evaluate(args.test_size, args.random_state)
    print(f"Accuracy: {acc:.3f}")

    OUTPUTS.mkdir(exist_ok=True)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot()
    plt.savefig(OUTPUTS / "confusion_matrix.png", bbox_inches="tight")
    joblib.dump(model, OUTPUTS / "model.joblib")
    print(f"Saved figure and model to {OUTPUTS}")


if __name__ == "__main__":
    main()