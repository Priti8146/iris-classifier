# Iris Classifier (Decision Tree)

## Overview
End-to-end ML example from the AI Fundamentals course: builds a decision-tree classifier on the classic Iris dataset with scikit-learn.

**Result:** 93.3% accuracy on 30 unseen flowers (28 correct, 2 wrong).

## Quick start
```bash
git clone https://github.com/Priti8146/iris-classifier.git
cd iris-classifier
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
python src/train.py --test-size 0.2 --random-state 42
```
Prints accuracy and saves `outputs/confusion_matrix.png` and `outputs/model.joblib`.

## Project structure
```
iris-classifier/
├── data/              # empty – Iris is loaded from scikit-learn
├── notebooks/
│   └── iris_model.ipynb
├── src/
│   └── train.py
├── tests/
│   └── test_train.py
├── outputs/           # created automatically
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Tests
```bash
pytest
```

## License
MIT – see `LICENSE`.