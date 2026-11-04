import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train import train_and_evaluate


def test_accuracy_at_least_90_percent():
    _, acc, _, _ = train_and_evaluate(test_size=0.2, random_state=42)
    assert acc >= 0.9