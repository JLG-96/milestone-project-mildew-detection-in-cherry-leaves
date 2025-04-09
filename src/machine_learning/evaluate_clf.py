import joblib
import os


def load_test_evaluation(version):
    """
    Load test performance results from disk
    """
    model_path = os.path.join("outputs", version, "test_evaluation", "test_evaluation.pkl")
    return joblib.load(model_path)
