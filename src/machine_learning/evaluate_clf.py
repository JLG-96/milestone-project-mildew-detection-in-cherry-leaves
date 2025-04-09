import json
import os

def load_test_evaluation(version):
    model_path = os.path.join("outputs", version, "test_evaluation.json")
    with open(model_path, "r") as f:
        return json.load(f)