import os
from tensorflow.keras.models import load_model

def load_mildew_model(version="v1"):
    """
    Loads the trained mildew detector model (.h5) from the outputs directory.
    """
    model_path = os.path.join("outputs", version, "mildew_detector_model.h5")
    model = load_model(model_path)
    return model
