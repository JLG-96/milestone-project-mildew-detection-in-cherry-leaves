# src/load_model.py
import tensorflow as tf
import os

@st.cache_resource
def load_mildew_model(model_path='outputs/v1/mildew_detector_model.h5'):
    """
    Load and cache the mildew detection model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at path: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model
