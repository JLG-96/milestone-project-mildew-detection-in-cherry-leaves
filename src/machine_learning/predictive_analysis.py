import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st


def resize_input_image(img, version, target_size=(256, 256)):
    """
    Resize input image to match model's input size.
    Returns a NumPy array (tensor) suitable for model prediction.
    """
    img = img.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize
    return img_array


def load_model_and_predict(img_tensor, version):
    """
    Loads the trained model and makes prediction on the input image tensor.
    Returns:
    - pred_proba: the predicted probability of the image being 'powdery_mildew'
    - pred_class_name: predicted class as string
    """
    model_path = f"outputs/{version}/mildew_detector_model.h5"
    model = tf.keras.models.load_model(model_path)

    prediction = model.predict(img_tensor)
    pred_proba = prediction[0][0]  # Single probability (scalar)

    if pred_proba > 0.5:
        pred_class_name = "powdery_mildew"
    else:
        pred_class_name = "healthy"

    return pred_proba, pred_class_name



def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plots class prediction probabilities for binary classifier with single output neuron.
    """
    class_names = ['healthy', 'powdery_mildew']
    colors = ['green', 'red']

    # Convert scalar probability to 2-class list
    if isinstance(pred_proba, (float, np.floating, np.ndarray)):
        pred_proba = float(pred_proba)  # make sure it's scalar
        probs = [1 - pred_proba, pred_proba]  # [healthy, mildew]
    else:
        probs = pred_proba  # fallback, shouldn't be needed

    fig, ax = plt.subplots()
    bars = ax.bar(class_names, probs, color=colors)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Prediction Probability')
    ax.set_title('Class Probability')

    for bar, label in zip(bars, class_names):
        if label == pred_class:
            bar.set_edgecolor('black')
            bar.set_linewidth(3)

    st.pyplot(fig)

