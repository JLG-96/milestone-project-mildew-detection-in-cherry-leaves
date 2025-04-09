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


def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plots class prediction probabilities.
    """
    class_names = ['healthy', 'powdery_mildew']
    colors = ['green', 'red']
    probs = pred_proba[0] if hasattr(pred_proba, '__len__') else [pred_proba]

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
