import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

def resize_input_image(img, version, target_size=(256, 256)):
    """Resize uploaded image for model input"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    resized_img = img.resize(target_size)
    return resized_img

def load_model_and_predict(img, version):
    """Load the model and run prediction"""
    model_path = f"outputs/{version}/model.h5"
    model = tf.keras.models.load_model(model_path)

    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred_proba = model.predict(img_array)
    pred_class = np.argmax(pred_proba, axis=1)
    
    return pred_proba, pred_class[0]

def plot_predictions_probabilities(pred_proba, pred_class):
    """Display predicted probabilities in a bar chart"""
    class_names = ['healthy', 'powdery_mildew']
    fig, ax = plt.subplots()
    bars = ax.bar(class_names, pred_proba[0], color='skyblue')
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    bars[pred_class].set_color('green')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()
