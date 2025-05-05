import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v1'

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    try:
        labels_distribution = imread(
            f"outputs/{version}/labels_distribution.png")
        st.image(
            labels_distribution,
            caption='Labels Distribution on Train, Validation and Test Sets')
    except FileNotFoundError:
        st.warning("Label distribution plot not found.")
    st.write("---")

    st.write("### Model History")

    col1, col2 = st.columns(2)

    with col1:
        try:
            model_acc = imread(f"outputs/{version}/model_training_acc.png")
            st.image(model_acc, caption='Model Training Accuracy')
        except FileNotFoundError:
            st.warning("Training accuracy plot not found.")

    with col2:
        try:
            model_loss = imread(f"outputs/{version}/model_training_losses.png")
            st.image(model_loss, caption='Model Training Losses')
        except FileNotFoundError:
            st.warning("Training loss plot not found.")

    st.write("---")

    st.write("### Generalised Performance on Test Set")

    try:
        test_metrics = load_test_evaluation(version)
        st.dataframe(pd.DataFrame(test_metrics, index=["Loss", "Accuracy"]))
    except Exception as e:
        st.error(f"Unable to load test evaluation results: {e}")
