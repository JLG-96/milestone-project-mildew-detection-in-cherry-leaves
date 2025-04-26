import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.model_loader import load_mildew_model
from src.machine_learning.predictive_analysis import (
    resize_input_image,
    plot_predictions_probabilities
)


def page_mildew_detector_body():
    st.info(
        "* The client wants to predict whether a given cherry leaf is healthy or infected with powdery mildew."
    )

    st.info(
    "* Need a sample image to test? You can download healthy or infected cherry leaf images from [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write(
        "* Upload a cherry leaf image. The model will classify it as **Healthy** or **Powdery Mildew** based on visual features."
    )

    st.write("---")

    uploaded_images = st.file_uploader(
        label="Upload cherry leaf image(s). You may select more than one.",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    if uploaded_images:
        version = "v1"
        model = load_mildew_model(version)
        df_report = pd.DataFrame([])

        for image in uploaded_images:
            img_pil = Image.open(image)
            st.info(f"Image uploaded: **{image.name}**")
            st.image(img_pil, caption=f"{image.name}", use_column_width=True)

            # Preprocess and predict
            img_tensor = resize_input_image(img=img_pil, version=version)
            pred_proba = model.predict(img_tensor)
            pred_class = "Powdery Mildew" if pred_proba[0][0] > 0.5 else "Healthy"

            # Plot results
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report._append(
                {"Filename": image.name, "Prediction": pred_class,"Confidence (%)": round(float(np.squeeze(pred_proba)) * 100, 2)},
                ignore_index=True
            )

        # Show prediction table
        if not df_report.empty:
            st.success("Prediction Summary")
            st.dataframe(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)
