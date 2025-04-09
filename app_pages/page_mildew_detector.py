import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def page_mildew_detector_body():
    st.info(
        f"* The client wants to predict whether a given cherry leaf is healthy or infected with powdery mildew."
    )

    st.write(
        f"* Upload an image of a cherry leaf (JPEG/PNG). The model will classify it as **Healthy** or **Powdery Mildew**."
    )

    st.write("---")

    # Upload one or more images
    uploaded_images = st.file_uploader(
        "Upload leaf images for prediction", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_images:
        model = load_mildew_model()  # Loads from outputs/v1/...

        results = []
        for img_file in uploaded_images:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption=f"Uploaded Image: {img_file.name}", use_column_width=True)

            # Preprocess image
            resized = img.resize((256, 256))
            img_array = np.expand_dims(np.array(resized) / 255.0, axis=0)

            # Predict
            prediction = model.predict(img_array)[0][0]
            pred_label = "Powdery Mildew" if prediction > 0.5 else "Healthy"
            confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)

            st.success(f"Prediction: **{pred_label}** ({confidence}% confidence)")
            st.write("---")

            results.append({"Filename": img_file.name, "Prediction": pred_label, "Confidence (%)": confidence})

        # Display all results
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
