# app.py
import streamlit as st
from app_pages.page_summary import page_summary_body
from app_pages.page_cherry_leaves_visualizer import page_cherry_leaves_visualizer_body
from app_pages.page_mildew_detector import page_mildew_detector_body
from app_pages.page_ml_performance import page_ml_performance_metrics

# Sidebar navigation
st.sidebar.title("Cherry Leaf Mildew Detection")
page = st.sidebar.radio("Go to", ["Project Overview", "Visual Study", "Make a Prediction", "Model Evaluation"])

# Route pages
if page == "Project Overview":
    page_summary_body()
elif page == "Visual Study":
    page_cherry_leaves_visualizer_body()
elif page == "Make a Prediction":
    page_mildew_detector_body()
elif page == "Model Evaluation":
    page_ml_performance_metrics()

