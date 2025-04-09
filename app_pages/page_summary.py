import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n"
        f"* Powdery mildew is a fungal disease that affects a wide range of plants, including cherry trees.\n"
        f"* Currently, manual inspection of each tree takes around 30 minutes, with an additional minute for treatment if mildew is found.\n"
        f"* This approach is not scalable across thousands of trees on multiple farms. Early and efficient detection is critical to protect crop yield and quality.\n\n"
        f"**Project Dataset**\n"
        f"* The dataset contains over 4,000 images of healthy cherry leaves and leaves affected by powdery mildew.\n"
        f"* Images were collected directly from cherry plantations owned by the client (Farmy & Foods).\n"
    )

    st.write(
        f"* For additional information, please read the full "
        f"[Project README](https://github.com/JLG-96/milestone-project-mildew-detection-in-cherry-leaves/blob/main/README.md)."
    )

    st.success(
        f"The project addresses 2 key business requirements:\n"
        f"* 1 - The client wants to visually differentiate healthy leaves from those with mildew.\n"
        f"* 2 - The client needs a prediction tool to instantly classify new leaf images as healthy or infected."
    )