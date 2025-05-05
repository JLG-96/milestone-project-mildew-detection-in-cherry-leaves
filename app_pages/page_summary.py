import streamlit as st


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        "**General Information**\n\n"
        "* Powdery mildew is a fungal disease that affects many plant "
        "species, including cherry trees.\n"
        "* Currently, Farmy & Foods inspects cherry trees manually — this "
        "takes around 30 minutes per tree, plus 1 minute for "
        "treatment if mildew is detected.\n"
        "* With thousands of trees across multiple farms, this manual process "
        "is not scalable.\n"
        "* Early, automated detection is crucial to maintaining crop "
        "health and quality."
    )

    st.info(
        "**Project Dataset**\n\n"
        "* The dataset contains over 4,000 images of cherry leaves — both "
        "healthy and affected by powdery mildew.\n"
        "* All images were collected from Farmy & Foods cherry plantations.\n"
        "* Each image is labelled to support supervised learning using "
        "convolutional neural networks (CNNs)."
    )

    st.write(
        "* For more details, please refer to the "
        "[full project README]"
        "("
        "https://github.com/JLG-96/"
        "milestone-project-mildew-detection-in-cherry-leaves/"
        "blob/main/README.md"
        ")."
    )

    st.success(
        "The project addresses two key business requirements:\n"
        "* 1 - Enable visual study of healthy vs infected cherry leaves.\n"
        "* 2 - Provide a prediction tool to instantly classify new images as "
        "healthy or infected."
    )
