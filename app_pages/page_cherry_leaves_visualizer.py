import streamlit as st
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random


def page_cherry_leaves_visualizer_body():
    st.write("### Cherry Leaf Visual Study")

    st.info(
        "* The client is interested in a study that visually differentiates "
        "healthy cherry leaves from those affected by powdery mildew."
    )

    version = 'v1'

    if st.checkbox("Average and variability images"):
        avg_healthy = imread(f"outputs/{version}/avg_var_healthy.png")
        avg_mildew = imread(f"outputs/{version}/avg_var_powdery_mildew.png")

        st.warning(
            "* The average and variability images may not show obvious"
            "patterns to the naked eye, but subtle differences in colour and"
            "texture are visible."
        )

        st.image(avg_healthy, caption='Healthy Leaf - Average and Variability')
        st.image(
            avg_mildew,
            caption='Powdery Mildew Leaf - Average and Variability')
        st.write("---")

    if st.checkbox("Difference between class averages"):
        diff_image = imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            "* This shows pixel-level differences between average healthy "
            "and mildew-affected leaves."
        )
        st.image(
            diff_image,
            caption='Difference Between Healthy and Mildew-Affected Leaves'
        )
        st.write("---")

    if st.checkbox("Image Montage"):
        st.write(
            "* To refresh the montage, click on the **Create Montage** button"
        )
        my_data_dir = 'inputs/cherry_leaves/cherry-leaves/validation'

        try:
            labels = os.listdir(my_data_dir)
        except FileNotFoundError:
            st.error(
                "Could not find validation folder. Please check your "
                "input path."
            )
            return

        label_to_display = st.selectbox(
            "Select label", options=labels, index=0
        )

        if st.button("Create Montage"):
            image_montage(
                dir_path=my_data_dir,
                label_to_display=label_to_display,
                nrows=4,
                ncols=4,
                figsize=(10, 10)
            )
        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    sns.set_style("white")

    try:
        labels = os.listdir(dir_path)
    except FileNotFoundError:
        st.error(f"Directory {dir_path} not found.")
        return

    if label_to_display not in labels:
        st.error(
            f"The label '{label_to_display}' does not exist in {dir_path}")
        return

    images_list = os.listdir(os.path.join(dir_path, label_to_display))

    if nrows * ncols > len(images_list):
        st.warning(
            f"Not enough images to fill a {nrows}x{ncols} grid. "
            f"Only {len(images_list)} images available."
        )
        return

    selected_imgs = random.sample(images_list, nrows * ncols)
    plot_idx = list(itertools.product(range(nrows), range(ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i in range(nrows * ncols):
        img_path = os.path.join(dir_path, label_to_display, selected_imgs[i])
        img = imread(img_path)
        shape = img.shape
        axes[plot_idx[i][0], plot_idx[i][1]].imshow(img)
        axes[plot_idx[i][0], plot_idx[i][1]].set_title(
            f"{shape[1]}x{shape[0]}")
        axes[plot_idx[i][0], plot_idx[i][1]].axis("off")

    plt.tight_layout()
    st.pyplot(fig=fig)
