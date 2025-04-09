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
        f"* The client is interested in a study that visually differentiates healthy cherry leaves from those affected by powdery mildew.")

    version = 'v1'
    if st.checkbox("Average and variability images"):
        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")
        avg_mildew = plt.imread(f"outputs/{version}/avg_var_powdery_mildew.png")

        st.warning(
            f"* The average and variability images may not show clear visual cues to the human eye, "
            f"but subtle differences in leaf texture and coloration are observable.")
        
        st.image(avg_healthy, caption='Healthy Leaf - Average and Variability')
        st.image(avg_mildew, caption='Powdery Mildew Leaf - Average and Variability')
        st.write("---")

    if st.checkbox("Difference between class averages"):
        diff_image = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            f"* This visualisation shows the pixel-level difference between the average images of each class.")
        st.image(diff_image, caption='Difference Between Healthy and Mildew-Affected Leaves')
        st.write("---")

    if st.checkbox("Image Montage"): 
        st.write("* To refresh the montage, click on the 'Create Montage' button")
        my_data_dir = 'inputs/cherry_leaves/cherry-leaves/validation'
        labels = os.listdir(my_data_dir)
        label_to_display = st.selectbox(label="Select label", options=labels, index=0)
        if st.button("Create Montage"):      
            image_montage(dir_path=my_data_dir,
                          label_to_display=label_to_display,
                          nrows=4, ncols=4, figsize=(10,10))
        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

    if label_to_display in labels:
        images_list = os.listdir(os.path.join(dir_path, label_to_display))
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.warning(f"Not enough images to create montage of size {nrows}x{ncols}.")
            return

        plot_idx = list(itertools.product(range(nrows), range(ncols)))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        
        for i in range(nrows * ncols):
            img_path = os.path.join(dir_path, label_to_display, img_idx[i])
            img = imread(img_path)
            shape = img.shape
            axes[plot_idx[i][0], plot_idx[i][1]].imshow(img)
            axes[plot_idx[i][0], plot_idx[i][1]].set_title(f"{shape[1]}x{shape[0]}")
            axes[plot_idx[i][0], plot_idx[i][1]].axis("off")

        plt.tight_layout()
        st.pyplot(fig=fig)

    else:
        st.error(f"The label '{label_to_display}' does not exist in {dir_path}")
