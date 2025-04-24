
## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
  - [Specific requirements](#specific-requirements)
- [The rationale to map the business requirements to the Data Visualisations and ML tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualisations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Dashboard Design](#dashboard-design)
- [Bugs](#bugs)
  - [IsADirectoryError During Dataset Cleanup](#isadirectoryerror-during-dataset-cleanup)
  - [Slug Size / Deployment Challenges](#slug-size--deployment-challenges)
- [Deployment](#deployment)
  - [Render](#render)
  - [Notes on Deployment:](#notes-on-deployment)
  - [Deployment Steps on Render:](#deployment-steps-on-render)
- [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
- [Notebook Overviews](#notebook-overviews)
  - [`01 - DataCollection.ipynb`](#01---datacollectionipynb)
  - [`02 - DataVisualization.ipynb`](#02---datavisualizationipynb)
  - [`03 - Modelling and Evaluating.ipynb`](#03---modelling-and-evaluatingipynb)
- [Source Code Structure](#source-code-structure)
- [App Testing](#app-testing)
- [Future Improvements](#future-improvements)
- [Credits](#credits)
  - [Content](#content)
  - [Media](#media)
- [Acknowledgements](#acknowledgements)

## Introduction

Farmy & Foods, a leading supplier of premium produce, has identified an operational bottleneck: manually inspecting cherry trees for signs of powdery mildew is time-consuming and unsustainable at scale. Employees currently spend up to 30 minutes per tree inspecting and treating leaves—limiting coverage across farms and delaying intervention.

This project explores the application of predictive analytics to automate the detection of mildew in cherry leaves using image classification. Leveraging a labeled dataset of cherry leaves, we build a dashboard that combines machine learning with visual tools to support early detection and inform decision-making.

The dashboard serves two primary business needs:

1. **Visual Differentiation** – Help users clearly distinguish between healthy and mildew-affected leaves.
2. **Predictive Classification** – Use a trained Convolutional Neural Network (CNN) to determine if an uploaded cherry leaf image is healthy or infected with mildew.

The solution was developed in Python using Jupyter notebooks and deployed using Streamlit on Render. A live version of the app may be found [here](https://project-mildew-detection.onrender.com)

## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).

It contains over 4,000 images taken from Farmy & Foods’ cherry tree plantations. These images are divided into two categories: healthy cherry leaves and leaves infected with powdery mildew — a fungal disease known to affect many plant species. 

This crop represents one of the most valuable products in Farmy & Foods’ portfolio, and the client has expressed concerns that the outbreak of powdery mildew may be compromising both the quality of their product and their reputation in the market. The dataset serves as the foundation for building and validating a machine learning model to assist with large-scale, accurate diagnosis and quality control across their nationwide farms.

## Business Requirements

Farmy & Foods, a national agricultural company, is currently facing a significant challenge in maintaining the quality of their cherry crop due to an outbreak of powdery mildew — a fungal disease that affects the leaves of cherry trees.

At present, the company relies on a manual inspection process. Each tree is assessed visually by an employee, who samples a few leaves to determine whether powdery mildew is present. If mildew is detected, a fungicidal compound is applied. This inspection process takes approximately 30 minutes per tree, making it slow, labour-intensive, and ultimately unscalable given the thousands of trees across the company's farms.

To improve efficiency and reduce labour costs, Farmy & Foods has requested a machine learning (ML) solution capable of detecting powdery mildew from uploaded leaf images. This would allow for faster diagnosis and treatment, reducing crop loss and supporting quality control across large volumes of trees.

Key stakeholders include:

- **Farmy & Foods** — the client seeking a scalable inspection solution.
- **End customers** — who expect consistently high-quality produce.

To meet these business requirements, the solution must:

- Accurately classify cherry leaf images as **healthy** or **infected with powdery mildew**.
- Provide fast predictions suitable for real-time use.
- Be easy to understand and operate for both technical staff and field personnel.
- Offer visual insights to support the ML findings, including comparisons of healthy vs infected leaves.

### Specific requirements

1. Conduct a **visual study** to help the client understand the differences between healthy and infected leaves. This includes:
   - Sample montages of both leaf types
   - Average images for each class
   - Variability and difference images

2. Develop a **Convolutional Neural Network (CNN)** capable of predicting whether a given cherry leaf is healthy or infected with powdery mildew.


---
---
---
---
---
---
---
---
---
---
---
---
---








## The rationale to map the business requirements to the Data Visualisations and ML tasks

| Business Requirement | ML Task                         | Visualisation Task                          |
|----------------------|----------------------------------|---------------------------------------------|
| Instant prediction    | Image classification with CNN    | Accuracy/loss plots, confusion matrix        |
| Visual differentiation       | Not required (visual only) | Image montage, average image comparison, difference image |

## ML Business Case

- **Objective:** Reduce inspection time and increase early detection of mildew.
- **Method:** Use a CNN model trained on labeled image data.
- **Ideal Outcome:** A model that correctly classifies leaf images in real-time.
- **Success Metrics:** High accuracy, balanced precision/recall, generalization to unseen data.
- **Output:** Prediction label and confidence score.
- **Relevance:** Enables large-scale, fast inspection across farms.
- **Limitation:** The model has been trained solely on cherry leaf images and is not designed to identify whether an uploaded image is a cherry leaf or not. If a non-leaf image is accidentally uploaded, the model will still classify it as either "healthy" or "powdery mildew," as it does not have a rejection mechanism or multi-class structure. This reflects a typical limitation of binary classifiers, and highlights the importance of ensuring only cherry leaf images are used during prediction.

## Dashboard Design

- Planned

| Page              | Content                                                                 |
|------------------|-------------------------------------------------------------------------|
| Home             | Project overview and instructions                                       |
| Model Prediction | Image uploader and prediction result display                            |
| Model Evaluation | Training accuracy/loss, confusion matrix, classification report         |
| Visual Study     | Shows sample healthy and mildew-affected leaves. Includes image montage, average image comparisons, and pixel-level difference image. Supports visual differentiation goal. |

> Note: All planned dashboard pages have been implemented using `Streamlit`.

## Bugs

### IsADirectoryError During Dataset Cleanup

- **Issue:** While attempting to remove non-image files from the dataset, the script triggered an `IsADirectoryError` because it attempted to delete folders instead of files.
- **Why it occurred:** The initial implementation used `os.listdir()` and assumed all items were files, but some were directories (e.g., `healthy`, `powdery_mildew` subfolders).
- **How it was resolved:** The file removal function was updated to use `os.walk()`, which recursively iterates through all directories and ensures only files are targeted for deletion.
- **Outcome:** The issue was resolved successfully, and the cleaned dataset is now verified and usable for further analysis.

### Slug Size / Deployment Challenges

- **Issue:** Initial deployment to Heroku failed due to the compiled slug exceeding the 500MB limit.
- **Why it occurred:** The project included large directories and unnecessary files (e.g., image datasets, test data, notebooks).
- **How it was resolved:** A `.slugignore` file was created to exclude folders and files not required to run the Streamlit app.
- **Outcome:** Deployment succeeded after switching to Render, which allows for larger project sizes and simpler configuration using a start command.

## Deployment

### Render

This project was deployed using [Render](https://render.com) on their free tier. The live web application can be accessed here:  
**[https://project-mildew-detection.onrender.com](https://project-mildew-detection.onrender.com)**

### Notes on Deployment:

- The app may take up to 30–60 seconds to load if it hasn't been accessed recently. This is due to cold starts on the free Render tier.
- Once the application loads, it performs as expected.

### Deployment Steps on Render:

1. Create a free account at [render.com](https://render.com).
2. Connect your GitHub repository to Render.
3. Create a new **Web Service**, selecting the correct repository.
4. Set the following options during configuration:
   - **Build Command:**  
     ```
     pip install -r requirements.txt && ./setup.sh
     ```
   - **Start Command:**  
     ```
     streamlit run app.py --server.address=0.0.0.0 --server.port=10000
     ```
5. Specify the Python version (e.g. `3.10.12`) using a `runtime.txt` file or a `PYTHON_VERSION` environment variable.
6. Click deploy and wait for the build process to complete.

## Main Data Analysis and Machine Learning Libraries

- `os`, `shutil`: File and directory operations
- `Pillow (PIL)`: Image loading and validation
- `numpy`: Used to calculate average and difference images, and format model input
- `matplotlib.pyplot`: Used for image display and prediction probability charts
- `tensorflow.keras`: For CNN model creation and prediction
- `streamlit`: To build an interactive user interface

## Notebook Overviews

### `01 - DataCollection.ipynb`

This notebook is responsible for loading and preparing the dataset for modelling. It includes:

- Verification and cleanup of the image directory
- Validation that all image files are in a supported format
- Splitting the dataset into `train`, `validation`, and `test` folders
- Setting the file structure required for training image classification models
- Summary cell outlining the conclusions and next steps

This notebook supports the business requirement by ensuring a high-quality and well-organised dataset for downstream machine learning tasks.

---

### `02 - DataVisualization.ipynb`

This notebook explores and visualises the image data. It includes:

- Calculation of image dimensions (width, height) across the dataset
- Image montage previewing healthy vs mildew-affected leaves
- Average image and pixel-level difference between healthy and mildew categories
- Confirmed image consistency and verified class balance
- Final markdown cell includes conclusions and next steps

This notebook supports the first business requirement by visually highlighting differences between healthy and infected leaves and confirming the dataset is balanced and suitable for modelling.

---

### `03 - Modelling and Evaluating.ipynb`

This notebook builds, trains, and evaluates a Convolutional Neural Network (CNN). It includes:

- Data augmentation using `ImageDataGenerator`
- CNN model creation and compilation using `tensorflow.keras`
- Application of early stopping to prevent overfitting
- Evaluation using test data, accuracy/loss curves, and a confusion matrix
- Saving the model and test performance metrics
- Concludes with model evaluation on unseen test data, saving accuracy/loss to a JSON file for dashboard integration.
- It is important to note that, as expected, the model misclassifies non-cherry leaf images (e.g., a dog), as it was trained exclusively on cherry leaf data. This confirms the model's appropriate specialization and highlights the importance of using representative input data during inference.

This notebook supports the second business requirement by enabling real-time prediction of whether a cherry leaf is healthy or has mildew.

---

## Source Code Structure

This project uses a modular structure with separate folders for notebooks, dashboard pages, and supporting code.

- `jupyter_notebooks/`  
  Contains the three core notebooks for data collection, visualisation, and modelling.

- `app_pages/`  
  Contains the Streamlit dashboard pages:
  - `page_summary.py`: Overview of the project and business requirements
  - `page_cherry_leaves_visualizer.py`: Visual study comparing healthy and mildew-affected leaves
  - `page_mildew_detector.py`: Upload interface to predict leaf health from an image

- `src/`  
  Holds reusable scripts:
  - `data_management.py`: Utility to download results as a timestamped CSV
  - `machine_learning/predictive_analysis.py`: Contains logic for resizing images, loading the model, making predictions, and plotting prediction probabilities

## App Testing

The final Streamlit app was tested using a range of inputs to confirm correct classification and to assess model behaviour under edge cases.

- A healthy cherry leaf image was correctly classified as *healthy*.
- A mildew-infected cherry leaf image was correctly classified as *powdery mildew*.
- A non-cherry image (e.g. a dog) was misclassified as either “healthy” or “powdery mildew,” which is expected due to the model's binary classification setup and exclusive training on cherry leaf data.

This testing confirmed the model is functioning reliably within its intended use case - the misclassification of non-leaf inputs also demonstrated the model's limitations.

## Future Improvements

- Add a class verification step to reject non-leaf images.
- Introduce a multi-class model for broader use across other crop types.
- Improve loading speed by moving to a paid hosting tier or lightweight model.

## Credits

### Content

- Dataset from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves)
- The overall structure of the project, including the Jupyter notebook workflows and the Streamlit dashboard layout, was based on materials and templates provided by Code Institute as part of the Predictive Analytics module.
- Specific components adapted and customised include:
  - The `DataCollection`, `DataVisualisation`, and `Modelling and Evaluating` notebooks.
  - The multi-page Streamlit structure (`app_pages/`, `src/` folder and routing).
  - The business problem and dataset were defined by the Code Institute Cherry Leaves project brief.
- Some utility function logic was adapted from Code Institute learning materials.
- All code has been reviewed and adapted where necessary to meet the specific goals of this project and ensure understanding and independent implementation.
- All external code was either provided by Code Institute, openly accessible for educational use, or adapted with appropriate understanding and acknowledgement.
- Help found

### Media

- Images displayed throughout the app are sourced from the cherry leaf dataset provided on [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).
- Additional test images used during app testing (e.g. downloaded leaf and non-leaf images) were sourced from public web searches for educational purposes only and are not part of the final deployed app.

## Acknowledgements

- Thanks to the Code Institute Slack community and tutors for their support and guidance, particularly when troubleshooting deployment issues and `.slugignore` configuration.
