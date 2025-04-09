# Cherry Leaf Mildew Detection 

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.


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

## Dashboard Design

- Planned

| Page              | Content                                                                 |
|------------------|-------------------------------------------------------------------------|
| Home             | Project overview and instructions                                       |
| Model Prediction | Image uploader and prediction result display                            |
| Model Evaluation | Training accuracy/loss, confusion matrix, classification report         |
| Visual Study     | Shows sample healthy and mildew-affected leaves. Includes image montage, average image comparisons, and pixel-level difference image. Supports visual differentiation goal. |

> Note: Pages will be implemented in `Streamlit`, and this section will be updated as the dashboard is developed.

## Bugs

### IsADirectoryError During Dataset Cleanup

- **Issue:** While attempting to remove non-image files from the dataset, the script triggered an `IsADirectoryError` because it attempted to delete folders instead of files.
- **Why it occurred:** The initial implementation used `os.listdir()` and assumed all items were files, but some were directories (e.g., `healthy`, `powdery_mildew` subfolders).
- **How it was resolved:** The file removal function was updated to use `os.walk()`, which recursively iterates through all directories and ensures only files are targeted for deletion.
- **Outcome:** The issue was resolved successfully, and the cleaned dataset is now verified and usable for further analysis.

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

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
- Concludes with model evaluation on unseen test data, saving accuracy/loss to a JSON file for dashboard integration

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

## Credits

- In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- Dataset from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves)
- The overall structure of the project, including the Jupyter notebook workflows and the Streamlit dashboard layout, was based on materials and templates provided by Code Institute as part of the Predictive Analytics module.
- Specific components adapted and customised include:
  - The `DataCollection`, `DataVisualisation`, and `Modelling and Evaluating` notebooks.
  - The multi-page Streamlit structure (`app_pages/`, `src/` folder and routing).
  - The business problem and dataset were defined by the Code Institute Cherry Leaves project brief.
- Some utility function logic was adapted from Code Institute learning materials.
- All code has been reviewed and adapted where necessary to meet the specific goals of this project and ensure understanding and independent implementation.

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.
