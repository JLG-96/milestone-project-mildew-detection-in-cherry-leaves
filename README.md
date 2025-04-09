# Cherry Leaf Mildew Detection – Predictive Analytics Dashboard

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
