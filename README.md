# Medical Image Segmentation for Disease Detection using Brain Tumor Classifier Application

## 1. Dataset Information

This project uses the **Brain Tumor Image DataSet: Semantic Segmentation**, which is available on Kaggle. The dataset can be downloaded from the following link:

[Brain Tumor Image DataSet: Semantic Segmentation](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation)

### Dataset Structure
- **Train, Validation, and Test Images**: The dataset contains labeled images organized for training, validation, and testing.
- **Annotations in COCO Format**: The dataset includes annotation files (`_annotations.coco.json`) used for labeling.

---

## 2. Application Overview

The **Brain Tumor Classifier Application** classifies brain tumor images into two categories:
- **Not Tumor**
- **Tumor**

It utilizes an enhanced version of the **ResNet-18** architecture that has been trained on the dataset.

---

## 3. How to Use the Application

### Step 1: Upload the Model File
The application requires the trained model file `enhanced_model.pth` to make predictions. This file contains:
- The trained weights of the **Enhanced ResNet-18** model.
- The optimizer state for potential resumption of training.
- Training metrics such as accuracy, precision, recall, and F1-score.

#### Create the `enhanced_model.pth` File
After training the **Enhanced ResNet-18** model, save the model file by running the following command in your training environment:

```python
torch.save({
    "model_state_dict": enhanced_model.state_dict(),
    "optimizer_state_dict": enhanced_optimizer.state_dict(),
    "metrics": enhanced_metrics
}, "enhanced_model.pth")
```
### Step 2. Launch the Application
To launch the application, use Streamlit to run it. Navigate to the application directory and execute the following command:

```bash
streamlit run app.py
```
### Step 3. Upload an Image
se the app to upload a brain tumor image in **JPEG**, **PNG**, or **JPG** format. The app will preprocess the image and display its grayscale version.

### Step 4. View Prediction Results
After uploading an image, the app will:

- Predict whether the image contains a Tumor or Not Tumor.
- Display probability scores for each class (Tumor and Not Tumor).
