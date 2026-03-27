# Transfer Learning Model Development: Chest X-Ray Classification

## Project Overview
This project focuses on developing a deep learning framework for multi-class chest X-ray classification using transfer learning. The primary objective is to fine-tune a pre-trained Convolutional Neural Network (CNN) model by modifying its top layers and optimizing hyperparameters to achieve effective results on medical image tasks.

## Objective
* To implement transfer learning by using a pre-trained model as a feature extractor.
* To modify and fine-tune the architecture for a specific 3-class classification problem (COVID, Pneumonia, and Normal).
* To evaluate the effectiveness of freezing convolutional bases versus fine-tuning for limited medical datasets.

## Dataset Information
The project utilizes the **Coronahack Chest X-Ray Dataset** sourced from Kaggle.
* **Source:** Kaggle Dataset Link: [Coronahack Chest X-Ray Dataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset)
* **Classes:** 3 (COVID, PNEUMONIA, NORMAL).
* **Organization:** The dataset was automatically reorganized into class-specific folders using filename keywords (virus, bacteria, normal) to comply with Keras's `flow_from_directory()` requirements.

## Research Context
The methodology is grounded in research such as *"Deep Transfer Learning for COVID-19 Detection using Chest X-Ray Images"* (2020) by P. K. Sethy and S. K. Behera. Their study demonstrated that pre-trained models like ResNet50 and VGG16 significantly improve classification accuracy compared to training from scratch, particularly when the dataset size is limited.

## Methodology

### Model Architecture
* **Base Model:** ResNet50, pre-trained on the ImageNet dataset.
* **Modifications:** The original classification layer was removed and replaced with task-specific layers, including:
    * Global Average Pooling (GlobalAveragePooling2D).
    * Fully connected layers (Dense).
    * Dropout layers for regularization.

### Training Approach
* **Data Augmentation:** Applied to the training set to improve model generalization.
* **Freezing:** The convolutional base was initially frozen to reduce training time while maintaining high accuracy.
* **Fine-tuning:** Top layers were subsequently fine-tuned to enhance validation performance.

## Performance & Results
* **Validation Accuracy:** Approximately **41%**.
* **Conclusion:** Transfer learning using ResNet50 demonstrated strong potential for multi-class chest X-ray classification, with fine-tuning proving essential for enhancing medical image classification performance.

## Technologies Used
* **Framework:** TensorFlow / Keras.
* **Data Handling:** NumPy, ImageDataGenerator.
* **Visualization:** Matplotlib, Seaborn.
* **Metrics:** Scikit-learn (Classification Report, Confusion Matrix).

## Contributors
* **Darshan Bhabad** (Roll No: 202301040169).
* **Krishna Tolani** (Roll No: 202301040073).
* **Mitesh Chaudhari** (Roll No: 202301040106).
