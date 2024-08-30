# Artificial-Intelligence

◇ This particular Repository is for my projects. These projects showcase all that I know about AI.

Contributing

◇ If you would like to contribute to this project, please fork the repository and create a pull request with your changes. Contributions are welcome!

Projects Completed Till Now

Machine Learning 
> Supervised Learning
-- Classification

### ◇ Project 1

# 🍷 Wine Quality Prediction

This project applies Logistic Regression, Random Forest, and Support Vector Machine (SVM) algorithms to predict wine quality, demonstrating model setup, evaluation, and performance comparison.

## 📂 Project Structure

- **data/**: Contains the dataset and any preprocessed data files.
- **notebooks/**: Jupyter notebooks with code for data exploration, model training, and evaluation.
- **src/**: Python scripts for data processing, model training, and utility functions.
- **dash_app/**: Contains the Dash application code for the interactive dashboard.
- **README.md**: This documentation file.
- **requirements.txt**: Python dependencies for the project.

## 🔍 Project Overview

This project implements a comprehensive analysis to predict wine quality using three different machine learning models: Logistic Regression, Random Forest, and SVM. The project covers the entire machine learning pipeline from data preprocessing to model evaluation and comparison.

### 🛠️ Steps Taken

1. **Data Preprocessing**
   - Filtered wine quality scores to include values between 3 and 8 and categorized them into three classes: 'low,' 'medium,' and 'high.'
   - One-hot encoded the quality labels for classification and standardized the features to ensure optimal model performance.

2. **Model Development**
   - **Logistic Regression**
     - Trained a multinomial Logistic Regression model.
     - Evaluated the model using a confusion matrix and classification report.
   - **Random Forest**
     - Trained a Random Forest classifier with class balancing.
     - Analyzed feature importance to identify the most significant features.
     - Adjusted thresholds in the model to optimize performance.
   - **Support Vector Machine (SVM)**
     - Trained an SVM model for classification.
     - Plotted ROC curves for each class to evaluate model performance.

3. **Model Evaluation**
   - Calculated performance metrics: accuracy, precision, recall, and F1-score.
   - Generated confusion matrices and ROC curves for each model.
   - Conducted a detailed comparison of the models' strengths and weaknesses.

4. **Dashboard Implementation**
   - Developed an interactive Dash application for model evaluation visualization.
   - Features include dropdowns, radio buttons, and interactive graphs.
   - Customized the layout and design for presentation readiness.

5. **Issue Resolution**
   - Implemented solutions for handling labels with no predicted samples using the `zero_division` parameter in precision, recall, and F1-score calculations.

## 📊 Model Evaluation Dashboard

An interactive dashboard built using Dash that allows users to:
- Select a model (Logistic Regression, Random Forest, SVM) and view its Confusion Matrix or ROC Curve.
- Compare model performance metrics through an easy-to-use interface.

### 📦 Installation

To run the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/wine-quality-prediction.git
   cd wine-quality-prediction
