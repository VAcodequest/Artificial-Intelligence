# Artificial-Intelligence

◇ This particular Repository is for my projects. These projects showcase all that I know about AI.

Contributing

◇ If you would like to contribute to this project, please fork the repository and create a pull request with your changes. Contributions are welcome!

## ◇ Projects Completed Till Now

Machine Learning 
> Supervised Learning
-- Classification

### ◇ Project 1

# 🍷 Wine Quality Prediction

This project applies Logistic Regression, Random Forest, and Support Vector Machine (SVM) algorithms to predict wine quality, demonstrating model setup, evaluation, and performance comparison.

## Project Overview

This project implements a comprehensive analysis to predict wine quality using three different machine learning models: Logistic Regression, Random Forest, and SVM. The project covers the entire machine learning pipeline from data preprocessing to model evaluation and comparison.

### Steps Taken

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

## Model Evaluation Dashboard

This project also has an interactive dashboard built using Dash that allows users to:
- Select a model (Logistic Regression, Random Forest, SVM) and view its Confusion Matrix or ROC Curve.
- Compare model performance metrics through an easy-to-use interface.

Machine Learning 
> Supervised Learning
-- Regression

### ◇ Project 2

# 📈 Stock Price Prediction Using Machine Learning

This project applies **Random Forest**, **Ridge Regression**, and **Gradient Boosting** to predict stock prices for **Apple Inc. (AAPL)**, showcasing model development, evaluation, and performance comparison.

## Project Overview

This project demonstrates a complete stock price prediction workflow using machine learning techniques. It integrates historical stock data and engineered technical indicators to train and evaluate multiple models.

### Steps Taken

1. **Data Preprocessing**
   * Cleaned and standardized historical stock price data from **Yahoo Finance**.
   * Removed missing values and handled data inconsistencies.
   * Normalized stock price features to optimize model performance.

2. **Feature Engineering**
   * Computed key technical indicators:
     * **Simple Moving Averages (SMA)**
     * **Exponential Moving Averages (EMA)**
     * **Bollinger Bands**
     * **Relative Strength Index (RSI)**
     * **Rate of Change (ROC)**
     * **Lagged stock prices**

3. **Model Development**
   * **Random Forest**
     * Trained a Random Forest Regressor.
     * Analyzed feature importance to identify key predictors.
   * **Ridge Regression**
     * Trained a Ridge Regression model to capture linear trends.
   * **Gradient Boosting**
     * Implemented Gradient Boosting to optimize predictions through iterative boosting.

4. **Model Evaluation**
   * Calculated **Mean Squared Error (MSE)** and **R-squared (R²)** to evaluate model performance.
   * **Random Forest** outperformed other models with the lowest MSE and highest R².
   * Conducted a detailed comparison of the models to highlight their strengths and weaknesses.

5. **Dashboard Implementation**
   * Developed an interactive **Dash application** for real-time stock analysis and visualization.
   * Features include dropdowns, radio buttons, and interactive graphs.
   * Users can explore stock price predictions, technical indicators, and actual prices.

6. **Issue Resolution**
   * Addressed performance limitations by adjusting model hyperparameters.
   * Implemented strategies for handling lag features and improving model accuracy.

## Model Evaluation Dashboard

This project also includes an interactive **Dash dashboard** that allows users to:
* Visualize stock price predictions alongside technical indicators.
* Compare model performance using **Random Forest**, **Ridge Regression**, and **Gradient Boosting**.
* Explore key technical indicators like **SMA**, **EMA**, **Bollinger Bands**, and more.

