# Artificial-Intelligence

◇ This particular Repository is for my projects. These projects showcase all that I know about AI.

---

## ◇ Projects Completed Till Now

Machine Learning 
> Supervised Learning
-- Regression

---

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

## ◇ Project 3  
# 🍱 **Airbnb Pricing Prediction using Machine Learning**

### Project Overview  
This project focuses on predicting Airbnb listing prices by applying multiple machine learning models and performing extensive feature engineering to enhance predictive performance.

The dataset includes key categorical and numerical attributes such as room type, neighbourhood group, availability, review scores, and host/booking characteristics that influence pricing.

Advanced feature engineering techniques were used, including:
- Ratio features (e.g., price per minimum night)
- Host verification and booking flags
- Log transformations
- Property age and grouped listing insights

A correlation matrix was generated to assess the strength of relationships between features and the target variable, helping to refine model inputs.

Final models trained include:
- Random Forest
- XGBoost
- LightGBM

These were evaluated using RMSE and R² metrics. Random Forest performed best after tuning and feature selection.

### Data Source  
Airbnb listing data from the U.S., containing variables like:
- `room_type`, `minimum_nights`, `host_identity_verified`, `cancellation_policy`, and `price`.

---

## ◇ Project 4  
# 🏢 **Energy Usage Prediction using Machine Learning**

### Project Overview  
This project aims to predict building energy usage (meter readings) by applying multiple regression-based models and powerful engineered features.

Key features include:
- Numerical and environmental factors: air temperature, dew point, wind speed
- Building-specific data like square footage and floor count

Feature engineering included:
- Interaction terms (e.g., temperature × building size)
- Log and polynomial transformations
- Aggregation and normalization of variables

Models evaluated:
- Linear Regression, Random Forest, Gradient Boosting, Decision Tree
- KNeighbors Regressor, SVR, AdaBoost

Model performance was measured using Mean Squared Error (MSE) and R². **Random Forest** achieved the best results with low error and high accuracy.

### Data Source  
The dataset includes daily energy consumption and weather readings across multiple commercial buildings.

---

## ◇ Project 5  
# 🗾 **Predicting Life Expectancy in Japanese Prefectures**

### Project Overview  
This project aims to understand and predict life expectancy across Japan’s prefectures using health, environmental, and economic indicators.

Features used include:
- Number of physicians and hospitals
- Healthcare expenditure
- Income per capita
- Educational attainment

Objectives:
- Perform EDA to explore relationships
- Train regression models to predict life expectancy
- Analyze feature importance and correlations

Machine learning models help identify the key drivers of longevity and provide policy insights.

### Data Source  
Life expectancy and socio-economic data collected from Japanese government statistics and public healthcare datasets.

---

Machine Learning 
> Supervised Learning
-- Classification

## ◇ Project 6  
# 📨 Spam Detection using Machine Learning and NLP

### Project Overview  
This project focuses on identifying spam messages in SMS data using a combination of traditional machine learning algorithms and advanced NLP techniques. It includes preprocessing, TF-IDF vectorization, model training, and evaluation. The pipeline also extends to deep learning using BERT.

### Features used include:  
- Cleaned SMS text (lowercasing, punctuation removal, whitespace normalization)  
- TF-IDF vector representations of messages  
- Binary classification labels (`spam` or `ham`)

### Models Applied:  
- Naive Bayes  
- Logistic Regression  
- XGBoost  
- BERT (via HuggingFace Transformers)

### Steps Taken:  
- Loaded and cleaned SMS text data  
- Applied regular expression-based text cleaning  
- Converted messages to numerical format using TF-IDF  
- Trained and compared various ML models  
- Evaluated performance using accuracy, precision, recall, and F1-score  
- Integrated HuggingFace BERT for deep learning classification

### Tools and Libraries Used:  

- **Python Libraries:**
  - `pandas` for data wrangling and transformation  
  - `numpy` for numerical operations  
  - `re` for text cleaning via regex  
  - `matplotlib` and `seaborn` for visualizations  
  - `scikit-learn` for modeling and evaluation  
  - `xgboost` for boosting classifiers  
  - `transformers` from HuggingFace for BERT  
  - `datasets` for preparing and managing text data  

- **Machine Learning Models:**
  - Naive Bayes
  - Logistic Regression
  - XGBoost
  - BERT (fine-tuned using HuggingFace)

### Data Source  
The dataset used is a publicly available SMS spam collection, which includes thousands of messages labeled as either spam or ham. It is commonly used for benchmarking text classification models.


