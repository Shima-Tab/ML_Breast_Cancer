# Brest_Cancer_Detection_ML
Breast Cancer Diagnosis Classification
This repository contains an end-to-end machine learning pipeline for the classification of breast cancer diagnosis. The project demonstrates data exploration, preprocessing, model training with multiple algorithms, hyperparameter tuning, advanced visualization, ensemble techniques, feature engineering/selection, and deployment using Flask. In addition, fairness metrics are computed using AIF360.

Table of Contents
Overview
Repository Structure
Installation
Usage
Pipeline Details
Data Exploration and Preprocessing
Model Training and Evaluation
Hyperparameter Tuning
Advanced Visualizations
Feature Engineering and Selection
Model Ensemble and Stacking
Model Deployment
Fairness Evaluation
Contributing
License
Overview
The goal of this project is to develop and compare multiple classification models using a dataset that contains various features related to breast cancer (e.g., radius_mean, texture_mean, area_mean, etc.). The pipeline includes:

Data Exploration: Visualizing the distribution and correlation of features.
Preprocessing: Handling missing values, encoding categorical data, and scaling features.
Modeling: Training several classifiers (Logistic Regression, Decision Tree, Random Forest, SVC, KNN) and evaluating their performance.
Hyperparameter Tuning: Using GridSearchCV to optimize a Random Forest model.
Visualization: Creating static (Matplotlib/Seaborn) and interactive (Plotly) visualizations.
Feature Engineering: Creating new features and selecting the most important ones using Recursive Feature Elimination (RFE).
Ensemble Learning: Combining multiple models with a VotingClassifier.
Deployment: Serving the final model via a Flask API.
Fairness Metrics: Computing fairness metrics using the AIF360 toolkit.
Repository Structure
graphql
Copy
├── data.csv                  # Input dataset file (ensure this file is in the root folder)
├── final_model.sav           # Serialized final model (generated after training)
├── app.py                    # Flask API for model deployment
├── notebook.ipynb            # Jupyter Notebook with full code implementation (or equivalent Python scripts)
├── README.md                 # This file
└── requirements.txt          # List of dependencies (if provided)
Installation
Ensure you have Python 3.8 or above installed. Install the required libraries using pip:

bash
Copy
pip install numpy pandas matplotlib seaborn plotly scikit-learn flask aif360
If you have a requirements.txt file, you can install dependencies with:

bash
Copy
pip install -r requirements.txt
Usage
Data Exploration & Model Training:

Open the Jupyter Notebook (notebook.ipynb) or run the Python script in your preferred IDE.
The notebook walks through data exploration, preprocessing, model training, evaluation, and hyperparameter tuning.
Model Deployment with Flask:

To start the Flask API, run the following command:
bash
Copy
python app.py
The API will start on http://127.0.0.1:5000.
Send a POST request with JSON data representing the feature values to the /predict endpoint.
Example JSON payload:
json
Copy
{
  "radius_mean": 14.2,
  "perimeter_mean": 92.5,
  "area_mean": 654.0,
  "symmetry_mean": 0.18,
  "compactness_mean": 0.12,
  "concave points_mean": 0.09
}
Fairness Metrics Evaluation:

The notebook also includes a section on computing fairness metrics using AIF360. Adjust the dataset and protected attribute names as needed.
Pipeline Details
Data Exploration and Preprocessing
Data Loading: The dataset is loaded from data.csv and basic statistics, head/tail samples, and info are printed.
Handling Missing Data: Columns with missing values are dropped.
Encoding: The diagnosis column is label encoded.
Visualization: Counts, pair plots, and a correlation heatmap are generated to understand feature relationships.
Model Training and Evaluation
Multiple classifiers (Logistic Regression, Decision Tree, Random Forest, SVC, and KNN) are trained.
Each model is evaluated using accuracy, classification reports, and confusion matrices.
Hyperparameter Tuning
A grid search is performed on a Random Forest classifier to optimize parameters such as n_estimators, max_depth, max_features, etc.
The best parameters and score are reported.
Advanced Visualizations
An interactive heatmap is created using Plotly to explore the correlation matrix of selected features.
Feature Engineering and Selection
A new feature (area_perimeter_ratio) is created.
Recursive Feature Elimination (RFE) is used to identify the most significant features.
Model Ensemble and Stacking
An ensemble model is built using a VotingClassifier that combines Random Forest, SVC, and Logistic Regression.
The ensemble's performance is evaluated on the test set.
Model Deployment
The final model is saved using pickle and later loaded by a Flask API to serve predictions.
The API accepts JSON input and returns the model’s prediction.
Fairness Evaluation
Using AIF360, the code wraps the test data into a BinaryLabelDataset and computes fairness metrics such as disparate impact.
This section demonstrates how to assess fairness across protected groups (e.g., race, gender).
