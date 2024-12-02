# RUL Prediction for Lithium-Ion Batteries

This project predicts the **Remaining Useful Life (RUL)** of lithium-ion batteries using advanced machine learning models. It leverages feature normalization, exploratory data analysis (EDA), and state-of-the-art regression models to deliver accurate predictions.

## Project Overview

- **Dataset**: The project uses a dataset containing features such as discharge time, charging time, voltage, and more.
- **Goal**: To predict RUL and evaluate model performance.
- **Models Implemented**:
  - Random Forest Regressor
  - k-Nearest Neighbors (kNN)
  - Support Vector Machines (SVM)
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score

## Key Steps

### 1. **Data Preprocessing**
- Handled missing values by removing incomplete rows.
- Features were normalized using **MinMaxScaler** to improve model performance.
- Dataset split into training (80%) and testing (20%) sets.

### 2. **Exploratory Data Analysis**
- Visualized the distribution of RUL values using histograms.
- Correlations between features analyzed using heatmaps.
- Interactive scatter plots created using **Plotly** for better insights.

### 3. **Model Implementation**
- Implemented and trained **Random Forest**, **kNN**, and **SVM** regression models.
- Tuned hyperparameters for optimal performance.

### 4. **Model Evaluation**
- Evaluated models using:
  - **MAE**: Measures average error magnitude.
  - **RMSE**: Penalizes larger errors more than MAE.
  - **R² Score**: Indicates goodness of fit.
