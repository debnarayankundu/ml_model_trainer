# Machine Learning Model Trainer

## Overview
This Streamlit-based web application enables users to upload datasets, preprocess data, visualize features, and train machine learning models for regression, classification, and clustering.

## Features
- **Data Upload**: Upload CSV files for analysis.
- **Data Preprocessing**:
  - Drops missing values and duplicates.
  - Encodes categorical features.
  - Scales numerical features.
- **Univariate & Bivariate Analysis**:
  - Count plots for categorical variables.
  - Histograms for numerical variables.
  - Scatter plots and box plots for feature relationships.
- **Feature Selection**:
  - Users can choose to run models on all or selected features.
- **Model Selection**:
  - **Regression**: `Linear Regression`, `Decision Tree Regressor`, `Random Forest Regressor`
  - **Classification**: `Logistic Regression`, `Decision Tree Classifier`, `SVM`, `KNN`, `Random Forest Classifier`
  - **Clustering**: `K-Means Clustering`
- **Performance Metrics**:
  - **Regression**: `MSE`, `RMSE`, `R² Score`, `MAE`
  - **Classification**: `Accuracy`, `Confusion Matrix`

## Installation
### Prerequisites
Ensure you have Python 3.x installed on your system.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/debnarayankundu/ml_model_trainer.git
   cd ml_model_trainer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload a CSV dataset.
2. Perform data preprocessing and visualization.
3. Select problem type (`Regression`, `Classification`, `Clustering`).
4. Choose and train a model.
5. View performance metrics.

## Dependencies
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install them using:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```
## model used for Diabetes prediction

![image](https://github.com/user-attachments/assets/f2999232-9d34-4230-ac9e-539290b7a243)

![image](https://github.com/user-attachments/assets/029657f2-81b9-4ba7-b478-922b7fdbf134)

![image](https://github.com/user-attachments/assets/3c12cb37-5e98-4a81-8fd3-7490cb730223)

![image](https://github.com/user-attachments/assets/0a656a2a-e508-4f45-928e-48dae7441eb9)

