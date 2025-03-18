import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error, confusion_matrix

# Streamlit App Title
st.title("Machine Learning Model Trainer")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset:")
    st.write(df.head())

    # Data Preprocessing
    st.write("### Data Preprocessing")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Univariate Analysis
    st.write("### Univariate Analysis")
    column_to_plot = st.selectbox("Select a Column to Visualize", df.columns)
    
    if not column_to_plot:
        st.warning("Please select a feature for visualization.")
    else:
        if df[column_to_plot].dtype == 'object' or len(df[column_to_plot].unique()) < 20:
            st.write("Categorical Column - Showing Count Plot")
            plt.figure(figsize=(8, 4))
            sns.countplot(x=df[column_to_plot])
            st.pyplot(plt)
        else:
            st.write("Numerical Column - Showing Histogram")
            plt.figure(figsize=(8, 4))
            sns.histplot(df[column_to_plot], bins=20, kde=True)
            st.pyplot(plt)
    
    # Bivariate Analysis
    st.write("### Bivariate Analysis")
    col1, col2 = st.selectbox("Select First Column", df.columns), st.selectbox("Select Second Column", df.columns)
    if not col1 or not col2:
        st.warning("Please select two features for visualization.")
    else:
        if df[col1].dtype != 'object' and df[col2].dtype != 'object':
            st.write("Numerical vs Numerical - Showing Scatter Plot")
            plt.figure(figsize=(8, 4))
            sns.scatterplot(x=df[col1], y=df[col2])
            st.pyplot(plt)
        elif df[col1].dtype == 'object' and df[col2].dtype != 'object':
            st.write("Categorical vs Numerical - Showing Box Plot")
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col1], y=df[col2])
            st.pyplot(plt)
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        st.write("Encoding categorical variables...")
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # Select target variable
    target_variable = st.selectbox("Select Target Variable", df.columns)
    feature_selection = st.radio("Do you want to select features manually?", ["Yes", "No"])
    
    if feature_selection == "Yes":
        selected_features = st.multiselect("Select Features", df.columns.tolist(), default=df.columns.tolist())
        if not selected_features:
            st.warning("Please select at least one feature.")
        X = df[selected_features]
    else:
        X = df.drop(columns=[target_variable])
    
    y = df[target_variable]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Choose Problem Type
    problem_type = st.radio("Select Problem Type:", ["Regression", "Classification", "Clustering"])
    
    if problem_type == "Regression":
        model_type = st.selectbox("Choose Model", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"])
    elif problem_type == "Classification":
        model_type = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree Classifier", "SVM", "KNN", "Random Forest Classifier"])
    elif problem_type == "Clustering":
        model_type = "KMeans Clustering"
    
    if st.button("Train Model"):
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree Regressor":
            model = DecisionTreeRegressor()
        elif model_type == "Random Forest Regressor":
            model = RandomForestRegressor()
        elif model_type == "Logistic Regression":
            model = LogisticRegression()
        elif model_type == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
        elif model_type == "SVM":
            model = SVC()
        elif model_type == "KNN":
            model = KNeighborsClassifier()
        elif model_type == "Random Forest Classifier":
            model = RandomForestClassifier()
        elif model_type == "KMeans Clustering":
            num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
            model = KMeans(n_clusters=num_clusters, random_state=42)
            model.fit(X_train)
            clusters = model.predict(X_test)
            st.write("Cluster Assignments:", clusters)
        else:
            st.write("Please select a valid model.")
        
        if problem_type in ["Regression", "Classification"]:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        
        if problem_type == "Regression":
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            metrics_df = pd.DataFrame({"Metric": ["MSE", "RMSE", "R2 Score", "MAE"], "Value": [mse, rmse, r2, mae]})
            st.write("### Regression Metrics")
            st.table(metrics_df)
        
        elif problem_type == "Classification":
            accuracy = accuracy_score(y_test, predictions)
            st.write("Accuracy:", accuracy)
            cm = confusion_matrix(y_test, predictions)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(plt)