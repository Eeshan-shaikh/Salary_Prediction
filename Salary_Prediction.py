import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Employee_Salaries_Large.csv")
    return df

df = load_data()

# Train & Save Model
@st.cache_resource
def train_and_save_model(data):
    X = data.drop("Salary", axis=1)
    y = data["Salary"]

    numeric_features = ["YearsExperience"]
    categorical_features = ["EducationLevel", "JobRole"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    #model: Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_val = r2_score(y_test, y_pred)

    # Save model
    joblib.dump(pipeline, "salary_model_gboost.pkl")
    return pipeline, rmse_val, r2_val

# Load or Train
if not os.path.exists("salary_model_gboost.pkl"):
    model, rmse, r2 = train_and_save_model(df)
else:
    model = joblib.load("salary_model_gboost.pkl")
    rmse, r2 = 0.0, 0.0  

# Streamlit Frontend
st.title("Employee Salary Prediction")

if rmse != 0.0 and r2 != 0.0:
    st.write(f"✅ Model trained with Gradient Boosting")
    st.write(f"📉 RMSE: {rmse:.2f}")
    st.write(f"📈 R² Score: {r2:.4f}")
else:
    st.write("✅ Model loaded from file (metrics not available unless retrained)")

st.write("Enter employee details to predict salary:")

years_exp = st.number_input("Years of Experience:", min_value=0.0, max_value=40.0, step=0.1)
education_level = st.selectbox("Education Level:", ["Bachelor", "Master", "PhD"])
job_role = st.selectbox("Job Role:", ["Data Scientist", "Software Engineer", "Manager", "Analyst", "HR"])

if st.button("Predict Salary"):
    input_df = pd.DataFrame({
        "YearsExperience": [years_exp],
        "EducationLevel": [education_level],
        "JobRole": [job_role]
    })
    predicted_salary = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ${predicted_salary:,.2f} per year")