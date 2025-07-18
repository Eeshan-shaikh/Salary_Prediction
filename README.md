# 💼 Employee Salary Prediction App

This is a **Streamlit web application** that predicts employee salaries based on:
- ✅ Years of Experience  
- ✅ Education Level  
- ✅ Job Role  

It uses a **Gradient Boosting Regressor** trained on a synthetic large dataset with excellent accuracy.

---

## 🚀 Features
✔️ Interactive **Streamlit** user interface  
✔️ Model training with **scikit‑learn** pipeline  
✔️ Automatic preprocessing (StandardScaler + OneHotEncoder)  
✔️ Gradient Boosting with strong performance (R² ≈ 0.97)  
✔️ Easy to run locally or deploy on Streamlit Cloud  

---

## 📂 Project Structure
├── Salary_Prediction.py # Main Streamlit app  
├── Employee_Salaries_Large.csv # Training dataset  
├── requirements.txt # Python dependencies  
└── README.md # Project documentation

yaml
---
## ⚙️ Installation & Setup

1.**Clone the repository**
```bash
git clone https://github.com/Eeshan-shaikh/Salary_Prediction.git
cd Salary_Prediction
```

2.**Install dependencies**
```
pip install -r requirements.txt
```
3.**Run the app**
```
streamlit run Salary_Prediction.py
```

4.**📊 Dataset Logic**

The dataset Employee_Salaries_Large.csv is synthetically generated with the following logic:

Base salary = 50,000 + (YearsExperience * 5,000)
+ Education bonus (Master: +10,000 | PhD: +20,000)
+ Role bonus (Manager: +20,000 | Data Scientist: +15,000 | Software Engineer: +10,000 | Analyst: +5,000)
+ Random noise (±5,000)

5.**📈 Model Performance**

RMSE: ~5,038

R² Score: 0.9704

(Based on 80/20 train-test split.)

