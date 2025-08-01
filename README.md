# diabetes-prediction
Machine Learning model for predicting diabetes, with FastAPI deployment as a RESTful API.

پروژه‌ای کامل برای تشخیص دیابت با استفاده از الگوریتم‌های یادگیری ماشین، شامل تحلیل داده‌ها، انتخاب بهترین مدل (LightGBM) و پیاده‌سازی API با FastAPI.

# Diabetes Prediction API

🎯 This project builds a machine learning model to predict diabetes using health-related features, and deploys the model as a RESTful API using FastAPI.


## 📁 Project Structure

```
diabetes-prediction-api/
│
├── app/                      # FastAPI app and saved model
│   ├── main.py
│   
│── model_lgbm_diabetes.pkl
│
│ 
├──model_art.ipynb/                 # Jupyter Notebooks for ML pipeline
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_best_model.ipynb
│
├── data/                     # Dataset (optional)
│   └── diabetes.csv
│
├── requirements.txt
│ 
├── README.md

```

---


## 📁 Project Structure
- `notebook/`: Data analysis and model selection
- `app/`: FastAPI app and trained model
- `data/`: Dataset (diabetes.csv)
- `requirements.txt`: Python dependencies

## 🔍 Notebooks Summary
1. **EDA**: Analyzing distribution, correlations, and outliers
2. **Modeling**: Testing multiple ML models with metrics
3. **Best Model**: Training & saving final LightGBM model

## 🚀 How to Run
```bash
# Install requirements
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload

curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d "{\"features\": [7,120,70,20,10,33.5,0.5,45,1100,3.5,0.708,0.5,22.5,True]}"


>curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"features\": [7,120,70,20,10,33.5,0.5,45,1100,3.5,0.708,0.5,22.5,True]}"
{"detail":[{"type":"json_invalid","loc":["body",65],"msg":"JSON decode error","input":{},"ctx":{"error":"Expecting value"}}]}
>curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"features\": [7, 120, 70, 20, 10, 33.5, 0.5, 45, 1100, 3.5, 0.708, 0.5, 22.5, true]}"
{"prediction":1}
```

## 📊 Model Info

- Selected Model: **LightGBMClassifier**
- Evaluation Metrics:
  - Accuracy: **0.8100**
  - F1 Score: **0.8190**
  - Cross-validated and tested
- Input Features: `[Pregnancies 	Glucose 	BloodPressure 	SkinThickness 	Insulin 	BMI 	DiabetesPedigreeFunction 	Age 	BMI_Age 	Glucose_BMI 	Insulin_to_Glucose 	Pregnancy_Rate 	DPF_Age 	AgeGroup_Middle]`

## 🌐 Tech Stack

- **Languages:** Python
- **ML Libraries:** Scikit-learn, LightGBM, Pandas, Seaborn
- **API Framework:** FastAPI
- **Deployment:** Uvicorn (local dev)

## 📄 License

MIT License — feel free to use or modify.

