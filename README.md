# 🍔 Smart Food Recommender API

A production-ready RESTful API that provides personalized food dish recommendations using Machine Learning. It analyzes contextual factors like weather, time, and location to suggest the best meal.

## 🚀 Key Features

- **ML-Powered**: RandomForest classifier trained on contextual food preferences.
- **FastAPI**: High-performance, async backend with auto-generated Swagger docs.
- **Dockerized**: Easy to deploy on any cloud platform (Render, Heroku, AWS).
- **Cyclical Encoding**: Intelligent handle of time (0-23h) using sine/cosine transforms.
- **Production-Ready**: Input validation with Pydantic and automated health checks.

## 🛠 Tech Stack

- **Backend**: FastAPI
- **Machine Learning**: Scikit-learn
- **Data Handling**: Pandas, NumPy
- **Serialization**: Joblib
- **DevOps**: Docker, Uvicorn

## 🛠 Local Setup

### 1. Clone & Setup Venv
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python3 train_model.py
```

### 3. Run the API
```bash
uvicorn main:app --reload
```
API will be live at: `http://localhost:8000`

## 🐳 Running with Docker

```bash
docker build -t smart-food-api .
docker run -p 8000:8000 smart-food-api
```

## 📡 API Endpoints

### 1. Health Check
`GET /health`
- Verifies if the API and ML model are loaded correctly.

### 2. Get Recommendation
`POST /recommend`
- **Request Body**:
```json
{
  "weather": "rainy",
  "time_hour": 19,
  "city": "Mumbai"
}
```
- **Success Response**:
```json
{
  "recommended_dish": "Biryani",
  "confidence": 0.85
}
```

## 🏆 Why this impresses startups?

This project demonstrates **End-to-End ML Deployment** skills. Startups value engineers who can not only build models but also wrap them into scalable, well-documented APIs that solve real-world problems like decision fatigue.

---
Built with ❤️ for the Smart Food Tech era.
