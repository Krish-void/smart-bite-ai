import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import kagglehub

# 1. Fetch and Load Real Data
def load_real_data():
    print("Fetching Zomato and Swiggy datasets...")
    zomato_path = kagglehub.dataset_download("saurabhbadole/zomato-delivery-operations-analytics-dataset")
    swiggy_path = kagglehub.dataset_download("rrkcoder/swiggy-restaurants-dataset")
    
    df_zomato = pd.read_csv(os.path.join(zomato_path, "Zomato Dataset.csv"))
    
    # 2. Preprocess Zomato (Context)
    df = df_zomato[['Weather_conditions', 'Time_Orderd', 'City', 'Type_of_order']].copy()
    df.columns = ['weather', 'time_str', 'city', 'order_type']
    
    # Clean weather
    df['weather'] = df['weather'].fillna('Sunny').str.strip()
    
    # Extract hour
    def extract_hour(t):
        try:
            return int(str(t).split(':')[0])
        except:
            return 12
    df['time_hour'] = df['time_str'].apply(extract_hour)
    
    # --- ADVANCED FEATURE ENGINEERING ---
    # 1. Time Segment
    def get_time_segment(h):
        if 5 <= h < 11: return 'Morning'
        if 11 <= h < 16: return 'Lunch'
        if 16 <= h < 19: return 'Tea-Time'
        if 19 <= h < 23: return 'Dinner'
        return 'Late-Night'
    df['time_segment'] = df['time_hour'].apply(get_time_segment)
    
    # 2. Weather Intensity
    def get_weather_type(w):
        if w in ['Sunny', 'Sandstorms', 'Windy']: return 'Hot/Dry'
        if w in ['Stormy', 'Cloudy', 'Fog']: return 'Cool/Rainy'
        return 'Normal'
    df['weather_type'] = df['weather'].apply(get_weather_type)
    
    # --- SMARTER MAPPING LOGIC ---
    def map_dish(row):
        h = row['time_hour']
        w = row['weather']
        ot = row['order_type']
        
        # Priority 1: Weather + Time (Strong Signal)
        if w in ['Stormy', 'Cloudy'] and 18 <= h <= 22: return 'Biryani'
        if w == 'Sunny' and 12 <= h <= 16: return 'Chaat'
        if w in ['Fog', 'Windy'] and 7 <= h <= 10: return 'Coffee'
        
        # Priority 2: Time based
        if 12 <= h <= 15: return 'North Indian'
        if 16 <= h <= 18: return 'Chaat'
        if 19 <= h <= 22: return 'Biryani'
        if h >= 22 or h <= 4: return 'Pizza'
        
        return 'North Indian'

    df['recommended_dish'] = df.apply(map_dish, axis=1)
    
    os.makedirs('model/datasets', exist_ok=True)
    df.to_csv('model/datasets/processed_food_data.csv', index=False)
    return df

# 3. Preprocess and Train
def train():
    df = load_real_data()
    print(f"Loaded {len(df)} rows. Preprocessing...")
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['time_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time_hour'] / 24)
    
    # Encoding Features
    X_cat = df[['weather', 'city', 'time_segment', 'weather_type']]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X_cat)
    
    X = np.hstack([X_encoded, df[['hour_sin', 'hour_cos']].values])
    y = df['recommended_dish']
    
    print("Training Intelligent Model...")
    model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
    model.fit(X, y)
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/food_recommender_model.joblib')
    joblib.dump(encoder, 'model/encoder.joblib')
    print("Intelligent Model Saved!")

if __name__ == "__main__":
    train()
