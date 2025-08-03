from fastapi import FastAPI
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

model = load_model("model_lstm_volume.h5", compile=False)
scaler = joblib.load("scaler_volume.save")

@app.get("/predict")
def predict():
    try:
        query = """
        SELECT time_bin, direction, volume 
        FROM traffic_data 
        WHERE direction IN (0, 1)
        ORDER BY time_bin DESC 
        LIMIT 12
        """
        df = pd.read_sql(query, engine)

        if df.empty:
            return {"error": "Data tidak tersedia di database"}

        df['volume'] = df['volume'].astype(int)
        df['time_bin'] = pd.to_datetime(df['time_bin'])
        df['direction'] = df['direction'].astype(int)

        pivot_df = df.pivot_table(
            index='time_bin', 
            columns='direction', 
            values='volume', 
            aggfunc='sum'
        )
        
        pivot_df.columns = ['volume_keluar', 'volume_masuk']
        pivot_df = pivot_df.sort_index().fillna(0)

        if len(pivot_df) < 6:
            return {"error": "Jumlah data tidak cukup untuk prediksi (minimum 6 data)"}

        last_6 = pivot_df[-6:].values
        print("Data original:", last_6)

        scaled_last_6 = scaler.transform(last_6)
        print("Data scaled:", scaled_last_6)

        X_input = np.expand_dims(scaled_last_6, axis=0)
        print("Input shape:", X_input.shape)

        predicted_scaled = model.predict(X_input, verbose=0)
        print("Prediction scaled:", predicted_scaled)

        predicted = scaler.inverse_transform(predicted_scaled)
        print("Final prediction:", predicted)

        volume_masuk_pred, volume_keluar_pred = predicted[0]

        return {
            "volume_masuk_prediksi": round(float(volume_masuk_pred), 2),
            "volume_keluar_prediksi": round(float(volume_keluar_pred), 2)
        }

    except Exception as e:
        return {"error": str(e)}
