import pandas as pd
from datetime import datetime
from src import (
    build_pipeline, create_aggregate_features,
    calculate_rfm, scale_rfm, cluster_customers,
    assign_risk_label, integrate_risk_label
)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/raw/data.csv')

    # Task 3: Feature Engineering
    agg_df = create_aggregate_features(df)
    pipe = build_pipeline()
    processed_df = pipe.fit_transform(df)
    final_df = pd.merge(
    agg_df,
    processed_df,
    on='CustomerId',
    how='left',
    suffixes=('_agg', '_proc')  # custom suffixes to avoid conflicts
)


    # Task 4: RFM + Clustering + Proxy Target
    snapshot_date = datetime(2025, 7, 1)  # Replace with actual snapshot if needed
    rfm_df = calculate_rfm(df.copy(), snapshot_date)
    rfm_scaled = scale_rfm(rfm_df)
    clusters = cluster_customers(rfm_scaled)
    risk_labels = assign_risk_label(rfm_df, clusters)

    # Merge proxy target into final processed data
    final_df = integrate_risk_label(final_df, risk_labels)

    # Save processed data
    final_df.to_csv('data/raw/data.csv', index=False)
    print("✅ Processed data with risk labels saved.")
from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import mlflow.pyfunc
import numpy as np

app = FastAPI()

# Load the best model from MLflow Model Registry
try:
    model = mlflow.pyfunc.load_model("models:/CreditRiskModel/Production")
except Exception as e:
    print("❌ Could not load model from MLflow Model Registry:", e)
    model = None

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: CustomerFeatures):
    if model is None:
        return {"risk_probability": None, "is_high_risk": None, "error": "Model not available. Please check MLflow registry."}
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)
    prob = model.predict_proba(input_array)[0][1]  # Probability of high risk
    return PredictionResponse(risk_probability=prob, is_high_risk=int(prob > 0.5))
