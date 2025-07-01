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
