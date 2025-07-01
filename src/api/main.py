from src import build_pipeline, create_aggregate_features
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('data/raw/data.csv')
    agg_df = create_aggregate_features(df)
    pipe = build_pipeline()
    processed_df = pipe.fit_transform(df)
    final_df = pd.merge(agg_df, processed_df, on='CustomerId', how='left')
    final_df.to_csv('data/raw/data.csv', index=False)
    print("✅ Processed data saved.")
