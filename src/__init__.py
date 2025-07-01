import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------
# Step 1: Aggregation
# --------------------------
def create_aggregate_features(df):
    agg_df = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std', 'count']
    }).reset_index()
    agg_df.columns = ['CustomerId', 'TotalAmount', 'AvgAmount', 'StdAmount', 'TxnCount']
    return agg_df

# --------------------------
# Step 2: Feature Extraction
# --------------------------
def extract_time_features(df):
    df = df.copy()
    df['TxnDate'] = pd.to_datetime(df[['TxnYear', 'TxnMonth', 'TxnDay']])
    df['TxnHour'] = df['TransactionStartTime'].dt.hour
    df['TxnDay'] = df['TransactionStartTime'].dt.day
    df['TxnMonth'] = df['TransactionStartTime'].dt.month
    df['TxnYear'] = df['TransactionStartTime'].dt.year
    return df.drop(columns=['TransactionStartTime'])

# --------------------------
# Step 3: Handle Missing
# --------------------------
def handle_missing(df):
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

# --------------------------
# Step 4: Encoding
# --------------------------
def encode_categoricals(df):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = ohe.fit_transform(df[['ProductCategory']])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['ProductCategory']))
    df = df.reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(['ProductCategory'], axis=1, inplace=True)
    return df

# --------------------------
# Step 5: Scaling
# --------------------------
def scale_numerical(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# --------------------------
# Step 6: Custom Preprocessor Class
# --------------------------
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        customer_ids = X[['CustomerId']]  # Save for merging later

        # REMOVE THIS: X = extract_time_features(X)
        X = handle_missing(X)
        # REMOVE THIS IF ProductCategory already encoded: X = encode_categoricals(X)

        X = scale_numerical(X, ['Amount', 'Value'])  # Confirm columns exist

        if 'CustomerId' in X.columns:
            X = X.drop(columns=['CustomerId'])

        return pd.concat([customer_ids.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

# --------------------------
# Step 7: Pipeline Builder
# --------------------------
def build_pipeline():
    pipe = Pipeline([
        ('preprocess', CustomPreprocessor())
    ])
    return pipe

# --------------------------
# Step 8: Runner
# --------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------
# Step 9: Calculate RFM Metrics
# -------------------------------
def calculate_rfm(df, snapshot_date):
    # Rename columns temporarily to match what pandas expects
    df = df.rename(columns={
        'TxnYear': 'year',
        'TxnMonth': 'month',
        'TxnDay': 'day'
    })

    df['TxnDate'] = pd.to_datetime(df[['year', 'month', 'day']])

    # Now compute RFM
    rfm = df.groupby('CustomerId').agg({
        'TxnDate': lambda x: (snapshot_date - x.max()).days,
        'CustomerId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TxnDate': 'Recency',
        'CustomerId': 'Frequency',
        'Amount': 'Monetary'
    }).reset_index()

    return rfm

# -------------------------------
# Step 10: Scale RFM
# -------------------------------
def scale_rfm(rfm_df):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    return rfm_scaled

# -------------------------------
# Step 11: Cluster Customers
# -------------------------------
def cluster_customers(rfm_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    return clusters

# -------------------------------
# Step 12: Label High-Risk Group
# -------------------------------
def assign_risk_label(rfm_df, clusters):
    rfm_df['Cluster'] = clusters
    cluster_stats = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats['Recency'].idxmax()  # most inactive group
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    return rfm_df[['CustomerId', 'is_high_risk']]

# -------------------------------
# Step 13: Merge with Processed
# -------------------------------
def integrate_risk_label(processed_df, rfm_with_risk):
    return pd.merge(processed_df, rfm_with_risk, on='CustomerId', how='left')
