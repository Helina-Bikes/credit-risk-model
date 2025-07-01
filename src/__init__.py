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
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
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
        X = extract_time_features(X)
        X = handle_missing(X)
        X = encode_categoricals(X)
        X = scale_numerical(X, ['Amount', 'Value'])  # Ensure these columns exist
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
