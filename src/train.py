import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def load_data(path='data/raw/data.csv'):
    df = pd.read_csv(path)

    y = df['is_high_risk']
    
    X = df.drop(columns=[
        'is_high_risk',
        'CustomerId',
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId',
        'CurrencyCode', 'CountryCode', 'ProviderId',
        'ProductId', 'ChannelId'
    ], errors='ignore')

    # Drop non-numeric/object columns
    object_cols = X.select_dtypes(include='object').columns.tolist()
    if object_cols:
        print(f"Dropping non-numeric cols: {object_cols}")
        X = X.drop(columns=object_cols)

    X = X.fillna(X.median(numeric_only=True))  # Fill any residual NaNs

    return X, y

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, proba) if proba is not None else None

    return accuracy, precision, recall, f1, roc_auc

def main():
    mlflow.set_experiment("Credit Risk Modeling")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    param_grids = {
        "LogisticRegression": {
            'C': [0.1, 1, 10],
            'solver': ['liblinear']
        },
        "RandomForest": {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
        }
    }

    best_score = 0
    best_model = None
    best_model_name = None

    for name, model in models.items():
        print(f"Training {name}...")

        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc')
        grid.fit(X_train, y_train)

        accuracy, precision, recall, f1, roc_auc = evaluate_model(grid.best_estimator_, X_test, y_test)

        print(f"{name} performance:\n Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        with mlflow.start_run(run_name=name):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            })
            mlflow.sklearn.log_model(grid.best_estimator_, "model")

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = grid.best_estimator_
            best_model_name = name

    print(f"✅ Best model: {best_model_name} with ROC-AUC: {best_score:.4f}")

    # Register the best model to MLflow model registry
    with mlflow.start_run(run_name="register_best_model"):
        result = mlflow.sklearn.log_model(best_model, "best_model", registered_model_name="CreditRiskModel")
        # Promote the latest version to Production
        client = MlflowClient()
        latest_versions = client.get_latest_versions("CreditRiskModel", stages=["None"])
        if latest_versions:
            version = latest_versions[0].version
            client.transition_model_version_stage(
                name="CreditRiskModel",
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
        else:
            print("No model version found to promote.")

if __name__ == "__main__":
    main()
