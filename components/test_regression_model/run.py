import mlflow
import pandas as pd
from mlflow.pyfunc import load_model
from sklearn.metrics import mean_absolute_error, r2_score

def go(mlflow_model: str, test_dataset: str):
    # Download test dataset
    print(f"Fetching test dataset: {test_dataset}")
    test_data_path = mlflow.artifacts.download_artifacts(test_dataset)
    print(f"Downloaded test data to: {test_data_path}")

    # Load test data
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    # Load production model
    print(f"Fetching model: {mlflow_model}")
    model_path = mlflow.artifacts.download_artifacts(mlflow_model)
    model = load_model(model_path)

    # Perform inference
    print("Performing inference...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae}, R2: {r2}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test regression model")
    parser.add_argument("--mlflow_model", required=True, type=str)
    parser.add_argument("--test_dataset", required=True, type=str)
    args = parser.parse_args()
    go(args.mlflow_model, args.test_dataset)
