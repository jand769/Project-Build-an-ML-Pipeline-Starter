import argparse
import logging
import os
import tempfile
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(args):
    logger.info("Downloading artifact")
    artifact_local_path = mlflow.artifacts.download_artifacts(args.trainval_artifact)

    trainval_data = pd.read_csv(os.path.join(artifact_local_path, "trainval_data.csv"))

    # Splitting data into X and y
    X = trainval_data.drop(columns=["price"])
    y = trainval_data["price"]

    # Splitting into train and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.random_seed, stratify=X[args.stratify_by] if args.stratify_by else None
    )

    logger.info("Building preprocessing pipeline")
    # Define the preprocessing pipeline
    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    logger.info("Building inference pipeline")
    # Create the full inference pipeline
    rf = RandomForestRegressor(**args.rf_config)
    sk_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ])

    logger.info("Training the model")
    sk_pipe.fit(X_train, y_train)

    logger.info("Evaluating the model")
    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"R2 Score: {r2}")

    # Log metrics to MLflow
    with mlflow.start_run():
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Save the pipeline as an artifact
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, "model.joblib")
            dump(sk_pipe, export_path)
            mlflow.log_artifact(export_path, artifact_path="model")

        # Log the model configuration
        mlflow.log_dict(args.rf_config, "rf_config.json")

    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest")

    parser.add_argument("--trainval_artifact", type=str, required=True, help="Input artifact containing the training/validation data")
    parser.add_argument("--val_size", type=float, required=True, help="Fraction of the dataset to use for validation")
    parser.add_argument("--random_seed", type=int, required=True, help="Seed for random number generation")
    parser.add_argument("--stratify_by", type=str, default=None, help="Column to use for stratification")
    parser.add_argument("--rf_config", type=str, required=True, help="Path to JSON file containing Random Forest parameters")
    parser.add_argument("--max_tfidf_features", type=int, required=True, help="Maximum number of features for TF-IDF vectorization")
    parser.add_argument("--output_artifact", type=str, required=True, help="Name for the output model artifact")

    args = parser.parse_args()

    # Load the Random Forest configuration
    with open(args.rf_config, "r") as f:
        args.rf_config = json.load(f)

    run(args)
