import mlflow
import mlflow.sklearn
from read_data import read_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import numpy as np


def random_search_cv_with_kfold(train_data_name, target_column, ml_client, KVUri,version=None, n_splits=3, n_iter=10):
    """
    Perform hyperparameter tuning using RandomizedSearchCV with k-fold cross-validation
    and log the best model and parameters in MLflow.

    Args:
        train_data_name (str): Name of the registered dataset.
        target_column (str): Target column in the dataset.
        version (str): Version of the dataset (default is None to use the latest version).
        ml_client (obj): client for authentication.
        n_splits (int): Number of folds for cross-validation.
        n_iter (int): Number of parameter combinations to try in RandomizedSearchCV.
    """
    # Load dataset
    df = read_data(train_data_name,version,ml_client,KVUri)
    print(df[target_column].value_counts())
    #print(df.head(10))
    

    # Split data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Define the parameter search space
    param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Create a Random Forest classifier
    rf_model = RandomForestClassifier(random_state=42)

    # Define k-fold cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Define RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=make_scorer(f1_score, average="macro"),
        cv=cv,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )

    # Perform hyperparameter tuning
    #with mlflow.start_run(run_id=run_id):
    print("Starting hyperparameter tuning...")
    random_search.fit(X, y)

    # Log best parameters and best score to MLflow
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print(f"Best Parameters: {best_params}")
    print(f"Best F1 Score: {best_score}")

    mlflow.log_params(best_params)
    mlflow.log_metric("mean_f1_score", best_score)

    # Log the best model to MLflow
    mlflow.sklearn.log_model(random_search.best_estimator_, artifact_path="best_hyp_model")

    print(f"Hyperparameter tuning complete. Best F1 Score: {best_score}")

    return best_params
