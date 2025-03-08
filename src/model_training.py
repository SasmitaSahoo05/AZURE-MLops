import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from read_data import read_data

def train_with_best_hyperparameters(
    train_data_name, 
    test_data_name,
    ml_client,
    KVUri,
    target_column,
    best_params, 
    version=None
    
):
    """
    Train a model using the best hyperparameters from a previous MLflow run and log the trained model to MLflow.

    Args:
        train_data_name (str): Name of the registered training dataset.
        test_data_name (str): Name of the registered testing dataset.
        target_column (str): Target column in the datasets.
        version (str): Version of the dataset (default is None to use the latest version).
        ml_client (obj): for authentication
    """
    # Load train and test datasets
    train_df = read_data(train_data_name,version,ml_client,KVUri)
    test_df = read_data(test_data_name,version,ml_client,KVUri)

    print(f"Training dataset shape: {train_df.shape}")
    print(f"Testing dataset shape: {test_df.shape}")
    print(f"Target distribution in train: \n{train_df[target_column].value_counts()}")
    print(f"Target distribution in test: \n{test_df[target_column].value_counts()}")

    # Split datasets into features and target
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Convert parameter values to the correct data type
    best_params = {
        "n_estimators": int(best_params["n_estimators"]),
        "max_depth": int(best_params["max_depth"]) if best_params["max_depth"] != "None" else None,
        "min_samples_split": int(best_params["min_samples_split"]),
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
    }

    # Train the model using the best hyperparameters
    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"F1 Score on test set: {f1}")

    # Log the trained model and metrics to MLflow
    mlflow.log_metric("test_f1_score", f1)
    #mlflow.sklearn.log_model(model, artifact_path="trained_model")

    print("Trained model logged to MLflow.")

    return model
