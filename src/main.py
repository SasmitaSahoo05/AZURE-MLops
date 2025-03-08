import mlflow
import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from read_data import read_data 
from register_dataset import register_dataset
from data_prep import data_prep
from hyperparameter_tuning import random_search_cv_with_kfold
from model_training import train_with_best_hyperparameters
from get_ml_client import get_ml_client


def main():
    
    """Main function of the script."""

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_name",required=True)
    #parser.add_argument("--version", required=True)
    #parser.add_argument("--ml_client", required=True)
    parser.add_argument("--train_data_name", required=True)
    parser.add_argument("--test_data_name", required=True)
    parser.add_argument("--target_column", required=True)
    parser.add_argument("--registered_model_name", required=True)
    args = parser.parse_args()

    # Start Logging
    #mlflow.set_tracking_uri("azureml://azmlworkspace")
    #mlflow.set_experiment("credit_mlflow_exp")
    
    # Use a with context for MLflow run
    with mlflow.start_run() as run:

    #mlflow.start_run()
    
        version = None
        KVUri = "https://azmlworkspace3750565306.vault.azure.net"
        ml_client = get_ml_client(KVUri)

        # Enable autologging
        mlflow.sklearn.autolog()

        # data preparation
        print("Starting data prep...")
        data_prep(args.input_data_name,version,ml_client,KVUri)

        print("Starting hyperparametertuning...")
        #Hyperparameter tuning 
        best_params = random_search_cv_with_kfold(args.train_data_name, args.target_column, ml_client, KVUri,
        version, n_splits=3, n_iter=10)

        #Model Training
        print("Starting model training...")
        model = train_with_best_hyperparameters(args.train_data_name, args.test_data_name,ml_client,
        KVUri, args.target_column, best_params, version)

        # Registering the model to the workspace
        print("Registering the model via MLFlow")
        mlflow.sklearn.log_model(
            sk_model=model,
            registered_model_name=args.registered_model_name,
            artifact_path=args.registered_model_name,
        )

        # Saving the model to a file
        mlflow.sklearn.save_model(
            sk_model=model,
            path=os.path.join(args.registered_model_name, "trained_model"),
        )

        # Stop Logging
        mlflow.end_run()

if __name__ == "__main__":
    main()
         
        

