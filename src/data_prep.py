import pandas as pd
import argparse
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from read_data import read_data
from register_dataset import register_dataset
import mlflow

def data_prep(input_data_name,version,ml_client,KVUri):
    # Parse input/output arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_data_name", type=str)
    # parser.add_argument("--version", type=str)
    # parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    
    # args = parser.parse_args()

    #mlflow.start_run(run_id=args.run_id)

    # Load the input DataFrame
    df = read_data(input_data_name,version,ml_client,KVUri)
    
    # Convert columns to integers
    df["person_emp_length"] = df["person_emp_length"].apply(lambda x: int(x) if pd.notnull(x) else x)
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].apply(lambda x: 1 if x == True else 0)

    # Impute numerical columns with median
    num_columns = df.select_dtypes(include=["number"]).columns.tolist()
    num_imputer = SimpleImputer(strategy="median")
    df[num_columns] = num_imputer.fit_transform(df[num_columns])

    # Impute categorical columns with mode
    cat_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_columns] = cat_imputer.fit_transform(df[cat_columns])

    # One-hot encode categorical columns
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop="first")  # Use sparse_output instead of sparse
    # Retrieve feature names using get_feature_names_out
    encoded_df = pd.DataFrame(encoder.fit_transform(df[cat_columns]), columns=encoder.get_feature_names_out(cat_columns), index=df.index)
    # Combine numerical and encoded categorical columns
    df = pd.concat([df[num_columns], encoded_df], axis=1)

    # Remove outliers using the five-number summary
    for column in num_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Define the target column
    target_column = "loan_status"

    # Standardize numerical columns
    # Exclude the target column from numerical columns
    num_columns_ex_target = [col for col in num_columns if col != target_column]
    scaler = StandardScaler()
    df[num_columns_ex_target] = scaler.fit_transform(df[num_columns_ex_target])

    # Log the transformations as artifacts
    joblib.dump(num_imputer, "num_imputer.joblib")
    joblib.dump(cat_imputer, "cat_imputer.joblib")
    joblib.dump(encoder, "encoder.joblib")
    joblib.dump(scaler, "scaler.joblib")

    # Log the artifacts in MLflow
    mlflow.log_artifact("num_imputer.joblib")
    mlflow.log_artifact("cat_imputer.joblib")
    mlflow.log_artifact("encoder.joblib")
    mlflow.log_artifact("scaler.joblib")

    # Train/Test split and registering dataset

    # Split the data into features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

    # Combine X_train and y_train into df_train
    df_train = pd.concat([X_train, y_train], axis=1)

    # Combine X_test and y_test into df_test
    df_test = pd.concat([X_test, y_test], axis=1)

    # Display the shapes of the resulting DataFrames
    print("Shape of df_train:", df_train.shape)
    print("Shape of df_test:", df_test.shape)

    #register train dataset
    register_dataset("azmlworkspace3750565306",df_train,"credit_train_dataset","credit_train_dataset") 

    #register test dataset
    register_dataset("azmlworkspace3750565306",df_test,"credit_test_dataset","credit_test_dataset")