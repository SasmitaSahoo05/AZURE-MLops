from azure.ai.ml.entities import Environment
from azure.ai.ml import MLClient,Output,dsl
from azure.ai.ml import command, Input, Output
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
from src.get_ml_client import get_ml_client

import os


KVUri = os.getenv("KEY_VAULT_URI")

# Initialize the Key Vault client with DefaultAzureCredential to access the Key Vault
key_vault_credential = DefaultAzureCredential()
kv_client = SecretClient(vault_url=KVUri, credential=key_vault_credential)

# Retrieve credentials from Key Vault
tenant_id = kv_client.get_secret("AZURETENANTID").value
client_id = kv_client.get_secret("AZURECLIENTID").value
client_secret = kv_client.get_secret("AZURECLIENTSECRET").value

ml_client = get_ml_client(KVUri)

# Define and register the environment explicitly

pipeline_job_env = Environment(
    name="credit-risk-env",
    description="Environment for credit risk pipeline",
    conda_file="./dependencies/conda.yaml",  # Path to your conda file with dependencies
    image="mcr.microsoft.com/azureml/curated/python-sdk-v2:23")

pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

pipeline_job_env.environment_variables = {
        "AZURE_CLIENT_ID": client_id,  # Replace with your Azure AD app's Client ID
        "AZURE_TENANT_ID": tenant_id,  # Replace with your Azure AD Tenant ID
        "AZURE_CLIENT_SECRET": client_secret,  # Replace with your Azure AD app's Client Secret
    }

print(f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}")



# Define the training step
train_register = command(
    name="credit_risk_training",
    display_name="Credit Risk Model Training",
    description="Train the credit risk model and register it",
    inputs={
        "input_data_name": Input(type="string"),
        "train_data_name": Input(type="string"),
        "test_data_name": Input(type="string"),
        "target_column": Input(type="string"),
        "registered_model_name": Input(type="string"),
    },
    outputs={
        "model_output": Output(type="uri_folder"),
    },
    code="./src/",  # Location of source code
    command=(
        "python main.py "
        "--input_data_name ${{inputs.input_data_name}} "
        "--train_data_name ${{inputs.train_data_name}} "
        "--test_data_name ${{inputs.test_data_name}} "
        "--target_column ${{inputs.target_column}} "
        "--registered_model_name ${{inputs.registered_model_name}}"
    ),
    environment=pipeline_job_env,  # Define your environment name here
)

# Define the deployment step
deployment_step = command(
    name="credit_risk_deployment",
    depends_on=[train_register],
    display_name="Credit Risk Model Deployment",
    description="Deploy the credit risk model to an endpoint",
    inputs={
        "registered_model_name": Input(type="string"),
        "endpoint_name": Input(type="string"),
    },
    outputs={
        "deployment_status": Output(type="uri_folder"),
    },
    code=".",  # Location of the deployment script
    command=(
        "python deploy/deployment.py "
        "--registered_model_name ${{inputs.registered_model_name}} "
        "--endpoint_name ${{inputs.endpoint_name}}"
    ),
    environment=pipeline_job_env,  # Define your environment name here
)

# Define the pipeline function
@dsl.pipeline(
    compute="cpu-cluster",
    description="Credit Risk Model Training and Deployment Pipeline",
)
def credit_risk_pipeline():
    # Steps will run sequentially: training first, deployment second
    train_register_step = train_register(
        input_data_name="credit_risk_dataset",
        train_data_name="credit_train_dataset",
        test_data_name="credit_test_dataset",
        target_column="loan_status",
        registered_model_name="credit_risk_model",
    )

    train_register_step.name = "train_register_step"
    
    deploy_step = deployment_step(
        registered_model_name="credit_risk_model",
        endpoint_name="credit-risk-model-endpoint",
    )

    deploy_step.name = "deploy_step"

    deploy_step.depends_on = [train_register_step.name]
    
    return {
        "training_output": train_register_step.outputs["model_output"],  # Get model output
        "deployment_output": deploy_step.outputs["deployment_status"],  # Get deployment output
    }

# Create and submit the pipeline job
pipeline = credit_risk_pipeline()
pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    experiment_name="credit_risk_experiment",
    properties={"azureml.enable_app_insights": "true"} 
)
ml_client.jobs.stream(pipeline_job.name)

# response = ml_client.online_endpoints.invoke(
#         endpoint_name=args.endpoint_name,
#         request_file="./deploy/sample-request.json",
#         deployment_name="blue"
#     )
#     print("Response from deployment test:", response)

