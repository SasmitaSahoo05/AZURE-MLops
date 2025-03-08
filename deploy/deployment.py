# deployment.py

from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from get_ml_client import get_ml_client
import argparse

def main():
   

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered_model_name",required=True)
    parser.add_argument("--endpoint_name", required=True)
    
    args = parser.parse_args()

    KVUri = "https://azmlworkspace3750565306.vault.azure.net"
    ml_client = get_ml_client(KVUri)
    
    
    # Get the latest version of the registered model
    latest_model_version = max(
        [int(m.version) for m in ml_client.models.list(name=args.registered_model_name)]
    )
    model = ml_client.models.get(name=args.registered_model_name, version=latest_model_version)

    # Create or update the online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=args.endpoint_name,
        description="This is an online endpoint",
        auth_mode="key",
        tags={"model_type": "randomforestclassifier"}
    )
    endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint {endpoint.name} provisioning state: {endpoint.provisioning_state}")

    # Create or update the blue deployment
    blue_deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=args.endpoint_name,
        model=model,
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )
    blue_deployment = ml_client.begin_create_or_update(blue_deployment).result()
    print(f"Deployment {blue_deployment.name} state: {blue_deployment.provisioning_state}")

    # # Test the blue deployment
    # response = ml_client.online_endpoints.invoke(
    #     endpoint_name=args.endpoint_name,
    #     request_file="./deploy/sample-request.json",
    #     deployment_name="blue"
    # )
    # print("Response from deployment test:", response)

if __name__ == "__main__":
    
    main()