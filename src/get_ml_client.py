from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.ai.ml import MLClient

def get_ml_client(kvuri: str):
    # Initialize the Key Vault client with DefaultAzureCredential to access the Key Vault
    key_vault_credential = DefaultAzureCredential()
    kv_client = SecretClient(vault_url=kvuri, credential=key_vault_credential)
    
    # Retrieve credentials from Key Vault
    tenant_id = kv_client.get_secret("AZURETENANTID").value
    client_id = kv_client.get_secret("AZURECLIENTID").value
    client_secret = kv_client.get_secret("AZURECLIENTSECRET").value
    workspace = kv_client.get_secret("AMLWORKSPACENAME").value
    subscription_id = kv_client.get_secret("AZURESUBSCRIPTIONID").value
    resource_group = kv_client.get_secret("AMLRESOURCEGROUP").value
    
    # Register the environment to the Azure ML workspace
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace
    )
    
    return ml_client

