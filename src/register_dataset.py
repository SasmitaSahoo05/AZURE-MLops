import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
import io
from datetime import datetime

def register_dataset(keyVaultName,df,blob_name,dataset_name):

    KVUri = f"https://{keyVaultName}.vault.azure.net"

    # Get the current timestamp
    current_timestamp = datetime.now()

    # Format the timestamp
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    print(formatted_timestamp)

    # Initialize the Key Vault client with DefaultAzureCredential to access the Key Vault
    key_vault_credential = DefaultAzureCredential()
    kv_client = SecretClient(vault_url=KVUri, credential=key_vault_credential)

    # Retrieve credentials from Key Vault
    tenant_id = kv_client.get_secret("AZURETENANTID").value
    client_id = kv_client.get_secret("AZURECLIENTID").value
    client_secret = kv_client.get_secret("AZURECLIENTSECRET").value

    # Retrieve Blob Storage credentials from Key Vault
    connection_string = kv_client.get_secret("blob-connection-string").value
    storage_account = kv_client.get_secret("storage-account").value
    container = kv_client.get_secret("container-name").value

    workspace = kv_client.get_secret("AMLWORKSPACENAME").value
    subscription_id = kv_client.get_secret("AZURESUBSCRIPTIONID").value
    resource_group = kv_client.get_secret("AMLRESOURCEGROUP").value

    # Create ClientSecretCredential with fetched credentials
    credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)

    # Connect to your Azure ML workspace
    ml_client = MLClient(credential=credential, subscription_id=subscription_id, \
                            resource_group_name=resource_group, 
                            workspace_name=workspace)


    # Convert DataFrame to CSV in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Upload to Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
    blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
        
    #Register the uploaded file as a URI file data asset
    my_data = Data(
        path=f"https://{storage_account}.blob.core.windows.net/{container}/{blob_name}",
        type=AssetTypes.URI_FILE,
        description="credit dataset for train/test",
        name=dataset_name,
        version= formatted_timestamp
    )

    ml_client.data.create_or_update(my_data)
    
    print(f"Dataset {dataset_name} has been registered successfully with version {formatted_timestamp}")
