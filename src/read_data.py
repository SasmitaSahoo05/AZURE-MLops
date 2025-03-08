import mltable
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
import pandas as pd


def read_data(data_name,version,ml_client,KVUri):

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
    sas_token = kv_client.get_secret("sas-token").value
    workspace = kv_client.get_secret("AMLWORKSPACENAME").value
    subscription_id = kv_client.get_secret("AZURESUBSCRIPTIONID").value
    resource_group = kv_client.get_secret("AMLRESOURCEGROUP").value

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace
    )
    
    # Load the dataset
    if version is None:
        data_asset = ml_client.data.get(data_name,label="latest")
    else:
        data_asset = ml_client.data.get(data_name, version=version)
    
    try:
        #data_asset = ml_client.data.get(data_name, version=version)
        tbl = mltable.load(f'azureml:/{data_asset.id}')
        df = tbl.to_pandas_dataframe()
        print(f"Dataset read successfully.. Shape of Dataframe: {df.shape}")
    except Exception as e:
        # Create the full blob URL with SAS token
        blob_url = f"https://{storage_account}.blob.core.windows.net/{container}/{data_name}?{sas_token}"
        df = pd.read_csv(blob_url)
        print(f"Dataset read successfully.. Shape of Dataframe: {df.shape}")
    return df
