"""
Azure Blob Storage — async upload.
Documents are stored at: {container}/{doc_id}/{filename}

Auth priority:
  1. Connection string (AZURE_STORAGE_CONNECTION_STRING set) — includes key, simple for dev
  2. Managed Identity / DefaultAzureCredential — requires AZURE_STORAGE_ACCOUNT_NAME
     and 'Storage Blob Data Contributor' role assigned in Azure IAM
"""
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import ContentSettings
from azure.identity.aio import DefaultAzureCredential
from core.config import settings


def _blob_service_client() -> BlobServiceClient:
    if settings.azure_storage_connection_string:
        return BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
    # Managed Identity path
    account_url = f"https://{settings.azure_storage_account_name}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())


async def upload_to_blob(
    file_bytes: bytes,
    blob_name: str,           # e.g. "{doc_id}/{filename.pdf}"
    content_type: str,
) -> str:
    """Upload bytes to Azure Blob Storage. Returns the blob URL."""
    async with _blob_service_client() as service_client:
        container_client = service_client.get_container_client(
            settings.azure_storage_container
        )

        # Create container if it doesn't already exist
        try:
            await container_client.create_container()
        except Exception:
            pass  # Container already exists — safe to ignore

        blob_client = container_client.get_blob_client(blob_name)
        await blob_client.upload_blob(
            file_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )

        # Return public-style URL (access controlled by SAS or private endpoint in prod)
        return blob_client.url
