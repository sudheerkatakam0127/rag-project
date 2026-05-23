import os
import json
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# This tells Google which permissions we need — read-only access to Drive files
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive():
    """
    Handles Google OAuth login.
    First time: opens browser to ask permission.
    After that: uses saved token.json automatically.
    """
    creds = None

    # If we already logged in before, load the saved token
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If no valid token, do the browser login flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save token so we don't need to log in again next time
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def list_pdfs_in_folder(service, folder_id):
    """
    Lists all PDF files inside the given Google Drive folder.
    Returns a list of dicts with file id, name, and webViewLink.
    """
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"

    results = service.files().list(
        q=query,
        fields="files(id, name, webViewLink)"
    ).execute()

    files = results.get('files', [])
    print(f"Found {len(files)} PDF(s) in the folder.")
    return files


def download_pdf(service, file_id, file_name, download_folder='downloads'):
    """
    Downloads a single PDF from Drive and saves it locally.
    Returns the local file path.
    """
    # Create downloads folder if it doesn't exist
    os.makedirs(download_folder, exist_ok=True)

    local_path = os.path.join(download_folder, file_name)

    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading {file_name}: {int(status.progress() * 100)}%")

    return local_path


def load_pdfs_from_drive(folder_id):
    """
    Main function: authenticates, lists PDFs, downloads them all.
    Returns a list of dicts with local_path, file_name, drive_url.
    """
    service = authenticate_drive()
    files = list_pdfs_in_folder(service, folder_id)

    downloaded = []
    for f in files:
        local_path = download_pdf(service, f['id'], f['name'])
        downloaded.append({
            'file_name': f['name'],
            'local_path': local_path,
            'drive_url': f.get('webViewLink', '')
        })

    return downloaded