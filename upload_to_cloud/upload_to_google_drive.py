import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

# 1. Setup Authentication
SCOPES = ['https://www.googleapis.com/auth/drive']

SERVICE_ACCOUNT_FILE = 'visual-monitoring-486318-1a131afe421a.json'  # Path to your JSON key file

creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

service = build('drive', 'v3', credentials=creds)

# 2. Define File Metadata
# 'parents' must be the ID of a folder shared with the service account email
folder_id = '0ACZDsFI2wW6tUk9PVA' 
file_metadata = {
    'name': 'test.txt',
    'parents': [folder_id]
}

# 3. Prepare the Media for Upload
media = MediaFileUpload('test.txt', 
                        mimetype='text/plain', 
                        resumable=True)

# 4. Execute the Upload
file = service.files().create(
    body=file_metadata,
    media_body=media,
    fields='id',
    supportsAllDrives=True  # MANDATORY for Shared Drives
).execute()