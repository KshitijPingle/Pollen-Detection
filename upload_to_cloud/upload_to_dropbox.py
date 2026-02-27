import os
from dotenv import load_dotenv
import dropbox
from dropbox.files import WriteMode

load_dotenv()

DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")

# Create a Dropbox client that auto-refreshes tokens forever
dbx = dropbox.Dropbox(
    oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
    app_key=DROPBOX_APP_KEY,
    app_secret=DROPBOX_APP_SECRET
)

def upload_to_dropbox(local_path, dropbox_path):
    """ Function to upload to DropBox
        Usage:
            local_path = file you want to upload
            dropbox_path = exact dropbox path you want to upload to
    """
    
    # Attach the file we want to upload to the dropbox path
    dropbox_path += "/" + local_path
    with open(local_path, "rb") as f:
        dbx.files_upload(
            f.read(),
            dropbox_path,
            mode=WriteMode("overwrite")
        )
    print(f"Uploaded {local_path} → {dropbox_path}")


# Testing
# file_name = "8_19_2024_2.mp4"
# dropbox_path = "/Kshitij and Marwant - data/Bee's Recording 2026/February"

# upload_to_dropbox(file_name, dropbox_path)