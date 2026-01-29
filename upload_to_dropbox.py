import os
import dropbox
from dropbox.files import WriteMode
from dotenv import load_dotenv

load_dotenv()       # Load all env variables from .env

def upload_to_dropbox(local_path, dropbox_path, access_token):
    """ Upload a file to Dropbox  """

    # Note: local_path: path to the file on your device
    #       dropbox_path: path in your Dropbox (must start with '/')

    dbx = dropbox.Dropbox(access_token)

    with open(local_path, "rb") as f:
        dbx.files_upload(
            f.read(),
            dropbox_path,
            mode=WriteMode("overwrite")
        )

    print(f"Uploaded {local_path} → {dropbox_path}")


# Debug and Testing
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

if (ACCESS_TOKEN is None) :
    raise ValueError("ACCESS_TOKEN not found in the .env file")

# TO DO: Get the correct video name later
video_name = ""

local_path = f"/Documents/{video_name}"

# TO DO: Get the correct month from video name later
month = ""
dropbox_path = f"/Kshitij and Marwan - data/Bee's Recording 2026/{month}"
