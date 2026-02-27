import os
import dropbox
from dropbox.oauth import DropboxOAuth2FlowNoRedirect
from dotenv import load_dotenv

""" Code to generate a refreshable token to upload to dropbox """

load_dotenv()       # Load all env variables

DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")

auth_flow = DropboxOAuth2FlowNoRedirect(
    DROPBOX_APP_KEY,
    DROPBOX_APP_SECRET,
    token_access_type='offline'            # <-- forces refresh token
)

authorize_url = auth_flow.start()
print("1. Go to:", authorize_url)
print("2. Click Allow")
print("3. Paste the authorization code here:")

auth_code = input("Enter code: ").strip()
result = auth_flow.finish(auth_code)

print("Please copy and save this refresh token as an env variable in the .env file")
print("REFRESH TOKEN:", result.refresh_token)
print("ACCESS TOKEN:", result.access_token)
