import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import httplib2


SCOPES = [
    "https://www.googleapis.com/auth/webmasters.readonly",
    "https://www.googleapis.com/auth/webmasters"]

creds = Credentials.from_authorized_user_file("token.json", scopes=SCOPES)
creds.refresh(Request())
print(creds.to_json())
service = build(serviceName="searchconsole", version="v1", credentials=creds)
KEY = "AIzaSyB93oZc9g4WCGqhdL5w7FKlHODMDT2ytQ0"
