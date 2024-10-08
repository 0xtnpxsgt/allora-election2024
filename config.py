import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")
model_file_path = os.path.join(data_base_path, "model")

TOKEN = ['R','D']
MODEL = os.getenv("MODEL")

