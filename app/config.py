import os
from dotenv import load_dotenv

# Ensure .env is loaded from the project root even when scripts run from subfolders
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# MongoDB connection string (use environment variable for security)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# Flask app configuration
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False") == "True"
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")

# API Configuration
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 300))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))

# Model Configuration
DEFAULT_MODEL_CONFIG = {
    "model_name": "default",
    "model_id": None,
    "weights_id": None,
    "results": {},
    "is_active": False,
    "preprocess_id": None,
    "postprocess_id": None,
}

# Preprocessing Configuration
PREPROCESS_TIMEOUT = int(os.getenv("PREPROCESS_TIMEOUT", 30))

# Database Configuration
DB_CONNECTION_TIMEOUT = int(os.getenv("DB_CONNECTION_TIMEOUT", 10000))
DB_MAX_POOL_SIZE = int(os.getenv("DB_MAX_POOL_SIZE", 50))
DB_MIN_POOL_SIZE = int(os.getenv("DB_MIN_POOL_SIZE", 10))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "False") == "True"
