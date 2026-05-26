import os
from dotenv import load_dotenv

load_dotenv()

# DB Settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Rate Limits
NEWS_API_CONCURRENCY = 5  # Max concurrent requests
NEWS_API_DELAY = 1.0      # Delay between requests in seconds
