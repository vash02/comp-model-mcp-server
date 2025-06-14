# config.py

import os

# OpenAI settings
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database path (SQLite)
DB_PATH = os.getenv("AGENT_DB_PATH", "mcp_server.db")

# Directory to save generated model code modules
GENERATED_MODELS_DIR = os.getenv("GENERATED_MODELS_DIR", "generated_models")

# Sandbox settings
# Maximum seconds to allow a single experiment subprocess to run
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "30"))