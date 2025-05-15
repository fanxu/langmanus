import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reasoning LLM configuration (for complex reasoning tasks)
REASONING_MODEL = os.getenv("REASONING_MODEL", "dashscope/qwen3-235b-a22b")

# Non-reasoning LLM configuration (for straightforward tasks)
BASIC_MODEL = os.getenv("BASIC_MODEL", "anthropic/claude-3-7-sonnet-20250219")

# Vision-language LLM configuration (for tasks requiring visual understanding)
VL_MODEL = os.getenv("VL_MODEL", "openai/gpt-4.1-mini")

# Chrome Instance configuration
CHROME_INSTANCE_PATH = os.getenv("CHROME_INSTANCE_PATH")
CHROME_HEADLESS = os.getenv("CHROME_HEADLESS", "False") == "True"
CHROME_PROXY_SERVER = os.getenv("CHROME_PROXY_SERVER")
CHROME_PROXY_USERNAME = os.getenv("CHROME_PROXY_USERNAME")
CHROME_PROXY_PASSWORD = os.getenv("CHROME_PROXY_PASSWORD")
