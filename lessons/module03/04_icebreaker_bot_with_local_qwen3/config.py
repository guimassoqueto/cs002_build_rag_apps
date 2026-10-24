"""Configuration settings for the Icebreaker Bot."""

import os

from dotenv import load_dotenv

# Read secrets from the .env file next to this project
load_dotenv()

# Local Ollama settings
OLLAMA_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 300.0  # seconds; the first request also loads the model into memory

# Model settings
LLM_MODEL_ID = "qwen3-8b-rag"
EMBEDDING_MODEL_ID = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
CONTEXT_WINDOW = 8192  # tokens Ollama reserves for prompt + answer (num_ctx)

# ProxyCurl API settings
PROXYCURL_API_KEY = os.getenv("PROXYCURL_API_KEY", "")  # Set it in .env

# Mock data URL
MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"

# Query settings
SIMILARITY_TOP_K = 5
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
TOP_K = 50
TOP_P = 1.0

# Node settings
CHUNK_SIZE = 500

# LLM prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, using only the information provided in the context. If the answer is not available in the context, say "I don't know. The information is not available on the LinkedIn page."
"""
