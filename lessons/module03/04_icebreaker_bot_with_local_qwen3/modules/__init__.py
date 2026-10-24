"""
Icebreaker Bot modules package.

This package contains the following modules:
- data_extraction: Functions for extracting LinkedIn profile data
- data_processing: Functions for processing and indexing LinkedIn profile data
- llm_interface: Functions for interfacing with the local Ollama models
- query_engine: Functions for querying indexed LinkedIn profile data
"""

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import create_vector_database, split_profile_data, verify_embeddings
from modules.llm_interface import change_llm_model, create_ollama_embedding, create_ollama_llm
from modules.query_engine import answer_user_query, generate_initial_facts

__all__ = [
    "extract_linkedin_profile",
    "split_profile_data",
    "create_vector_database",
    "verify_embeddings",
    "create_ollama_embedding",
    "create_ollama_llm",
    "change_llm_model",
    "generate_initial_facts",
    "answer_user_query",
]
