"""Module for interfacing with the local Qwen3 models served by Ollama."""

import logging

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import config

logger = logging.getLogger(__name__)


def create_ollama_embedding() -> OllamaEmbedding:
    """Creates a local Qwen3 Embedding model, served by Ollama, for vector representation.

    Returns:
        OllamaEmbedding model.
    """
    ollama_embedding = OllamaEmbedding(
        model_name=config.EMBEDDING_MODEL_ID,
        base_url=config.OLLAMA_URL,
    )
    logger.info(f"Created Ollama Embedding model: {config.EMBEDDING_MODEL_ID}")
    return ollama_embedding


def create_ollama_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample",
) -> Ollama:
    """Creates a local Qwen3 LLM, served by Ollama, for generating responses.

    Args:
        temperature: Temperature for controlling randomness in generation (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.
        decoding_method: Decoding method to use (sample, greedy).

    Returns:
        Ollama LLM.
    """
    additional_kwargs = {
        "num_predict": max_new_tokens,
        "top_k": 1 if decoding_method == "greedy" else config.TOP_K,
        "top_p": config.TOP_P,
    }

    ollama_llm = Ollama(
        model=config.LLM_MODEL_ID,
        base_url=config.OLLAMA_URL,
        temperature=temperature,
        context_window=config.CONTEXT_WINDOW,
        request_timeout=config.REQUEST_TIMEOUT,
        thinking=False,
        additional_kwargs=additional_kwargs,
    )
    logger.info(f"Created Ollama LLM: {config.LLM_MODEL_ID}")
    return ollama_llm


def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.

    Args:
        new_model_id: New LLM model ID to use.
    """
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")
