"""Module for querying indexed LinkedIn profile data."""

import logging
from typing import Any

from llama_index.core import PromptTemplate, VectorStoreIndex, get_response_synthesizer

import config
from modules.llm_interface import create_ollama_llm

logger = logging.getLogger(__name__)


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generates interesting facts about the person's career or education.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.

    Returns:
        String containing interesting facts about the person.
    """
    try:
        # Create LLM for generating facts
        ollama_llm = create_ollama_llm(temperature=0.0, max_new_tokens=500, decoding_method="sample")

        # Create prompt template
        facts_prompt = PromptTemplate(template=config.INITIAL_FACTS_TEMPLATE)

        # Create query engine
        query_engine = index.as_query_engine(
            streaming=False, similarity_top_k=config.SIMILARITY_TOP_K, llm=ollama_llm, text_qa_template=facts_prompt
        )

        # Execute the query
        query = "Provide three interesting facts about this person's career or education."
        response = query_engine.query(query)

        # Return the facts
        return response.response
    except Exception as e:
        logger.error(f"Error in generate_initial_facts: {e}")
        return "Failed to generate initial facts."


def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    """Answers the user's question using the vector database and the LLM.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
        user_query: The user's question.

    Returns:
        Response object containing the answer to the user's question.
    """
    try:
        # Create LLM for answering questions
        ollama_llm = create_ollama_llm(temperature=0.0, max_new_tokens=250, decoding_method="greedy")

        # Create prompt template
        question_prompt = PromptTemplate(template=config.USER_QUESTION_TEMPLATE)

        # Retrieve relevant nodes
        base_retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        source_nodes = base_retriever.retrieve(user_query)

        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(llm=ollama_llm, text_qa_template=question_prompt)

        # Generate the answer from the retrieved nodes
        answer = response_synthesizer.synthesize(user_query, nodes=source_nodes)
        return answer
    except Exception as e:
        logger.error(f"Error in answer_user_query: {e}")
        return "Failed to get an answer."
