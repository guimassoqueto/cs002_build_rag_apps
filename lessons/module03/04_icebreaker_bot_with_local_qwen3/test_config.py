import config

print("Initial Facts Template defined:", bool(config.INITIAL_FACTS_TEMPLATE))
print("User Question Template defined:", bool(config.USER_QUESTION_TEMPLATE))
print("LLM model:", config.LLM_MODEL_ID)
print("Embedding model:", config.EMBEDDING_MODEL_ID)
print("Chunk Size:", config.CHUNK_SIZE)
print("Similarity Top K:", config.SIMILARITY_TOP_K)
