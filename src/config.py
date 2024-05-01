import os

from dotenv import load_dotenv

load_dotenv()

# Llama-API
LLAMA_API_TOKEN = os.getenv('LLAMA_API_TOKEN')
LLAMA_MODEL = os.getenv('LLAMA_MODEL', 'llama-7b-chat')

# Embedding model
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')

# Rerank model
RERANK_MODEL = os.getenv('RERANK_MODEL', 'colbert-ir/colbertv2.0')
