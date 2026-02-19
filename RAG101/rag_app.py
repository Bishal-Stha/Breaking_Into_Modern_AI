import os
from dotenv import load_dotenv
from llama_index.llms.openrouter import OpenRouter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables from .env file
load_dotenv()

def run_openrouter_rag():
    # 2. CONFIGURE THE MODEL
    # You can pick any model from openrouter.ai/models
    # Examples: "meta-llama/llama-3-8b-instruct", "google/gemini-pro-1.5", etc.
    llm = OpenRouter(model="x-ai/grok-4.1-fast")
    
    # Set embedding model (using HuggingFace instead of OpenAI)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # We tell LlamaIndex to use OpenRouter as the global 'brain'
    Settings.llm = llm

    # 3. LOAD DATA
    print("Reading documents...")
    documents = SimpleDirectoryReader("./data").load_data()

    # 4. BUILD INDEX
    print("Indexing...")
    index = VectorStoreIndex.from_documents(documents)

    # 5. QUERY
    query_engine = index.as_query_engine()
    response = query_engine.query(input("Enter your prompt: "))

    print("\n--- OPENROUTER AI RESPONSE ---")
    print(response)

if __name__ == "__main__":
    run_openrouter_rag()