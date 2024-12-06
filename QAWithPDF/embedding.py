from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model,document):
    print(model)
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("")
        
         # Ensure the storage directory exists
        storage_dir = './storage'
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        #service_context = ServiceContext.from_defaults(llm=model,embed_model=gemini_embed_model, chunk_size=800, chunk_overlap=20)
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        Settings.num_output = 512
        Settings.context_window = 3900
        
        logging.info("")
        index = VectorStoreIndex.from_documents(documents=document, show_progress=True)
        #index = VectorStoreIndex.from_documents(document,service_context=service_context)
        index.storage_context.persist(persist_dir=storage_dir)
        
        logging.info("")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        raise customexception(e,sys)