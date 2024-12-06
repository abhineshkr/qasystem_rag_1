import sys
from exception import customexception
from logger import logging
from typing import List
from llama_index.core import SimpleDirectoryReader, Document
import os

def load_data(uploaded_file) -> List[Document]:
    """
    Load PDF documents from the uploaded file.

    Parameters:
    - uploaded_file: A file-like object from Streamlit's file uploader.

    Returns:
    - A list of loaded documents.
    """
    try:
        logging.info("Data loading started...")

        # Ensure ./Data directory exists
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)

        # Store the uploaded file in ./Data
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logging.info(f"File saved to {file_path}")

        # Use SimpleDirectoryReader to read from ./Data
        loader = SimpleDirectoryReader(input_dir=data_dir)
        documents = loader.load_data()

        logging.info("Data loading completed...")
        return documents

    except Exception as e:
        logging.error("Exception in loading data...")
        raise customexception(e, sys)
