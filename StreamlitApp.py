import streamlit as st
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model

def main():
    st.set_page_config(page_title="QA with Documents")

    st.header("QA with Documents (Information Retrieval)")
    
    # File uploader
    doc = st.file_uploader("Upload your document", type=["pdf", "txt"])

    # User input for question
    user_question = st.text_input("Ask your question")

    if st.button("Submit & Process"):
        if doc is None:
            st.warning("Please upload a document.")
            return

        with st.spinner("Processing..."):
            # Load document
            document = load_data(doc)
            
            # Load model
            model = load_model()
            
            # Embed and create query engine
            query_engine = download_gemini_embedding(model, document)
            
            # Query engine response
            response = query_engine.query(user_question)
            
            # Display response
            st.write(response.response)

if __name__ == "__main__":
    main()
