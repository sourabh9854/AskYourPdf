import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def load_openai_api_key():
    # Loading  the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Ensuring that API key is not empty or None
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    return api_key

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Check for OpenAI API key
    api_key = load_openai_api_key()
    
    # Extract the text
    if pdf is not None:
        text = extract_text_from_pdf(pdf)

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
            
            st.write(response)

if __name__ == '__main__':
    main()
