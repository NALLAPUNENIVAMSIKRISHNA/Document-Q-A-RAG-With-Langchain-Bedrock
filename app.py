import json
import os
import sys
import boto3
import streamlit as st

# Updated imports for Bedrock and BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_aws import BedrockEmbeddings

# Data ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector embeddings and vector store
from langchain_community.vectorstores import FAISS

# LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)

def data_ingestion():
    """Loads and splits PDF documents into chunks."""
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    """Creates a FAISS vector store from documents and saves it locally."""
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    """Loads the Claude LLM from AWS Bedrock."""
    return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock, model_kwargs={"max_tokens_to_sample": 300})

def get_llama3_llm():
    """Loads the Llama 3 LLM from AWS Bedrock."""
    return Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={"max_gen_len": 512})

# Prompt template for LLMs
prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use at least 250 words with detailed explanations. If you don't know the answer,
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
Question: {question}
Assistant: """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    """Generates a response from the LLM based on vector search."""
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    response = qa.invoke({"query": query})  # Use invoke() instead of run()
    answer = response["result"]  # Extract only the answer

    return answer


def main():
    """Main function for the Streamlit app."""
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")
    
    user_question = st.text_input("Ask a question from the PDF Files")
    
    with st.sidebar:
        st.title("Update or create vector store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    # Load FAISS index safely
    try:
        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        return
    
    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")
    
    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            llm = get_llama3_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
