# project :- {Aws Bedrock + Langcian}
# pdf -> vector store -> langchain (LLM -> Aws Bedrock)
# step - 0:- Data ingestion -> pdf
# step - 1:- Prepare documents (Documents -> split into chunks -> create embeddigs (Amazon titan)-> Vector store(FAISS))
# step - 2:- Ask question (Question -> with similarity search go to Vector store -> revlevant chunks -> prompt -> LLM -> Answer)

import json
import os
import sys
import boto3
import streamlit as st

# we will be using titan embeddings model to generate embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock


# Data ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# vector embddings and vector store
from langchain_community.vectorstores import FAISS

# llm models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-image-v1", client=bedrock)

# data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # in our testing character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# vector embedding and vector store


def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")


def get_claude_llm():
    # create the anthropic model
    llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock,
                  model_kwargs={"max_tokens_to_sample": 300})
    return llm


def get_llama3_llm():
    # create the Llama3 model
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock,
                  model_kwargs={"max_gen_len": 512})
    return llm


prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use atleast summarize 
with 250 words with detailed explanations. If you don't know the answer,
Just say that you don't know, don't try to maeup an answer.
<context>
{context}
</context>
Question: {question}
Assistant : """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriver=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        retrun_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")
    
    user_question = st.text_input("Ask a question from the PDF Files")
    
    with st.sidebar:
        st.title("Update or create vector store:")
        
        if st.button("Vectors Update"):
           with st.spinner("Processing..."):
               docs=data_ingestion()
               get_vector_store(docs)
               st.success("Done")
        
    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings)
            llm=get_claude_llm
            
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")
            
    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings)
            llm=get_llama3_llm
            
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")