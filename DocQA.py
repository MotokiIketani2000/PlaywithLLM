import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("gemma model chat bot Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name = "gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
"""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embeding-001")
        st.session_state.loader = PyPDFLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS(st.session_state.final_documents, st.session_state.embeddings)



prompt1 = st.text_input("資料の何が知りたいですか?")

if st.button("ベクターストア作成"):
    vector_embedding()
    st.write("作成完了")

import time


if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(document_chain, retriever)

    start = time.process_time()
    response = retrieval_chain.invoke({"input:":prompt1})
    st.write(response['answer'])

# find the relivant chunks
    with st.expander("Document Similality Search"):
        for i, doc in enumerate(response['documents']):
            st.write(doc.page_content)
            st.write("------------------------------")















