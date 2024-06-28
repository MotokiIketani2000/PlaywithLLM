import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key = os.getenv("GENAI_API_KEY"))

#1
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
#2
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
#3
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversation_chain():
    prompt_template = """
    与えられた質問に対してできる限り正確に答えてください。詳細な情報も加えてください。もし
    質問に答えられなければ。「文章から判断できませんでした」と返答してください。間違っている
    答えを与えないでください。
    
    答え
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature=0.3)
    prompt = PromptTemplate(prompt=prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff", prompt = prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(moddel = "models/embedding-01")
    
    new_db = FAISS.load_local("faiss_index",embeddings)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversation_chain()
    response = chain({"input_documents":docs,"question": user_question}, return_only_outputs=

                     True)
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("ChatPDF")
    st.header("Chat with PDF using gemini")

    user_question = st.text_input("ask a question from the pdf fies")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your pdf files and click on the submit button", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Procesing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()


    


