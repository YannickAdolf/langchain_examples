import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

st.title('🦜🔗 Trainee PDF Search App')

openai_api_key = st.text_input(
    "Enter OpenAI API Key",
    type="password"
)

def generate_response(input_text):
  loader = PyPDFLoader("documents/duerr-ag-jahresabschluss-2022.pdf")
  pages = loader.load_and_split()
  faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(openai_api_key=openai_api_key))
  qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type="stuff", retriever=faiss_index.as_retriever())
  st.info(qa.run(input_text))

with st.form('my_form'):

  query = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
  submitted = st.form_submit_button('Submit')
  generate_response(query)


