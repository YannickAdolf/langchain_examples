from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


loader = PyPDFLoader("documents/duerr-ag-jahresabschluss-2022.pdf")
pages = loader.load_and_split()

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=faiss_index.as_retriever())

query = "Wie hat sich der Umsatz entwickelt"
qa.run(query)
print(qa.run(query))