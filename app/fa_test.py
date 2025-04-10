from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import weaviate
from weaviate.classes.config import Property, DataType
from langchain_weaviate.vectorstores import WeaviateVectorStore
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from fastapi.responses import JSONResponse
from .env import settings
import os
import uuid

app = FastAPI()

weaviate_cluster_url = settings.API_KEY["weaviate_cluster_url"]
weaviate_api_key = settings.API_KEY["weaviate_api_key"]
groq_api_key = settings.API_KEY["groq_api_key"]
huggingface_api_key = settings.API_KEY["huggingface_api_key"]

@app.post("/upload")
async def upload_pdf(pdf_file: UploadFile):
  current_dir = os.getcwd()
  content = await pdf_file.read()
  filename = f'{str(uuid.uuid4())}.pdf'
  with open(os.path.join(current_dir, filename), "wb") as fp:
    fp.write(content)
  
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len,
  )
  
  loader = PyPDFLoader(os.path.join(current_dir, filename))
  documents = loader.load()
  docs = text_splitter.split_documents(documents)

  weaviate_client = weaviate.connect_to_weaviate_cloud(
  cluster_url=weaviate_cluster_url,
  auth_credentials= weaviate.classes.init.Auth.api_key(api_key=weaviate_api_key),
  )
  
  embeddings = HuggingFaceEndpointEmbeddings(
    model="intfloat/multilingual-e5-large-instruct",
    task="feature-extraction",
    huggingfacehub_api_token=huggingface_api_key,
  )

  db = WeaviateVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    client=weaviate_client,
    index_name="PdfDocument"
  )
  
  weaviate_client.close()
  
  return JSONResponse(content={"filename": filename})

@app.post("/search")
def get_context_query(query: str):
  
  weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_cluster_url,
    auth_credentials= weaviate.classes.init.Auth.api_key(api_key=weaviate_api_key),
  )

  llm = ChatGroq(
    model = 'llama3-70b-8192',
    temperature = 0.7,
    max_tokens = 300,
    api_key=groq_api_key
  )

  embeddings = HuggingFaceEndpointEmbeddings(
    model="intfloat/multilingual-e5-large-instruct",
    task="feature-extraction",
    huggingfacehub_api_token=huggingface_api_key,
  )

  db = WeaviateVectorStore(client=weaviate_client, index_name="PdfDocument", text_key="text", embedding=embeddings)

  retriever = db.as_retriever()

  template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
  Question: {question}
  Context: {context}
  Answer:
  """

  prompt = ChatPromptTemplate.from_template(template)

  rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
  )
  
  result = rag_chain.invoke(query)
  
  weaviate_client.close()
  
  return JSONResponse(content={"chain_result": result})
