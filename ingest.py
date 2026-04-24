from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import os


loader = DirectoryLoader('./data/notes', glob="**/*.md", loader_cls=TextLoader,loader_kwargs={"autodetect_encoding": True})
docs = loader.load()

print(f"Length{len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunks = splitter.split_documents(docs)
                                         
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore")


print("Done. Index saved to faiss_index/")
