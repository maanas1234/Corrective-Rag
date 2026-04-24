from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local("vectorstore",embeddings=embeddings,allow_dangerous_deserialization=True)
retriver = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":4,"fetch_k":7})

