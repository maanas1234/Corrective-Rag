from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from query import prompt
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import os
from sentence_transformers import CrossEncoder
from __future__ import annotations

from typing import Iterable, Tuple, List

load_dotenv()


prompt = ChatPromptTemplate.from_template("""
You are a careful assistant. Use ONLY the provided context to answer.

Rules:
- Do not use outside knowledge.
- If the context does not contain enough information to answer, reply exactly: I don't know.
- Cite sources by adding (source: <source>) after each claim, using the document metadata 'source' when available.
- If the context is long, synthesize it into a direct answer (no rambling).

{chat_history_messages}

Context (multiple retrieved passages):
{context}

Question:
{question}

Answer (grounded in context, with sources):
""")


def build_llm():
    load_dotenv()
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key =os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-oss-120b:free"  # free tier model
    )

def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore() -> FAISS:
    embeddings = build_embeddings()
    return FAISS.load_local("vectorstore",embeddings=embeddings,allow_dangerous_deserialization=True)


def build_base_retriever():
    vectorstore = build_vectorstore()
    vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":4,"fetch_k":7})

chat_history_memory = ChatMessageHistory()
def get_messages(x):
    return chat_history_memory.messages

def build_retriever()->MultiQueryRetriever:
    base= build_base_retriever()
    llm -= build_llm()
    return MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    



#for i, doc in enumerate(check, 1):
 #   print(f"\n📄 Document {i}")
  #  print("-" * 50)
   # print(doc.page_content)

#print(len(check))

print(" ")
print(" ")
print(" ")
print(" ")

def rerank_with_scores(
    question: str,
    docs: List[Document],
    reranker: CrossEncoder,
    top_k: int = 4,
) -> Tuple[List[Document], List[float]]:
    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:top_k]]
    top_scores = [float(s) for _, s in ranked[:top_k]]
    return top_docs, top_scores


def format_docs(docs: Iterable[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        parts.append(f"(source: {src})\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def build_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

#cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

#context_runnable = RunnableLambda(lambda q:rerank(q,retriever_from_llm.invoke(q),cross,top_k=4))

def generate_answer(question:str, docs:list[Document])-> str:
    llm = build_llm()
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": format_docs(docs), "question": question})