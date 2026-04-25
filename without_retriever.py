from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from query import prompt
from langchain_core.prompts import ChatPromptTemplate
import os
from sentence_transformers import CrossEncoder

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



llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key =os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-oss-120b:free"  # free tier model
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



vectorstore = FAISS.load_local("vectorstore",embeddings=embeddings,allow_dangerous_deserialization=True)
base_retriver = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":4,"fetch_k":7})

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=base_retriver, 
    llm=llm
)

chat_history_memory = ChatMessageHistory()
def get_messages(x):
    return chat_history_memory.messages


check = retriever_from_llm.invoke("is suffering essential for humans?")



for i, doc in enumerate(check, 1):
    print(f"\n📄 Document {i}")
    print("-" * 50)
    print(doc.page_content)

print(len(check))

print(" ")
print(" ")
print(" ")
print(" ")


chain = ({"context": retriever_from_llm,"question":RunnablePassthrough(),"chat_history_messages":RunnableLambda(get_messages)} | prompt |  llm | StrOutputParser())
final_result = chain.invoke("what is the biggest problem in India?")
print(final_result)