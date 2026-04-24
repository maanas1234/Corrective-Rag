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


llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key =os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-oss-120b:free"  # free tier model
)



prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."
(("placeholder", "{chat_history_messages}")
Context:
{context}

Question: {question}

Answer:
""")


chat_history_memory = ChatMessageHistory()
def get_messages(x):
    return chat_history_memory.messages


chain = ({"context": retriver,"question":RunnablePassthrough(),"chat_history_messages":RunnableLambda(get_messages)} | prompt |  llm | StrOutputParser())




response = retriver.invoke("what will happen if we get Ai drivers?")
print(response)

print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")


result = chain.invoke("what if we have ai drivers?")
print(result[:500])