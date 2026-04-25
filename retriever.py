from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()



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

check = retriever_from_llm.invoke("What will happen if we have AI driver?")
for i, doc in enumerate(check, 1):
    print(f"\n📄 Document {i}")
    print("-" * 50)
    print(doc.page_content)

print(len(check))