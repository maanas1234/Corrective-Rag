from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os

class gradeDoc(BaseModel):
    score:str

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key =os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-oss-120b:free"  # free tier model
)

structured_llm = llm.with_structured_output(gradeDoc)

prompt = ChatPromptTemplate.from_template("""
You are a relevance grader.
Document: {document}
Question: {question}
Is this document relevant to answer the question? Answer 'yes' or 'no'.
""")

grader = prompt | structured_llm

def grade_doc(question:str, doc) -> str:
    result = grader.invoke({"question": question, "document": doc.page_content})
    return result.score

