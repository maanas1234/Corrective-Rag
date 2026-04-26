from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

import os

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key =os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-oss-120b:free"  # free tier model
)

#structured_llm = llm.with_structured_output(gradeDoc)

prompt = ChatPromptTemplate.from_template("""
You are a relevance grader.
Document: {document}
Question: {question}
Answer with a single word: yes or no. No punctuation
Is this document relevant to answer the question? Answer 'yes' or 'no'.
""")

grader = prompt | llm | StrOutputParser()

from langchain_core.output_parsers import StrOutputParser
grader = prompt | llm | StrOutputParser()

def grade_doc(question: str, doc) -> str:
    text = grader.invoke(
        {"question": question, "document": doc.page_content}
    ).strip().lower()

    first = text.split()[0] if text else ""
    if first in ("yes", "no"):
        return first
    return "no"

