from grader import grade_doc
from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search: bool


def grade_node(state):
    docs = state['documents']
    question = state['question']
    any_bad = any(grade_doc(question, d) == "no" for d in docs)
    return {"go_web": any_bad}

