from grader import grade_doc
from web_search import web_search
from web_search import web_scrape
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

def web_search_node(state):
    question = state['question']
    search_result = web_search(question,3)
    scraped_result = web_scrape(search_result)
    state['documents'].append(scraped_result)