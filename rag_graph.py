from __future__ import annotations
from functools import lru_cache
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from grader import grade_doc
from web_search import web_search, web_scrape

from retriever import (
    build_retriever,
    build_reranker,
    rerank_with_scores,
    generate_answer,
)

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search: bool

@lru_cache(maxsize=1)
def _retriever():
    return build_retriever()


@lru_cache(maxsize=1)
def _reranker():
    return build_reranker()




def retrieve_node(state:GraphState):
    docs = _retriever().invoke(state['question'])
    return{"documents":docs}

def rerank_node(state:GraphState):
    top_docs, _scores = rerank_with_scores(
        state["question"],
        state["documents"],
        _reranker(),
        top_k=4,
    )
    return{"documents":top_docs}
    

def generate_node(state: GraphState):
    answer = generate_answer(state["question"], state["documents"])
    return {"generation": answer}

def grade_node(state: GraphState):
    question = state["question"]
    docs = state["documents"]
    kept_docs = []
    any_bad = False
    for d in docs:
        verdict = grade_doc(question, d)  # expects "yes"/"no"
        if verdict == "yes":
            kept_docs.append(d)
        else:
            any_bad = True
    go_web = (len(kept_docs) == 0)
    return {"documents": kept_docs, "web_search": go_web}


def web_search_node(state:GraphState):
    question = state['question']
    search_result = web_search(question,3)
    scraped_result = web_scrape(search_result)
    docs = state['documents']
    docs.append(scraped_result)
    return {"documents":docs}

def route_after_grade(state: GraphState) -> str:
    return "web_search_node" if state["web_search"] else "rerank_node"

# add near the bottom of rag_graph.py

def build_app():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("grade_node", grade_node)
    graph.add_node("web_search_node", web_search_node)
    graph.add_node("rerank_node", rerank_node)
    graph.add_node("generate_node", generate_node)

    graph.add_edge("retrieve_node", "grade_node")
    graph.add_conditional_edges(
        "grade_node",
        route_after_grade,
        {"web_search_node": "web_search_node", "rerank_node": "rerank_node"},
    )
    graph.add_edge("web_search_node", "rerank_node")
    graph.add_edge("rerank_node", "generate_node")
    graph.add_edge("generate_node", END)

    graph.set_entry_point("retrieve_node")
    return graph.compile()

def pretty_print(result: dict, max_snippet_chars: int = 250):
        print("\n" + "=" * 80)
        print("Q:", result["question"])
        print("-" * 80)
        print("A:", result["generation"].strip())
        print("-" * 80)

        docs = result.get("documents", [])
        print(f"Sources ({len(docs)} chunks):")
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            snippet = d.page_content.replace("\n", " ").strip()[:max_snippet_chars]
            print(f"{i}. {src}")
            print(f"   {snippet}...")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    app = build_app()
    result = app.invoke({"question": "test", "documents": [], "generation": "", "web_search": False})
    pretty_print(result)

