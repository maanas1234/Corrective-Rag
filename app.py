import streamlit as st
from rag_graph import build_app

st.set_page_config(page_title="Second Brain CRAG", layout="wide")
st.title("Second Brain (CRAG)")

@st.cache_resource
def get_graph():
    return build_app()

graph = get_graph()

question = st.text_input("Ask a question", placeholder="e.g. What is the purpose of life?")
top_k = st.slider("Top K chunks", 1, 10, 4)

if st.button("Run", type="primary", disabled=not question.strip()):
    with st.spinner("Thinking..."):
        result = graph.invoke(
            {"question": question, "documents": [], "generation": "", "web_search": False},
        )

    st.subheader("Answer")
    st.write(result["generation"])

    st.subheader("Sources")
    for i, d in enumerate(result.get("documents", []), 1):
        src = d.metadata.get("source", "unknown")
        with st.expander(f"{i}. {src}", expanded=False):
            st.write(d.page_content)
