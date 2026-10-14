import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import CHROMA_DIR, COLLECTION_NAME

load_dotenv()
st.set_page_config(page_title="Agri Knowledge Assistant", page_icon="🌾", layout="wide")


@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an agriculture expert assistant. Answer ONLY using the context below. "
     "If the answer is not in the context, say you don't have that information in the "
     "loaded documents. Keep answers clear and practical for farmers and agri officers."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


def source_label(doc):
    m = doc.metadata
    if m.get("source_type") == "pdf":
        return f"📄 {m.get('source')} (page {int(m.get('page', 0)) + 1})"
    return f"🌐 {m.get('source')} ({m.get('loader', 'web')})"


# ---------- Guard: vector DB must exist ----------
if not os.path.exists(CHROMA_DIR):
    st.error("Knowledge base not found. Run `python ingest.py` first.")
    st.stop()

vectorstore = get_vectorstore()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🔎 Search settings")
    scope = st.radio("Search in", ["All sources", "PDFs only", "Web pages only"])
    k = st.slider("Chunks to retrieve", 2, 10, 4)
    st.caption(f"Chunks in knowledge base: {vectorstore._collection.count()}")
    if st.button("Clear chat"):
        st.session_state.messages = []

search_kwargs = {"k": k}
if scope == "PDFs only":
    search_kwargs["filter"] = {"source_type": "pdf"}
elif scope == "Web pages only":
    search_kwargs["filter"] = {"source_type": "web"}
retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

# ---------- Main chat ----------
st.title("🌾 Agriculture Knowledge Assistant")
st.write("Ask anything about the agricultural PDFs and web pages loaded into the system.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

question = st.chat_input("e.g. What are the benefits of crop rotation?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        docs = retriever.invoke(question)
        if not docs:
            answer, sources = "I couldn't find anything relevant in the selected sources.", []
            st.markdown(answer)
        else:
            chain = PROMPT | get_llm() | StrOutputParser()
            answer = st.write_stream(
                chain.stream({"question": question, "context": format_docs(docs)})
            )
            sources = list(dict.fromkeys(source_label(d) for d in docs))  # unique, ordered
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )