import os
import glob
import shutil
from dotenv import load_dotenv

load_dotenv()  # load keys before LangChain imports read USER_AGENT

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    AsyncChromiumLoader,
)
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import (
    PDF_FOLDER, STATIC_URLS, DYNAMIC_URLS,
    CHROMA_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
)


def load_pdfs():
    docs = []
    for path in glob.glob(os.path.join(PDF_FOLDER, "*.pdf")):
        pages = PyPDFLoader(path).load()
        for p in pages:
            p.metadata["source_type"] = "pdf"
            p.metadata["source"] = os.path.basename(path)
        docs.extend(pages)
        print(f"  PDF: {os.path.basename(path)} -> {len(pages)} pages")
    return docs


def load_static_web():
    if not STATIC_URLS:
        return []
    loader = WebBaseLoader(STATIC_URLS)
    docs = loader.load()
    for d in docs:
        d.metadata["source_type"] = "web"
        d.metadata["loader"] = "WebBaseLoader"
    print(f"  WebBaseLoader: {len(docs)} pages")
    return docs


def load_dynamic_web():
    if not DYNAMIC_URLS:
        return []
    # Playwright opens each page in a headless Chromium browser (handles JavaScript)
    loader = AsyncChromiumLoader(DYNAMIC_URLS, user_agent=os.getenv("USER_AGENT"))
    html_docs = loader.load()
    # Convert raw HTML into clean readable text
    docs = Html2TextTransformer().transform_documents(html_docs)
    for d in docs:
        d.metadata["source_type"] = "web"
        d.metadata["loader"] = "Playwright"
    print(f"  Playwright: {len(docs)} pages")
    return docs


def main():
    print("1) Loading documents...")
    docs = load_pdfs() + load_static_web() + load_dynamic_web()
    docs = [d for d in docs if d.page_content.strip()]  # drop empty pages
    print(f"   Total documents: {len(docs)}")

    print("2) Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = filter_complex_metadata(splitter.split_documents(docs))
    print(f"   Total chunks: {len(chunks)}")

    print("3) Embedding and saving to Chroma...")
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)  # fresh rebuild, avoids duplicate chunks
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done! Vector DB saved in '{CHROMA_DIR}/'")


if __name__ == "__main__":
    main()