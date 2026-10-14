 # 🌾 Agriculture Knowledge Assistant (RAG)

An AI-powered question-answering app that lets users search and ask questions about
agricultural information drawn from both **PDF documents** and **web pages**.

Built for the RAG Buildathon, the app uses Retrieval-Augmented Generation (RAG): instead
of relying on the LLM's general knowledge, it first retrieves the most relevant passages
from a curated agriculture knowledge base, then generates an answer grounded only in
those passages, with the sources shown for every response.

### Key features
- **Multi-source ingestion:** agricultural PDFs (PyPDFLoader), static web pages
  (WebBaseLoader) and JavaScript-rendered pages (Playwright via AsyncChromiumLoader)
- **Vector search:** content is split into chunks, embedded with OpenAI embeddings and
  stored in a Chroma vector database
- **Grounded answers:** the LLM answers only from retrieved context and says so when
  the information isn't available
- **Source filter:** search across all sources, PDFs only, or web pages only
- **Transparent sources:** every answer lists the PDF (with page number) or web page it came from
- **Chat interface:** built with Streamlit, with streaming responses and chat history

### Tech stack
LangChain · OpenAI (GPT + text-embedding-3-small) · Chroma · Playwright · BeautifulSoup · Streamlit · Python
the tech stack (LangChain, WebBaseLoader, Playwright, Chroma, OpenAI, Streamlit)
how to run it: python3 -m pip install -r requirements.txt, python3 -m playwright install chromium, add your key to .env, python3 ingest.py, streamlit run app.py

## Screenshots
1. [Home page of Agriculture Knowledge Assistant](screenshots/Buildathon_Agriculture_screenshot1.png)
2. [chat screenshot of results-screenshot 1](screenshots/Buildathon_Agriculture_screenshot2.png)
3. [chat screenshot of results-screenshot 2](<screenshots/Buildathon_Agriculture_screenshot3 .png>)
4. [web source results screenshot](<screenshots/Buildathon_Source Screenshot1.png>)
5. [web and pdf source results screenshot](screenshots/Buildathon_Source_Screenshot2.png)


