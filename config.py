# All data-source settings live here, so you only edit this file to add sources.

PDF_FOLDER = "data/pdfs"

# Simple/static pages -> loaded with WebBaseLoader (fast, no browser)
STATIC_URLS = [
    "https://en.wikipedia.org/wiki/Agriculture_in_India",
    "https://en.wikipedia.org/wiki/Organic_farming",
    "https://en.wikipedia.org/wiki/Crop_rotation",
]

# JavaScript-heavy pages -> loaded with Playwright (real headless browser)
DYNAMIC_URLS = [
    "https://agritech.tnau.ac.in/",
]

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "agri_knowledge"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150