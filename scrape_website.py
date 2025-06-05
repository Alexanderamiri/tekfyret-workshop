import logging
import os
import re
import uuid
import time
from collections import deque
from typing import List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import numpy as np
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# These can be overridden with environment variables for quick tuning.
MAX_PAGES = 50  # crawl limit per website
CHUNK_SIZE = 800  # in tokens (approx.)
CHUNK_OVERLAP = 40 # overlap between chunks
COLLECTION_NAME = "docs"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "claude-sonnet-4-20250514"
SITE_TO_INDEX = "https://ruter.no"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Utility Functions (Copied from original script) ---
def _normalize_url(url: str) -> str:
    url = url.split("#")[0].split("?")[0]
    if not os.path.splitext(url)[1] and not url.endswith("/"):
        url += "/"
    return url

def scrape_site(base_url: str, max_pages: int = MAX_PAGES) -> List[Tuple[str, str]]:
    logging.info(f"Starting to scrape {base_url}, up to {max_pages} pages.")
    session = requests.Session()
    seen, result = set(), []
    queue = deque([base_url])
    domain = urlparse(base_url).netloc
    pages_processed = 0

    def extract_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "nav", "aside"]):
            tag.decompose()
        text = "\n".join(t.strip() for t in soup.get_text("\n").splitlines() if t.strip())
        text = re.sub(r"\s{3,}", " ", text)
        return text

    while queue and len(result) < max_pages:
        url = queue.popleft()
        normalized_url = _normalize_url(url)
        
        if normalized_url in seen:
            continue
        
        current_domain = urlparse(normalized_url).netloc
        if current_domain != domain:
            logging.debug(f"Skipping external link: {normalized_url}")
            continue
            
        seen.add(normalized_url)
        pages_processed += 1
        logging.info(f"Processing page {pages_processed}/{max_pages if max_pages != float('inf') else 'unlimited'}: {normalized_url}")

        try:
            r = session.get(normalized_url, timeout=10, headers={'User-Agent': 'WorkshopCrawler/1.0'})
            r.raise_for_status() 
            if "text/html" not in r.headers.get("content-type", "").lower():
                logging.warning(f"Skipping non-HTML content at {normalized_url}")
                continue
            
            text = extract_text(r.text)
            if text:
                result.append((normalized_url, text))
            
            # Enqueue internal links
            soup = BeautifulSoup(r.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                link = urljoin(normalized_url, a_tag["href"])
                link_normalized = _normalize_url(link)
                if urlparse(link_normalized).netloc == domain and link_normalized not in seen and len(queue) + len(result) < max_pages * 2 : # Avoid overly large queue
                    queue.append(link_normalized)
        except requests.RequestException as e:
            logging.warning(f"Failed to fetch or process {normalized_url}: {e}")
            continue
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing {normalized_url}: {e}")
            continue
            
    logging.info(f"Scraping complete. Found {len(result)} pages with text.")
    return result

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    approx_tokens_per_word = 0.75 
    chunk_size_words = int(size / approx_tokens_per_word)
    overlap_words = int(overlap / approx_tokens_per_word)
    step = chunk_size_words - overlap_words
    if step <= 0: # Ensure step is positive, adjust overlap if necessary
        step = int(chunk_size_words / 2) if chunk_size_words > 1 else 1
        logging.warning(f"Adjusted step to {step} words due to large overlap/small chunk size.")

    chunks = [" ".join(words[i : i + chunk_size_words]) for i in range(0, len(words), step)]
    return [c for c in chunks if c.strip()]

# --- Qdrant and OpenAI Clients ---
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") # Though not used in this script, good to check

if not openai_api_key:
    logging.error("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")
    exit(1)
# if not anthropic_api_key:
#     logging.warning("ANTHROPIC_API_KEY environment variable not found. Not needed for indexing, but for chat app.")

openai_client = OpenAI(api_key=openai_api_key)
qdrant = qdrant_client.QdrantClient(url=QDRANT_URL)

# --- Core Indexing Logic ---
def _ensure_collection(vector_size: int = 1536):
    logging.info(f"Ensuring Qdrant collection '{COLLECTION_NAME}' exists with vector size {vector_size}.")
    try:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            timeout=20 # Increased timeout for robustness
        )
        logging.info(f"Collection '{COLLECTION_NAME}' recreated successfully.")
        time.sleep(2) # Increased delay
    except Exception as e:
        logging.error(f"Critical error during collection recreation for '{COLLECTION_NAME}': {e}")
        # Attempt to get collection info for more detailed error
        try:
            info = qdrant.get_collection(COLLECTION_NAME)
            logging.info(f"Existing collection info: {info}")
        except Exception as info_e:
            logging.error(f"Could not get info for collection '{COLLECTION_NAME}': {info_e}")
        raise

def index_website(base_url: str):
    logging.info(f"Starting indexing process for {base_url}...")
    pages = scrape_site(base_url, max_pages=MAX_PAGES)
    if not pages:
        logging.warning(f"No pages scraped from {base_url}. Nothing to index.")
        return

    texts, ids, payloads = [], [], []
    for url, page_text in pages:
        page_chunks = chunk_text(page_text)
        if not page_chunks:
            logging.warning(f"No text chunks generated for page {url}. Skipping.")
            continue
        for chunk_idx, chunk in enumerate(page_chunks):
            # Create a more unique ID including page URL and chunk index
            unique_id_str = f"{url}#{chunk_idx}"
            # Hash to a 64-bit integer compatible ID. Using simple hash for illustration.
            # Consider a more robust hashing or UUID for production.
            ids.append(abs(hash(unique_id_str)) % (2**63 -1) ) 
            texts.append(chunk)
            payloads.append({"source": url, "text": chunk})
    
    if not texts:
        logging.warning("No text chunks available for embedding after processing all pages.")
        return

    logging.info(f"Generated {len(texts)} text chunks for embedding.")

    # Batch embed
    vectors = []
    BATCH_SIZE_EMBEDDING = 100  # OpenAI batch limit can vary, 100 is usually safe
    for i in range(0, len(texts), BATCH_SIZE_EMBEDDING):
        batch_texts = texts[i : i + BATCH_SIZE_EMBEDDING]
        logging.info(f"Embedding batch {i//BATCH_SIZE_EMBEDDING + 1}/{(len(texts) + BATCH_SIZE_EMBEDDING - 1)//BATCH_SIZE_EMBEDDING}...")
        try:
            resp = openai_client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
            vectors.extend([d.embedding for d in resp.data])
        except Exception as e:
            logging.error(f"Error during OpenAI embedding request for batch starting at index {i}: {e}")
            # Decide if to continue with partial embeddings or stop
            logging.warning("Skipping this batch due to embedding error. Some data might not be indexed.")
            # Need to handle missing vectors if we skip a batch
            # For simplicity, this example might error out if vectors list is not full.
            # A robust implementation would pad with zeros or remove corresponding texts/ids/payloads.
            continue 

    if not vectors or len(vectors) != len(texts):
        logging.error(f"Embedding failed or produced incomplete results. Expected {len(texts)} vectors, got {len(vectors)}. Aborting upload.")
        return

    logging.info(f"Successfully embedded {len(vectors)} chunks.")
    
    _ensure_collection(vector_size=len(vectors[0]))
    
    logging.info(f"Uploading {len(ids)} points to Qdrant collection '{COLLECTION_NAME}'.")
    try:
        qdrant.upload_collection(
            collection_name=COLLECTION_NAME,
            ids=ids,
            vectors=np.array(vectors, dtype=np.float32),
            payload=payloads,
            batch_size=256, # Qdrant client batch size
            parallel=2 # Number of parallel workers for upload
        )
        logging.info("Successfully uploaded data to Qdrant.")
    except Exception as e:
        logging.error(f"Failed to upload data to Qdrant: {e}")
        # You might want to include more details from the exception if available
        # e.g. if e has a response attribute: logging.error(f"Qdrant server response: {e.response.text}")


if __name__ == "__main__":
    logging.info(f"--- Starting Ruter.no Indexing Script ---")
    
    # Check if Qdrant is available
    try:
        qdrant.get_collections()
        logging.info("Successfully connected to Qdrant.")
    except Exception as e:
        logging.error(f"Could not connect to Qdrant at {QDRANT_URL}. Please ensure Qdrant is running. Error: {e}")
        exit(1)
        
    index_website(SITE_TO_INDEX)
    logging.info(f"--- Indexing Script Finished ---") 