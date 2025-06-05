import asyncio
import logging
import os
from typing import List, Tuple

import qdrant_client
from openai import OpenAI
import anthropic
import chainlit as cl
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

COLLECTION_NAME = "docs"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "claude-sonnet-4-20250514"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Clients ---
# Check for API keys (OpenAI client init will fail if not found)
if not os.getenv("OPENAI_API_KEY"):
    logging.error("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")
    # Optionally, you could exit or raise an error here for Chainlit to catch
if not os.getenv("ANTHROPIC_API_KEY"):
    logging.error("ANTHROPIC_API_KEY environment variable not found. Please set it in your .env file.")
    # Optionally, you could exit or raise an error here for Chainlit to catch

try:
    openai_client = OpenAI() # API key is read from OPENAI_API_KEY env var
    anthropic_client = anthropic.Anthropic() # API key is read from ANTHROPIC_API_KEY env var
    qdrant = qdrant_client.QdrantClient(url=QDRANT_URL)
except Exception as e:
    logging.error(f"Failed to initialize API clients: {e}")
    # This is a critical error for the app to function
    # Chainlit might hang or show an error if clients are not available.
    # Consider how to inform the user gracefully in the UI if this happens.
    raise


SYSTEM_PROMPT = (
    "You are an assistant that answers **only** from the provided <context>. "
    "If the answer cannot be found, simply reply with `I don't know`."
)

def check_qdrant_collection():
    """Checks if the Qdrant collection exists and has points."""
    try:
        collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
        logging.info(f"Qdrant collection '{COLLECTION_NAME}' status: {collection_info.status}, points count: {collection_info.points_count}")
        if collection_info.points_count is not None and collection_info.points_count > 0:
            return True, f"Collection '{COLLECTION_NAME}' found with {collection_info.points_count} points."
        else:
            return False, f"Collection '{COLLECTION_NAME}' is empty or points count is unavailable. Please run the indexing script."
    except Exception as e:
        # Handling cases where collection might not exist (often a 404 type error from client)
        logging.error(f"Could not get Qdrant collection info for '{COLLECTION_NAME}': {e}")
        return False, f"Could not connect to or find Qdrant collection '{COLLECTION_NAME}'. Please ensure Qdrant is running and the indexing script has been run."


def answer_query(query: str, top_k: int = 5) -> Tuple[str, List[str]]:
    """Retrieve context from Qdrant and ask Anthropic; return (answer, sources)."""
    logging.info(f"Embedding query: '{query[:50]}...'")
    try:
        v = openai_client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    except Exception as e:
        logging.error(f"Failed to embed query with OpenAI: {e}")
        return f"Error: Could not embed your question. {e}", []

    logging.info(f"Searching Qdrant collection '{COLLECTION_NAME}' for top {top_k} results.")
    try:
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=v,
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:
        logging.error(f"Failed to search Qdrant: {e}")
        return f"Error: Could not retrieve information from the knowledge base. {e}", []

    if not hits:
        logging.warning("No relevant documents found in Qdrant for the query.")
        return "I couldn't find any relevant information in the knowledge base to answer your question.", []

    sources = list(set([h.payload.get("source", "Unknown source") for h in hits if h.payload])) # Deduplicate sources
    context_texts = [h.payload.get("text", "") for h in hits if h.payload and h.payload.get("text")]
    
    if not context_texts:
        logging.warning("Retrieved documents from Qdrant but they contained no text in payload.")
        return "I found some documents, but they didn't contain text to form an answer.", sources

    context = "\n---\n".join(context_texts)
    user_block = f"<context>\n{context}\n</context>\n\nUser: {query}"

    logging.info(f"Sending query and context (approx {len(context)} chars) to Anthropic model {LLM_MODEL}.")
    try:
        response = anthropic_client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024, # Increased max_tokens for potentially longer answers
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_block}],
        )
        
        if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
            extracted_text = response.content[0].text
        else:
            logging.warning(f"Unexpected Anthropic response structure: {response.content}")
            extracted_text = "I received a response, but couldn't understand its format."

    except Exception as e:
        logging.error(f"Failed to get answer from Anthropic: {e}")
        return f"Error: Could not get an answer from the AI model. {e}", sources
        
    return extracted_text.strip(), sources


# --- Chainlit Callbacks ---

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    logging.info("New chat session started.")
    
    # Check Qdrant collection status
    collection_ok, status_message = await asyncio.get_event_loop().run_in_executor(None, check_qdrant_collection)
    
    if collection_ok:
        await cl.Message(
            f"👋 Welcome! I'm ready to answer questions about the content indexed from Ruter.no (via Qdrant). {status_message}"
        ).send()
        cl.user_session.set("ready_to_chat", True)
    else:
        await cl.Message(
            f"⚠️ Welcome! There seems to be an issue with the knowledge base. {status_message}"
        ).send()
        cl.user_session.set("ready_to_chat", False)
    
    # Optional: verify client initializations
    if not all([openai_client, anthropic_client, qdrant]):
        await cl.Message(
            "Critical error: One or more API clients failed to initialize. Please check server logs."
        ).send()
        cl.user_session.set("ready_to_chat", False)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""
    if not cl.user_session.get("ready_to_chat", False):
        await cl.Message(
            "I'm not ready to chat yet. Please check the initial messages or server logs for issues."
        ).send()
        return

    query = message.content.strip()
    logging.info(f"Received query: '{query}'")

    msg = cl.Message(content="") # Initialize an empty message to stream into
    await msg.send()

    try:
        answer_text, sources = await asyncio.get_event_loop().run_in_executor(
            None, answer_query, query
        )
        
        final_content = answer_text
        if sources:
            # Format sources to show full URL as clickable link text
            source_items = []
            for s_url in sources:
                if s_url.startswith("http"):
                    source_items.append(f"- [{s_url}]({s_url})")
                else:
                    source_items.append(f"- {s_url}") # Non-URL sources, if any
            
            if source_items:
                sources_markdown = "\n".join(source_items)
                final_content += f"\n\n**Sources**:\n{sources_markdown}"
        
        msg.content = final_content
        await msg.update()

    except Exception as e:
        logging.error(f"Error processing message: {e}")
        await cl.Message(content=f"❌ An unexpected error occurred: {e}").send() 