from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 1. PATH RESOLUTION
# Ensures we always find the chroma_db folder at the root of your project
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

def _get_vector_store():
    """Helper function to load the local Chroma DB."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Vector database not found at {CHROMA_DIR}. Please run build_vector.py first.")
        
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

# --- THE AGENT TOOL ---
# In ADK, standard Python functions with clear Type Hints and Docstrings 
# are automatically converted into tools the LLM can use.

def search_research_papers(query: str) -> str:
    """
    Searches the local arXiv vector database for semantic concepts, methodologies, 
    and general abstract summaries. 
    
    Use this tool whenever the user asks about:
    - Research methodologies or techniques.
    - Summaries of papers.
    - Challenges, limitations, or results mentioned in research.
    - General concepts related to AI, quantum computing, trading, etc.
    """
    print(f"\n[Tool Execution] Searching Vector DB for: '{query}'")
    
    try:
        vectordb = _get_vector_store()
        
        # Retrieve the top 4 most relevant chunks
        docs = vectordb.similarity_search(query, k=4)
        
        if not docs:
            return "No relevant documents found in the database for this query."
            
        # Build the context string
        context_parts = []
        for i, d in enumerate(docs, start=1):
            title = d.metadata.get('title', 'Unknown Title')
            content = d.page_content.replace('\n', ' ')
            context_parts.append(f"[Document {i} - {title}]: {content}")
            
        return "\n\n".join(context_parts)
        
    except Exception as e:
        return f"Error executing vector search: {str(e)}"