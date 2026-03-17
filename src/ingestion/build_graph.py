import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

# Force Python to load our Neo4j and Gemini API keys from the .env file
load_dotenv()

# Path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "papers.csv"

def load_papers(path: Path):
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    if not {"title", "abstract"}.issubset(df.columns):
        raise ValueError("CSV must contain 'title' and 'abstract' columns")
    
    docs = []
    for _, row in df.iterrows():
        # We include the title in the content so the LLM knows the name of the 'Paper'
        content = f"Title: {row['title']}\nAbstract: {row['abstract']}"
        docs.append(Document(page_content=content))
    return docs

def build_knowledge_graph(docs):
    print("\n--- Connecting to Neo4j ---")
    # This automatically reads NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD from .env
    graph = Neo4jGraph() 
    
    print("--- Initializing Gemini Graph Extractor ---")
    # We use Flash because it is fast and excellent at following JSON/Schema instructions
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
    
    # Here we define the EXACT schema we want Gemini to find. 
    # If we don't constrain it, the graph will become a messy hairball.
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Paper", "Concept", "Methodology", "Task"],
        allowed_relationships=["DISCUSSES_CONCEPT", "USES_METHOD", "ADDRESSES_TASK"]
    )
    
    print(f"--- Extracting Graph Entities from {len(docs)} papers ---")
    print("This may take a minute depending on the size of your CSV...")
    
    # Gemini reads the text and converts it into Graph format
    graph_documents = llm_transformer.convert_to_graph_documents(docs)
    
    total_nodes = sum(len(doc.nodes) for doc in graph_documents)
    total_edges = sum(len(doc.relationships) for doc in graph_documents)
    print(f"Extraction Complete! Found {total_nodes} nodes and {total_edges} relationships.")
    
    print("--- Pushing to Neo4j AuraDB ---")
    graph.add_graph_documents(graph_documents)
    print("Success! Data is now live in your Neo4j database.")

if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"Error: Could not find data file at {DATA_PATH}")
    else:
        documents = load_papers(DATA_PATH)
        # We are slicing the list to just process the first 5 rows for this POC 
        # so you don't have to wait 10 minutes. Change to `documents` to process all.
        build_knowledge_graph(documents[:5])