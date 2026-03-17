import os
import pandas as pd
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. PATH RESOLUTION
# This automatically finds the root 'arxiv-graphrag' folder, regardless of where you run the script from
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "papers.csv"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

def load_papers(path: Path):
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    
    # Validation check
    if not {"title", "abstract"}.issubset(df.columns):
        raise ValueError("CSV must contain 'title' and 'abstract' columns")
        
    docs = []
    for _, row in df.iterrows():
        content = f"Title: {row['title']}\nAbstract: {row['abstract']}"
        metadata = {"title": row["title"]}
        docs.append(Document(page_content=content, metadata=metadata))
        
    print(f"Loaded {len(docs)} papers.")
    return docs

def build_vector_store(docs, persist_dir: Path):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    print(f"Created {len(split_docs)} chunks.")

    print("Initializing embedding model (HuggingFace BGE)...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Building Chroma vector database at: {persist_dir}")
    vectordb = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    print("Success! Vector store built and saved.")
    return vectordb

if __name__ == "__main__":
    print("--- Starting Vector DB Ingestion ---")
    
    # Check if the user remembered to put the CSV in the data folder
    if not DATA_PATH.exists():
        print(f"Error: Could not find data file at {DATA_PATH}")
        print("Please place your 'papers.csv' inside the 'data' folder and try again.")
    else:
        documents = load_papers(DATA_PATH)
        build_vector_store(documents, CHROMA_DIR)
        print("--- Ingestion Complete ---")