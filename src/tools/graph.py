import os
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

def search_knowledge_graph(query: str) -> str:
    """
    Searches the Neo4j Knowledge Graph to answer questions about relationships between entities.
    
    Use this tool when the user asks about:
    - Connections between specific papers, concepts, or methodologies.
    - "Which papers discuss concept X?"
    - "What methodologies are used in Paper Y?"
    - Any structural or relational questions across the research data.
    """
    print(f"\n[Tool Execution] Translating query to Cypher and searching Graph DB: '{query}'")
    
    try:
        graph = Neo4jGraph()
        llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
        
        chain = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=llm,
            verbose=True,
            allow_dangerous_requests=True,
            # THE FIX: Skip the buggy summarization step and return raw data!
            return_direct=True 
        )
        
        result = chain.invoke({"query": query})
        
        # We convert the raw database dictionary into a string so the Supervisor can read it
        return str(result.get("result", "No relevant relational data found in the graph."))
        
    except Exception as e:
        return f"Error executing graph search: {str(e)}"