import os
from dotenv import load_dotenv
from google.adk.agents import Agent

# Force Python to read your .env file
load_dotenv() 

# Import BOTH tools
from src.tools.vector import search_research_papers
from src.tools.graph import search_knowledge_graph

# --- THE HYBRID SUPERVISOR AGENT ---
root_agent = Agent(
    name="arxiv_research_supervisor",
    model="gemini-2.5-flash", 
    description="An intelligent supervisor agent that answers questions about arXiv research papers.",
    instruction=(
        "You are an expert academic research assistant. Your goal is to provide accurate, "
        "well-synthesized answers based strictly on the provided research papers.\n\n"
        "RULES:\n"
        "1. You have two tools at your disposal:\n"
        "   - `search_research_papers`: Use this Vector DB tool for finding general text, summaries, abstracts, and broad semantic concepts.\n"
        "   - `search_knowledge_graph`: Use this Graph DB tool for finding exact relationships, such as which papers use a specific methodology, or how concepts are linked.\n"
        "2. Analyze the user's question and intelligently decide which tool to use. You may use both if the question requires deep synthesis.\n"
        "3. Do not use outside knowledge. If the answer is not in the tools' outputs, explicitly state that.\n"
        "4. Cite your sources based on the tool output.\n"
    ),
    # The Agent now has two hands!
    tools=[search_research_papers, search_knowledge_graph] 
)

if __name__ == "__main__":
    print("Agent is configured and ready to run via the ADK CLI.")