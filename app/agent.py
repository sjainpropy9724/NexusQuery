print(" --- LOADING agent.py (Newest Version) ---")
import os
import pickle
from typing import List, Dict, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from . import processing, graph_db
from rank_bm25 import BM25Okapi
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- 1. Define the LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- 2. Define the State ---
class AgentState(TypedDict):
    """
    The state for our agent.
    """
    # Inputs (passed in at the start)
    query: str
    history: List[Dict[str, str]]
    retries: int
    faiss_path: str
    bm25_path: str
    chunks: List[str]

    # Populated by nodes
    context: str
    sources: Dict[str, str]
    memory_str: str
    answer: str
    last_tool_used: str  # <-- ADD THIS

# --- 3. Define the Nodes ---

def retrieve_context_node(state: AgentState):
    """
    This is our first node. It retrieves context using the function we just built.
    """
    print("--- AGENT: Retrieving Context ---")
    
    # Get all the necessary inputs from the current state
    query = state['query']
    history = state['history']
    faiss_path = state['faiss_path']
    bm25_path = state['bm25_path']
    chunks = state['chunks']

    try:
        # Load the indices from disk
        embeddings_model = processing.get_embedding_model()
        vector_db = FAISS.load_local(
            faiss_path,
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)

        # Call our new retrieval function from processing.py
        context_str, source_id_map, memory_str = processing.retrieve_and_rank_context(
            query, history, vector_db, bm25_index, chunks
        )
        
        # Return a dictionary to update the state
        return {
            "context": context_str,
            "sources": source_id_map,
            "memory_str": memory_str,
            "last_tool_used": "retrieve_context"
        }
    except Exception as e:
        print(f"Error in retrieve_context_node: {e}")
        # Return an error state
        return {
            "context": "Error retrieving context.",
            "sources": {},
            "memory_str": "",
            "answer": f"An error occurred during retrieval: {e}"
        }
    
# --- 4. Define Generation Node ---

# This is the prompt we perfected to get good, cited answers
generation_prompt_template = """
You are a helpful Q&A assistant. Your task is to answer the user's question based *strictly and only* on the provided numbered context sources.

Follow these instructions precisely:
1.  Read the user's question carefully.
2.  Analyze the [CONTEXT SOURCES] to find the relevant information.
3.  **Answer ONLY with information found in the sources.**
4.  If the sources contain relevant information, synthesize it into a concise answer.
5.  **If the sources contain partial information** (e.g., the user asks for "courses" and the context only mentions "over 70 programs"), you MUST provide that partial information. Do not apologize for missing details.
6.  **If the sources do not contain *any* relevant information** to answer the question, you MUST respond with: "The provided context does not contain information to answer this question." Do not add any other explanation.
7.  After writing the answer, add a section called `CITATIONS` on a new line.
8.  In the `CITATIONS` section, list the exact [SOURCE_ID] (e.g., [SOURCE_1]) that you used.
9.  If you did not find an answer (Rule 6), do not include a `CITATIONS` section.

---
[CONVERSATION HISTORY]
{memory_str}

---
[CONTEXT SOURCES]
{context}

---
[USER QUESTION]
{query}

[YOUR RESPONSE]
Answer:
"""

# Create a prompt template object
generation_prompt = ChatPromptTemplate.from_template(generation_prompt_template)

# Create a LangChain "chain" that pipes the prompt to the LLM
generation_chain = generation_prompt | llm

def generate_answer_node(state: AgentState):
    """
    This node generates an answer using the LLM.
    """
    print("--- AGENT: Generating Answer ---")
    
    # Get the inputs from the state
    query = state['query']
    context = state['context']
    memory_str = state['memory_str']
    
    # Check if the previous node failed
    if context == "Error retrieving context.":
        # Pass the error through without calling the LLM
        return {"answer": state['answer']}

    # Invoke the LLM chain
    try:
        response = generation_chain.invoke({
            "query": query,
            "context": context,
            "memory_str": memory_str
        })
        
        answer = response.content
        
        # Return a dictionary to update the state
        return {"answer": answer}
    
    except Exception as e:
        print(f"Error in generate_answer_node: {e}")
        return {"answer": f"An error occurred during generation: {e}"}
    

# --- 5. Define Re-phrasing Node ---

rephrase_prompt_template = """
You are a helpful AI assistant. A user's query failed to retrieve any relevant context.
Your task is to rephrase the user's original query into a new, more specific query that is more likely to find relevant documents.
Pay close attention to the chat history for context.

[CHAT HISTORY]
{history_str}

[ORIGINAL QUERY]
{query}

[YOUR RESPONSE]
Rephrased Query:
"""

rephrase_prompt = ChatPromptTemplate.from_template(rephrase_prompt_template)
rephrase_chain = rephrase_prompt | llm

def rephrase_query_node(state: AgentState):
    """
    This node rephrases the query if the previous attempt failed.
    """
    print("--- AGENT: Rephrasing Query ---")
    
    # Get the inputs from the state
    query = state['query']
    history = state['history']
    
    # Create a simple string from the history
    history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
    
    try:
        # Invoke the rephrase chain
        response = rephrase_chain.invoke({
            "history_str": history_str,
            "query": query
        })
        
        new_query = response.content.strip()
        print(f"--- AGENT: New Query: {new_query} ---")
        
        # Return a dictionary to update the state
        return {
            "query": new_query, # Overwrite the query with the new one
            "retries": state['retries'] + 1, # Increment the retry counter
            "last_tool_used": "rephrase_query"
        }
    
    except Exception as e:
        print(f"Error in rephrase_query_node: {e}")
        # If rephrasing fails, just end the loop
        return {"answer": f"An error occurred during rephrasing: {e}"}
    
def query_graph_node(state: AgentState):
    """
    This node queries the Neo4j graph for an answer.
    """
    print("--- AGENT: Querying Knowledge Graph ---")
    
    query = state['query']
    kb_id = state['faiss_path'].split(os.path.sep)[-2] # Get kb_id from path
    
    # 1. Call our new tool
    try:
        graph_data = graph_db.query_graph_db(query, kb_id)
    except Exception as e:
        print(f"--- Graph: Error in query_graph_node: {e} ---")
        return {
            "context": "Error querying the graph database.",
            "retries": state['retries'] + 1
        }

    # 2. Format the graph data as text context
    if graph_data and not graph_data[0].get("error"):
        # Convert the list of dictionaries into a simple string
        # This makes it easy for the LLM to read
        context_str = "Found the following information in the knowledge graph:\n"
        for item in graph_data:
            context_str += f"- {str(item)}\n"
        
        print(f"--- Graph: Found context: {context_str} ---")
        return {
            "context": context_str,
            "retries": state['retries'] + 1,
            "last_tool_used": "query_graph"
        }
    else:
        print("--- Graph: No information found in graph. ---")
        # No info found, return empty context
        return {
            "context": "",
            "retries": state['retries'] + 1,
            "last_tool_used": "query_graph"
        }
    

# --- 6. Define Conditional Edges ---

MAX_RETRIES = 3 # Let's give it 3 attempts total

def decide_next_step_router(state: AgentState):
    """
    This is the "brain" of our agent. It decides where to go next.
    It follows a specific strategy:
    1. Try vector search (retrieve_context_node).
    2. If that fails, try graph search (query_graph_node).
    3. If that fails, rephrase and go back to vector search (rephrase_query_node).
    4. If that fails, end.
    """
    print("--- AGENT: Deciding Next Step ---")
    
    answer = state['answer']
    retries = state['retries']
    last_tool = state.get('last_tool_used', 'retrieve_context')
    
    # Check if the answer is a "not found" response
    if "The provided context does not contain information" in answer:
        # If it failed, check if we have retries left
        if retries >= MAX_RETRIES:
            print("--- AGENT: Decision: END (Max retries reached) ---")
            return "end"
        
        # Strategy:
        if last_tool == 'retrieve_context' or last_tool == 'rephrase_query':
            # Last attempt was vector search. Let's try the graph.
            print(f"--- AGENT: Decision: QUERY_GRAPH (Attempt {retries + 1}) ---")
            return "query_graph"
        
        elif last_tool == 'query_graph':
            # Last attempt was graph search. Let's rephrase and try vector again.
            print(f"--- AGENT: Decision: REPHRASE (Attempt {retries + 1}) ---")
            return "rephrase"

    # If the answer is good, or if an error occurred, end the loop
    print("--- AGENT: Decision: END (Answer found) ---")
    return "end"


# --- 7. Assemble the Graph ---

print("--- AGENT: Compiling Graph ---")

# Initialize a new graph
workflow = StateGraph(AgentState)

# Add the nodes
# These are all the "tools" and "steps" our agent can take
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("rephrase_query", rephrase_query_node)
workflow.add_node("query_graph", query_graph_node)  

# Set the entry point
# This is the first node to be called
workflow.set_entry_point("retrieve_context")

# Add the edges
# These are the "arrows" connecting the nodes

# 1. The primary path: retrieve -> generate
workflow.add_edge("retrieve_context", "generate_answer")

# 2. The first loop: rephrase -> retrieve
workflow.add_edge("rephrase_query", "retrieve_context")

# 3. The new loop: query_graph -> generate
workflow.add_edge("query_graph", "generate_answer")

# 4. The conditional "brain"
# This router checks the answer and decides which path to take
workflow.add_conditional_edges(
    "generate_answer",  # The node we're branching from
    decide_next_step_router, # The function that makes the decision
    {
        # IF router returns "rephrase", go to "rephrase_query" node
        "rephrase": "rephrase_query",
        
        # IF router returns "query_graph", go to "query_graph" node
        "query_graph": "query_graph", # <-- ADDED NEW PATH
        
        # IF router returns "end", stop the graph
        "end": END
    }
)

# Compile the graph into a runnable app
app = workflow.compile()

print("--- AGENT: Graph Compiled Successfully ---")