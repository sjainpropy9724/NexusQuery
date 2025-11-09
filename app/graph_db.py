import os
print("--- RELOADED graph_db.py ---")
from neo4j import GraphDatabase, basic_auth
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough, Runnable

# --- Connection Details ---
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "Swarit123")

# --- Initialize the Driver ---
# The driver is the main entry point to the database
try:
    driver = GraphDatabase.driver(
        NEO4J_URI, 
        auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    print("--- Neo4j Driver: Connection Verified ---")
    
except Exception as e:
    print(f"--- Neo4j Driver: FAILED to connect. Error: {e} ---")
    print("--- Please ensure the Neo4j Docker container is running. ---")
    driver = None

def get_driver():
    """
    Returns the initialized Neo4j driver.
    """
    if not driver:
        raise Exception("Neo4j driver not initialized. Check connection details.")
    return driver

def close_driver():
    """
    Closes the Neo4j driver connection.
    """
    if driver:
        driver.close()

# --- 1. Define LLM for Graph Extraction ---
# We use a Pydantic model to force the LLM to return structured JSON
class GraphEntity(BaseModel):
    subject: str = Field(description="The subject or entity, e.g., 'VIT', 'Dr. G. Viswanathan'")
    relationship: str = Field(description="The relationship between the subject and object, e.g., 'FOUNDED', 'IS_LOCATED_IN'")
    object: str = Field(description="The object or entity, e.g., 'Vellore Engineering College', 'Vellore'")

class GraphOutput(BaseModel):
    """The final structured output for the graph."""
    triples: List[GraphEntity]

# Initialize the LLM (Gemini 2.5 Flash is great for this)
# We chain it with .with_structured_output() to force JSON
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
structured_llm = llm.with_structured_output(GraphOutput)

# --- 2. Define the Graph Extraction Prompt ---
graph_extraction_prompt = ChatPromptTemplate.from_template("""
You are a data extraction expert. Your task is to extract entities and their relationships from the provided text.
Extract all relationships as simple (Subject, Relationship, Object) triples.
Do not extract relationships that are not explicitly stated in the text.
Use clear, all-caps relationship names like 'HAS_MOTTO', 'FOUNDED_BY', 'LOCATED_IN'.
Always use the full name for subjects and objects (e.g., "Vellore Institute of Technology" not "VIT").

Text to analyze:
{chunk}
""")

# Create the final chain that will do the extraction
graph_extraction_chain = graph_extraction_prompt | structured_llm

# --- 3. Function to Add Triples to Neo4j ---

def extract_and_store_graph(chunk: str, kb_id: str):
    """
    Extracts graph triples from a text chunk and stores them in Neo4j,
    linking them to the specific knowledge base.
    """
    print(f"--- Graph: Extracting from chunk for KB {kb_id} ---")
    
    # 1. Call the LLM chain to get structured triples
    try:
        response = graph_extraction_chain.invoke({"chunk": chunk})
        triples = response.triples
    except Exception as e:
        print(f"--- Graph: LLM extraction failed. Error: {e} ---")
        return
        
    if not triples:
        print(f"--- Graph: No triples found in chunk. ---")
        return

    # 2. Connect to Neo4j and store the triples
    # We use MERGE to create nodes/relationships if they don't exist
    # This query links all entities to the KnowledgeBase node
    cypher_query = """
    MERGE (kb:KnowledgeBase {id: $kb_id})
    WITH kb
    UNWIND $triples AS triple
    
    MERGE (s:Entity {name: triple.subject})
    MERGE (o:Entity {name: triple.object})
    
    MERGE (s)-[r:RELATIONSHIP {type: triple.relationship}]->(o)
    
    // Connect the entities to this specific KB
    MERGE (kb)-[:CONTAINS_ENTITY]->(s)
    MERGE (kb)-[:CONTAINS_ENTITY]->(o)
    """
    
    try:
        driver = get_driver()
        with driver.session() as session:
            session.run(
                cypher_query,
                kb_id=kb_id, 
                triples=[t.dict() for t in triples]
            )
        print(f"--- Graph: Stored {len(triples)} triples for KB {kb_id} ---")
        
    except Exception as e:
        print(f"--- Graph: Neo4j store failed. Error: {e} ---")

# --- 4. Text-to-Cypher Generation ---

# This string defines our graph schema for the LLM.
# It's simple: all nodes are :Entity, all relationships are :RELATIONSHIP
GRAPH_SCHEMA = """
Node properties:
- name: str
- id: str (This is the KnowledgeBase ID)

Node Labels:
- Entity
- KnowledgeBase

Relationship properties:
- type: str

Relationship Types:
- RELATIONSHIP
- CONTAINS_ENTITY

Relationships are structured as:
(:Entity)-[:RELATIONSHIP {{type: 'RELATION_NAME'}}]->(:Entity)
(:KnowledgeBase)-[:CONTAINS_ENTITY]->(:Entity)
"""

# This prompt teaches the LLM to write Cypher
cypher_prompt = ChatPromptTemplate.from_template("""
You are an expert Neo4j developer. Given a graph schema and a user question, generate a Cypher query.
Only use the schema provided. Do not use any node labels or relationship types not in the schema.
Only return the Cypher query and nothing else.

IMPORTANT: Your queries MUST be scoped to a single Knowledge Base.
Use this Cypher clause to start your query:
`MATCH (kb:KnowledgeBase {{id: $kb_id}})`
And this clause to find entities within that KB:
`MATCH (kb)-[:CONTAINS_ENTITY]->(entity_node)`

CRITICAL: All conceptual relationships are stored with the generic type :RELATIONSHIP.
To answer a question, you must find a path.
Instead of guessing an exact relationship type like 'AUTHORED_BY', you MUST write a flexible query that searches for keywords in the 'type' property.

Use `WHERE r.type CONTAINS '...'` for this.

Example for "Who authored X?":
`MATCH (kb:KnowledgeBase {{id: $kb_id}})-[:CONTAINS_ENTITY]->(s:Entity)`
`MATCH (kb)-[:CONTAINS_ENTITY]->(o:Entity {{name: 'X'}})`
`MATCH (s)-[r:RELATIONSHIP]->(o) WHERE r.type CONTAINS 'AUTHOR'`
`RETURN s.name`

Example for "What did X introduce?":
`MATCH (kb:KnowledgeBase {{id: $kb_id}})-[:CONTAINS_ENTITY]->(s:Entity {{name: 'X'}})`
`MATCH (kb)-[:CONTAINS_ENTITY]->(o:Entity)`
`MATCH (s)-[r:RELATIONSHIP]->(o) WHERE r.type CONTAINS 'INTRODUCE'`
`RETURN o.name`

Schema:
{schema}

User Question:
{question}

Cypher Query:
```cypher
""")

# This chain will generate the Cypher query
cypher_generation_chain = (
    RunnablePassthrough.assign(schema=lambda x: GRAPH_SCHEMA)
    | cypher_prompt
    | llm
)

def extract_and_store_graph_batch(chunks: List[str], kb_id: str):
    """
    Extracts graph triples from a BATCH of text chunks and stores them in Neo4j.
    This is much faster than one-by-one.
    """
    print(f"--- Graph: Starting BATCH extraction for {len(chunks)} chunks for KB {kb_id} ---")

    # 1. Prepare the batch of inputs
    inputs = [{"chunk": chunk} for chunk in chunks if chunk.strip()]
    if not inputs:
        print("--- Graph: No valid chunks to process. ---")
        return

    all_triples = []
    try:
        # 2. Call the LLM chain in batch
        # This makes multiple parallel API calls
        print(f"--- Graph: Calling LLM batch API for {len(inputs)} chunks... ---")
        responses = graph_extraction_chain.batch(inputs)
        print("--- Graph: LLM batch call complete. ---")

        # 3. Collect all triples from all responses
        for response in responses:
            if response and response.triples:
                all_triples.extend(response.triples)

    except Exception as e:
        print(f"--- Graph: LLM batch extraction failed. Error: {e} ---")
        # We might have partial data, but for now, let's just return
        return

    if not all_triples:
        print(f"--- Graph: No triples found in {len(chunks)} chunks. ---")
        return

    # 4. Connect to Neo4j and store ALL triples in ONE transaction
    cypher_query = """
    MERGE (kb:KnowledgeBase {id: $kb_id})
    WITH kb
    UNWIND $triples AS triple
    
    MERGE (s:Entity {name: triple.subject})
    MERGE (o:Entity {name: triple.object})
    
    MERGE (s)-[r:RELATIONSHIP {type: triple.relationship}]->(o)
    
    // Connect the entities to this specific KB
    MERGE (kb)-[:CONTAINS_ENTITY]->(s)
    MERGE (kb)-[:CONTAINS_ENTITY]->(o)
    """
    
    try:
        driver = get_driver()
        with driver.session() as session:
            session.run(
                cypher_query,
                kb_id=kb_id, 
                triples=[t.dict() for t in all_triples]
            )
        print(f"--- Graph: Stored {len(all_triples)} triples for KB {kb_id} in a single batch. ---")
        
    except Exception as e:
        print(f"--- Graph: Neo4j batch store failed. Error: {e} ---")

def get_cypher_query(question: str) -> str:
    """
    Generates a Cypher query from a natural language question.
    """
    print(f"--- Graph: Generating Cypher for: {question} ---")
    response = cypher_generation_chain.invoke({"question": question})
    query = response.content.strip().replace("```cypher", "").replace("```", "")
    print(f"--- Graph: Generated Cypher: {query} ---")
    return query


def query_graph_db(question: str, kb_id: str) -> List[Dict]:
    """
    Main function to query the graph.
    1. Generates a Cypher query.
    2. Executes the query.
    3. Returns the result.
    """
    try:
        # 1. Generate Cypher query
        cypher_query = get_cypher_query(question)
        
        # 2. Execute query
        driver = get_driver()
        with driver.session() as session:
            result = session.run(cypher_query, kb_id=kb_id)
            data = result.data()
            
        print(f"--- Graph: Query result: {data} ---")
        return data
        
    except Exception as e:
        print(f"--- Graph: Failed to query graph. Error: {e} ---")
        return [{"error": str(e)}]
    
# --- 5. Graph Data Exporter ---

def get_graph_data(kb_id: str) -> Dict:
    """
    Fetches all nodes and relationships for a specific KB
    and formats them for a visualization library.
    """
    print(f"--- Graph: Fetching graph data for KB {kb_id} ---")
    
    # This query finds all entities (s) and (o) connected by any relationship (r)
    # that are also linked to the specified KnowledgeBase (kb).
    cypher_query = """
    MATCH (kb:KnowledgeBase {id: $kb_id})-[:CONTAINS_ENTITY]->(s:Entity)
    MATCH (kb)-[:CONTAINS_ENTITY]->(o:Entity)
    MATCH (s)-[r:RELATIONSHIP]->(o)
    RETURN s.name AS source, r.type AS type, o.name AS target
    """
    
    nodes = set()
    edges = []
    
    try:
        driver = get_driver()
        with driver.session() as session:
            results = session.run(cypher_query, kb_id=kb_id)
            for record in results:
                source = record["source"]
                target = record["target"]
                
                # Add nodes to a set to avoid duplicates
                nodes.add(source)
                nodes.add(target)
                
                # Add edge
                edges.append({
                    "source": source,
                    "target": target,
                    "label": record["type"]
                })
                
        print(f"--- Graph: Found {len(nodes)} nodes and {len(edges)} edges. ---")
        
        # Format for visualization library
        # We need a list of nodes and a list of edges
        formatted_nodes = [{"id": node, "label": node, "shape": "dot", "size": 10} for node in nodes]
        
        return {"nodes": formatted_nodes, "edges": edges}

    except Exception as e:
        print(f"--- Graph: Failed to fetch graph data. Error: {e} ---")
        return {"nodes": [], "edges": []}

# --- Main function to test the connection ---
if __name__ == "__main__":
    if driver:
        try:
            with driver.session() as session:
                result = session.run("RETURN 'Hello, Neo4j!' AS message")
                print(f"Test query result: {result.single()['message']}")
        finally:
            close_driver()
            print("--- Neo4j Driver: Connection Closed ---")

