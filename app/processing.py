import docx
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image
# from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import torch
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import os
import google.generativeai as genai
from functools import lru_cache
import fitz
from sentence_transformers import CrossEncoder

load_dotenv()
# print("API Key from env: ", os.getenv("GOOGLE_API_KEY"))
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Explicitly setting the Tesseract command path
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    print(f"Tesseract path might not be set correctly: {e}")

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Loads and returns the SentenceTransformerEmbeddings model,
    caching it after the first call.
    """
    print("Loading embeddings model for the first time...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_model = SentenceTransformerEmbeddings(
        model_name = "all-MiniLM-L6-v2",
        model_kwargs = {'device': device}
    )
    return embeddings_model

# --- Text Extraction Logic ---
def extract_text_from_pdf(pdf_path):
    # reader = PdfReader(pdf_path)
    # text = ""
    # for page in reader.pages:
    #     text += page.extract_text() or ""
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

def extract_text_from_url(url):
    response=  requests.get(url, verify=False)
    soup = BeautifulSoup(response.content, "html.parser")
    text = ' '.join(t.get_text(strip=True) for t in soup.find_all(['p', 'h1', 'h2', 'h3']))
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join(para.text for para in doc.paragraphs if para.text)

# --- Main processing functions ---
def get_text(source: str) -> str:
    """Takes source (file path or URL) and returns the extracted text"""
    if source.startswith(('http', 'https')) :
        return extract_text_from_url(source)
    elif source.endswith('.pdf'):
        return extract_text_from_pdf(source)
    elif source.endswith('.docx') :
        return extract_text_from_docx(source)
    elif source.endswith('.jpg', '.png', '.jpeg') :
        return extract_text_from_image(source)
    elif source.endswith('.txt'):
        with open(source, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type")

def chunk_text(text:str):
    """Splits a long text into smaller chunks"""
    if not isinstance(text, str):
        raise ValueError(f"Input to chunk_text must be a string, got {type(text)} instead.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    return text_splitter.split_text(text)

def build_hybrid_indices(chunks: List[str]):
    """
    Takes list of the text chunks and builds both FAISS and BM25 indices
    """
    print("Building hybrid indices...")

    # -- Build Semantic index (FAISS) --- (embeddings_model is Global)
    embeddings_model = get_embedding_model()
    vector_db = FAISS.from_texts(texts=chunks, embedding=embeddings_model)
   
    # -- Build Lexical Index (BM25)
    tokenized_chunks = [chunk.split(" ") for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    print("Indices built successfully.")
    return vector_db, bm25_index

def reciprocal_rank_fusion(results_lists: List[List[Dict]], k: int=60) -> List[Dict]:
    fused_scores = {}
    for doc_list in results_lists:
        for rank, doc in enumerate(doc_list):
            doc_content = doc['content']
            if doc_content not in fused_scores:
                fused_scores[doc_content] = {'score': 0, 'sources': set()}
            fused_scores[doc_content]['score'] += 1 / (rank + k)
            fused_scores[doc_content]['sources'].add(doc['source'])

    reranked_results = [
        {'content': content, 'score': data['score'], 'source': list(data['sources'])[0]}
        for content, data in fused_scores.items()
    ]

    return sorted(reranked_results, key=lambda x: x['score'], reverse=True)


def expand_query(query: str) -> List[str]:
    """
    Uses an LLM to expand a simple user query into several more specific queries.
    """
    prompt = f"""
    You are a helpful AI assistant. Your task is to expand a user's query into 3 more specific, high-quality search queries that are likely to yield better results from a vector database.
    
    Return ONLY the 3 queries, each on a new line. Do not add numbers or any other text.

    Original Query: {query}
    
    Expanded Queries:
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        # Split the response by newline to get a list of queries
        expanded_queries = response.text.strip().split('\n')
        # Also include the original query in the final list
        all_queries = [query] + expanded_queries
        print(f"Expanded queries: {all_queries}")
        return all_queries
    except Exception as e:
        print(f"Error during query expansion: {e}")
        return [query] # Fallback to original query on error
    

@lru_cache(maxsize=1)
def get_reranker_model():
    """Loads and returns the CrossEncoder model."""
    print("Loading reranker model for the first time...")
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def rerank_results(query: str, results: List[Dict]) -> List[Dict]:
    """Re-ranks a list of documents based on a query using a CrossEncoder."""
    reranker = get_reranker_model()
    # Create pairs of [query, document_content] for the model
    pairs = [(query, item['content']) for item in results]
    
    # Get the relevance scores
    scores = reranker.predict(pairs)
    
    # Add the scores to our results and sort
    for i in range(len(results)):
        results[i]['rerank_score'] = scores[i]
        
    return sorted(results, key=lambda x: x['rerank_score'], reverse=True)


def build_memory_index(history: List[Dict]):
    """Takes a list of chat messages and build a FAISS index"""
    if not history:
        return None

    # Considering that we only have the content of the messages
    history_texts = [f"{msg['role']}: {msg['content']}" for msg in history]
    embeddings = get_embedding_model()
    memory_vector_db = FAISS.from_texts(history_texts, embeddings)
    return memory_vector_db


def query_and_generate(query: str, history: List[Dict], vector_db: FAISS, bm25_index: BM25Okapi, chunks: List[str]):
    # Expand the user's query
    all_queries = expand_query(query)
    # ---------------------------

    # Perform searches for ALL queries
    all_semantic_results = []
    all_keyword_results = []

    # Perform searches
    for query in all_queries:
        tokenized_query = query.lower().split(" ")
        semantic_res = vector_db.similarity_search_with_score(query, k=3)
        all_semantic_results.extend(semantic_res)
        keywords_res = bm25_index.get_top_n(tokenized_query, chunks, n=3)
        all_keyword_results.extend(keywords_res)

    # Build and search memory index
    retrieved_memories = []
    if history:
        memory_index = build_memory_index(history)
        if memory_index:
            # Searching for the most relevant past conversations (exchanges)
            memory_results = memory_index.similarity_search(query, k=2)
            retrieved_memories = [doc.page_content for doc in memory_results]

    # Format results for fusion
    semantic_formatted = [{'content': doc.page_content, 'source': 'semantic'} for doc in all_semantic_results]
    keyword_formatted = [{'content': content, 'source': 'keyword'} for content in all_keyword_results]

    # Fuse the results
    fused_results = reciprocal_rank_fusion([semantic_formatted, keyword_formatted])

    # Re-rank the fused results
    reranked_results = rerank_results(query, fused_results)  # using the original query for precision

    # Construct context and generate answer
    top_k = 3
    context = "\n\n---\n\n".join([item['content'] for item in reranked_results[:top_k]])

    memory_str = "\n".join(retrieved_memories) if retrieved_memories else "No relevant conversation history found."
    prompt = f"""
    You are a helpful Q&A assistant. Use the following conversation history and context from documents to answer the user's new question. If the context does not contain the answer, state that.

    Conversation History: {memory_str}
    Context from documents and URLs: {context}

    Based on the history and the context, answer this question: {query}

    Answer:
    """

    try:
        # OPENAI Code
        # response = client.chat.completions.create(
        #     model = "gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.0,
        # )
        # answer = response.choices[0].message.content

        # Gemini Code
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Error generating answer from LLM: {e}"
    
    # 5. Prepare sources for the final response
    sources = [
        {"file_name": "Multiple Sources", "content": item['content'], "score": item['rerank_score']}
        for item in reranked_results[:top_k]
    ]

    return answer, sources