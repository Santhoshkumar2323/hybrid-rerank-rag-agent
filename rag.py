import os
import sys
import time
import json
import math
import urllib.request
import warnings
from typing import List, Dict, Any, TypedDict
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq
from langgraph.graph import StateGraph, END


class RAGState(TypedDict):
    query: str
    vector_results: List[Dict[str, Any]]
    bm25_results: List[Dict[str, Any]]
    fused_candidates: List[Dict[str, Any]]
    reranked_candidates: List[Dict[str, Any]]
    max_score: float
    response: str


class AdvancedRAGPipeline:
    def __init__(self, db_path: str = "./local_chroma_db", collection_name: str = "pdf_books"):
        print("[SYSTEM] Initializing Advanced RAG Pipeline...")
        
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key or self.groq_api_key == "your_actual_api_key_here":
            print("[ERROR] Missing or invalid GROQ_API_KEY in .env file. Exiting.")
            sys.exit(1)
            
        self.groq_model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        self.threshold = float(os.getenv("RERANK_THRESHOLD", "0.35"))
        self.top_k = int(os.getenv("HYBRID_TOP_K", "10"))
        self.max_tokens = int(os.getenv("MAX_PROMPT_TOKENS", "4000"))
        
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        print(f"[SYSTEM] Configuration Loaded:")
        print(f"         - Primary LLM: Groq ({self.groq_model})")
        print(f"         - Fallback LLM: Local Ollama ({self.ollama_model})")
        print(f"         - RERANK_THRESHOLD: {self.threshold}")
        print(f"         - TPM SAFEGUARD: {self.max_tokens} Max Tokens")

        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-large-en-v1.5"
            )
            self.collection = self.client.get_collection(
                name=collection_name, 
                embedding_function=self.embedding_fn
            )
            total_chunks = self.collection.count()
            print(f"[SYSTEM] Connecting to local_chroma_db... ✔ ({total_chunks} chunks loaded)")
        except Exception as e:
            print(f"[ERROR] Failed to connect to ChromaDB: {e}")
            sys.exit(1)

        t0 = time.time()
        db_data = self.collection.get(include=["documents", "metadatas"])
        self.corpus_docs = []
        tokenized_corpus = []
        
        for i in range(len(db_data['documents'])):
            text = db_data['documents'][i]
            meta = db_data['metadatas'][i]
            doc_id = f"chunk_{i}"
            
            self.corpus_docs.append({"id": doc_id, "text": text, "meta": meta})
            tokenized_corpus.append(text.lower().split())
            
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[SYSTEM] Building BM25 Keyword Index in memory... ✔ ({time.time() - t0:.2f}s)")

        print("[SYSTEM] Loading Cross-Encoder (ms-marco-MiniLM-L-6-v2)... ✔")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.llm_client = Groq(api_key=self.groq_api_key)
        self.app = self._build_graph()
        print("[SYSTEM] Agentic State Machine Compiled. Ready.\n")

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.ollama_base_url}/api/chat"
        payload = json.dumps({
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1}
        }).encode("utf-8")
        
        req = urllib.request.Request(
            url, 
            data=payload, 
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("message", {}).get("content", "").strip()

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def node_retrieve(self, state: RAGState) -> RAGState:
        print("\n> [Node: Retrieve] Executing Hybrid Search...")
        query = state["query"]
        
        vector_res = self.collection.query(query_texts=[query], n_results=self.top_k)
        vector_candidates = []
        if vector_res['documents'] and vector_res['documents'][0]:
            for i in range(len(vector_res['documents'][0])):
                vector_candidates.append({
                    "id": vector_res['ids'][0][i],
                    "text": vector_res['documents'][0][i],
                    "meta": vector_res['metadatas'][0][i]
                })
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = bm25_scores.argsort()[::-1][:self.top_k]
        
        bm25_candidates = []
        for idx in top_n_indices:
            if bm25_scores[idx] > 0:
                bm25_candidates.append(self.corpus_docs[idx])
                
        return {"vector_results": vector_candidates, "bm25_results": bm25_candidates}

    def node_fuse_and_rerank(self, state: RAGState) -> RAGState:
        print("> [Node: Fuse] Applying Reciprocal Rank Fusion (RRF)...")
        unique_docs = {}
        
        def apply_rrf(candidates, weight=60):
            for rank, doc in enumerate(candidates):
                doc_id = doc["id"]
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = {"doc": doc, "score": 0.0}
                unique_docs[doc_id]["score"] += 1.0 / (rank + weight)

        apply_rrf(state["vector_results"])
        apply_rrf(state["bm25_results"])
        
        fused_list = sorted(unique_docs.values(), key=lambda x: x["score"], reverse=True)
        top_fused = [item["doc"] for item in fused_list[:5]]
        
        if not top_fused:
            return {"reranked_candidates": [], "max_score": 0.0}

        print("> [Node: Rerank] Cross-Encoder evaluating top 5 chunks...")
        query = state["query"]
        cross_inp = [[query, doc["text"]] for doc in top_fused]
        
        scores = self.cross_encoder.predict(cross_inp)
        reranked = []
        max_score = -999.0
        
        for i, doc in enumerate(top_fused):
            raw_score = float(scores[i])
            percent_score = 1.0 / (1.0 + math.exp(-raw_score))
            
            if percent_score > max_score:
                max_score = percent_score
            
            doc_meta = doc["meta"]
            status = "Approved" if percent_score >= self.threshold else "Rejected"
            print(f"  -> Chunk {doc_meta.get('chunk_index', '?')} Confidence: {percent_score:.2f} ({status})")
            
            if percent_score >= self.threshold:
                reranked.append({"text": doc["text"], "meta": doc["meta"], "score": percent_score})
                
        reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
        return {"reranked_candidates": reranked, "max_score": max_score}

    def route_evaluation(self, state: RAGState) -> str:
        max_score = state.get("max_score", 0.0)
        if max_score >= self.threshold:
            return "generate"
        else:
            print(f"> [Router] ALERT: Max confidence ({max_score:.2f}) is below threshold ({self.threshold}). Redirecting...")
            return "alternative"

    def node_generate(self, state: RAGState) -> RAGState:
        print(f"> [Node: Generate] Constructing prompt and managing tokens...")
        
        approved_chunks = state["reranked_candidates"]
        context_blocks = []
        citations = set()
        current_tokens = self._estimate_tokens(state["query"]) + 100 
        chunks_used = 0
        
        for doc in approved_chunks:
            chunk_tokens = self._estimate_tokens(doc['text'])
            if current_tokens + chunk_tokens > self.max_tokens:
                print(f"  -> Dropping remaining chunks to prevent exceeding {self.max_tokens} token limit.")
                break
                
            context_blocks.append(f"DOCUMENT EXTRACT:\n{doc['text']}")
            source_file = doc['meta'].get('source_file', 'Unknown')
            chunk_idx = doc['meta'].get('chunk_index', 'Unknown')
            citations.add(f"- {source_file} (Chunk {chunk_idx})")
            
            current_tokens += chunk_tokens
            chunks_used += 1
            
        print(f"  -> Selected Top {chunks_used} Chunks. Estimated Prompt Size: {current_tokens} Tokens.")
        
        context_str = "\n\n".join(context_blocks)
        citation_str = "\n".join(sorted(list(citations)))
        
        prompt = f"""You are a precise, highly accurate AI assistant. Use ONLY the provided extracts to answer the question.
If the extracts do not contain the answer, state that you do not have sufficient information.

EXTRACTS:
{context_str}

USER QUESTION: {state["query"]}"""

        print(f"> [Node: Generate] Sending request to Groq ({self.groq_model})...")
        try:
            response = self.llm_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.groq_model,
                temperature=0.1
            )
            final_text = response.choices[0].message.content
            model_used = f"Groq ({self.groq_model})"
            
        except Exception as groq_err:
            print(f"\n[ALERT] Groq API failed (Likely invalid model name or rate limit): {groq_err}")
            print(f"> [Node: Generate] Initiating Local Fallback: Routing to Ollama ({self.ollama_model})...")
            
            try:
                final_text = self._call_ollama(prompt)
                model_used = f"Local Ollama ({self.ollama_model})"
            except Exception as ollama_err:
                return {
                    "response": (
                        f"[CRITICAL ERROR] Both LLM providers failed.\n"
                        f"- Groq Error: {groq_err}\n"
                        f"- Ollama Error: {ollama_err}\n"
                        f"Ensure Ollama is actively running in the background."
                    )
                }

        full_response = f"{final_text}\n\n🤖 Generated using: {model_used}\n📚 CITATIONS:\n{citation_str}"
        return {"response": full_response}

    def node_alternative(self, state: RAGState) -> RAGState:
        print("> [Node: Alternative_State] Bypassing LLM generation.")
        fallback = "I found data related to your query, but its confidence score was too low to guarantee an accurate answer. Please rephrase your question with more specific keywords."
        return {"response": fallback}


    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("fuse_and_rerank", self.node_fuse_and_rerank)
        workflow.add_node("generate", self.node_generate)
        workflow.add_node("alternative", self.node_alternative)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "fuse_and_rerank")
        workflow.add_conditional_edges(
            "fuse_and_rerank",
            self.route_evaluation,
            {"generate": "generate", "alternative": "alternative"}
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("alternative", END)
        
        return workflow.compile()

    def run_query(self, query: str):
        initial_state = {"query": query}
        result = self.app.invoke(initial_state)
        print("\n" + "-"*50)
        print("Output:")
        print(result["response"])
        print("-" * 50 + "\n")


if __name__ == "__main__":
    pipeline = AdvancedRAGPipeline()
    
    print("==================================================")
    print("   GROQ + LOCAL OLLAMA HYBRID RAG Pipeline        ")
    print("==================================================")
    
    while True:
        try:
            user_input = input("Enter your Prompt (or 'exit'): ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("\n[SYSTEM] Shutting down pipeline. Goodbye!")
                break
            if user_input:
                pipeline.run_query(user_input)
        except KeyboardInterrupt:
            print("\n[SYSTEM] Shutting down pipeline. Goodbye!")
            break