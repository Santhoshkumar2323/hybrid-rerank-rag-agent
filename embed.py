import os
import uuid
import numpy as np
import pandas as pd
from typing import List
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions
from sklearn.manifold import TSNE
import plotly.express as px
from pypdf import PdfReader

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  

from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import logging
logging.getLogger("pypdf").setLevel(logging.ERROR) 
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


class LocalPDFIndexer:
    def __init__(self, db_path: str = "./local_chroma_db", collection_name: str = "pdf_books"):
        self.client = chromadb.PersistentClient(path=db_path)
        
        print("Loading local Embedding Model (BAAI/bge-large-en-v1.5)...")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-en-v1.5"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            full_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
            return "\n".join(full_text)
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 400, chunk_overlap: int = 40) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def process_data_folder(self, folder_path: str = "data", batch_size: int = 64):
        if not os.path.exists(folder_path):
            print(f"Error: The folder '{folder_path}' does not exist. Creating it now...")
            os.makedirs(folder_path)
            print("Folder created. Please put your PDFs in the 'data' folder and try again.")
            return

        results = self.collection.get(include=["metadatas"])
        existing_files = set()
        if results['metadatas']:
            for meta in results['metadatas']:
                if 'source_file' in meta:
                    existing_files.add(meta['source_file'])

        ids, documents, metadatas = [], [], []
        doc_counter = 0
        files_found = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

        if not files_found:
            print("No PDFs found in the 'data' folder.")
            return

        for filename in files_found:
            if filename in existing_files:
                print(f"\nSkipping '{filename}' - Already exists in the database.")
                continue
            
            pdf_path = os.path.join(folder_path, filename)
            print(f"\nProcessing new book: {filename}...")
            
            raw_text = self.extract_text_from_pdf(pdf_path)
            if not raw_text.strip():
                print(f"Skipping {filename} (No readable text found).")
                continue
            
            chunks = self.chunk_text(raw_text)
            print(f" -> Extracted a total of {len(chunks)} chunks. Saving to database in batches...")

            for index, chunk in enumerate(chunks):
                ids.append(f"book_{doc_counter}_chunk_{index}_{str(uuid.uuid4())[:8]}")
                documents.append(chunk)
                metadatas.append({
                    "source_file": filename,
                    "chunk_index": index
                })
                
                if len(documents) >= batch_size:
                    self._write_to_db(ids, documents, metadatas)
                    ids, documents, metadatas = [], [], []
            
            doc_counter += 1

        if documents:
            self._write_to_db(ids, documents, metadatas)
            
        if doc_counter == 0:
            print("\nFinished! No new books needed to be processed.")
        else:
            print(f"\nFinished! Successfully processed {doc_counter} new books.")

    def _write_to_db(self, ids, docs, metas):
        print(f" -> Saving {len(docs)} chunks to local vector database...")
        self.collection.add(ids=ids, documents=docs, metadatas=metas)


    def get_total_count(self) -> int:
        return self.collection.count()

    def list_indexed_books(self):
        results = self.collection.get(include=["metadatas"])
        if not results['metadatas']:
            print("\nThe database is currently empty.")
            return []

        sources = [meta['source_file'] for meta in results['metadatas'] if 'source_file' in meta]
        book_counts = Counter(sources)

        print("\n=== Currently Indexed Files ===")
        for book, count in book_counts.items():
            print(f"- {book} ({count} chunks)")
        
        return list(book_counts.keys())

    def delete_book_from_db(self, filename: str):
        print(f"\nLocating and deleting chunks for '{filename}'...")
        try:
            self.collection.delete(where={"source_file": filename})
            print(f"✔ Successfully wiped '{filename}' from the vector database.")
        except Exception as e:
            print(f"Error deleting '{filename}': {e}")

    def inspect_sample_records(self, num_samples: int = 3):
        results = self.collection.get()
        total = len(results['ids'])
        print(f"\n=== Database Content Snapshot (Total Chunks: {total}) ===")
        
        if total == 0:
            print("The database is currently empty.")
            return

        for i in range(min(num_samples, total)):
            print(f"\n[Sample Chunk #{i+1}]")
            print(f"ID: {results['ids'][i]}")
            print(f"Source Document: {results['metadatas'][i]['source_file']}")
            print(f"Chunk Index Location: {results['metadatas'][i]['chunk_index']}")
            print(f"Snippet: {results['documents'][i][:200]}...")
            print("-" * 40)

    def inspect_raw_embeddings(self):
        results = self.collection.get(include=["embeddings"])
        
        vectors = results.get('embeddings')
        if vectors is None or len(vectors) == 0:
            print("\nNo vectors found. Have you indexed your data folder yet?")
            return
            
        first_vector = vectors[0]
        print("\n=== Vector Embedding Structure ===")
        print(f"Model Dimensionality: {len(first_vector)} dimensions")
        print(f"First 10 numerical values of the vector array:\n{first_vector[:10]}")

    def query_similarity_search(self, user_query: str, top_k: int = 2):
        print(f"\nSearching for chunks matching: '{user_query}'...")
        results = self.collection.query(
            query_texts=[user_query],
            n_results=top_k
        )
        
        if not results['documents'] or not results['documents'][0]:
            print("No matching text evidence found.")
            return

        print("\n=== Best Matching Evidence Snippets ===")
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            
            print(f"\n[Match #{i+1}] (Semantic Distance: {dist:.4f})")
            print(f"Source: {meta['source_file']} (Chunk {meta['chunk_index']})")
            print(f"Evidence Text:\n{doc}")
            print("-" * 50)
            
    def visualize_and_save_db(self):
        print("\nFetching vectors from database...")
        results = self.collection.get(include=["embeddings", "documents", "metadatas"])
        
        vectors = results.get('embeddings')
        if vectors is None or len(vectors) < 2:
            print("Error: You need at least 2 text chunks in the database to build a map!")
            return

        print(f"Converting {len(vectors)} high-dimensional vectors. Please wait...")
        
        matrix = np.array(vectors)
        perplexity_value = min(30, len(vectors) - 1)
        
        try:
            visual_dir = "visual"
            if not os.path.exists(visual_dir):
                os.makedirs(visual_dir)
                
            counter = 1
            while True:
                output_filename = os.path.join(visual_dir, f"vector_universe_{counter}.html")
                if not os.path.exists(output_filename):
                    break
                counter += 1


            tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
            embeddings_2d = tsne.fit_transform(matrix)
            
            sources = [meta['source_file'] for meta in results['metadatas']]
            snippets = [doc[:120].replace("\n", " ") + "..." for doc in results['documents']]
            
            df = pd.DataFrame({
                "Dimension A": embeddings_2d[:, 0],
                "Dimension B": embeddings_2d[:, 1],
                "Source File": sources,
                "Text Snippet": snippets
            })
            
            fig = px.scatter(
                df,
                x="Dimension A",
                y="Dimension B",
                color="Source File",
                hover_name="Source File",
                hover_data={"Text Snippet": True, "Dimension A": False, "Dimension B": False},
                title="Local Vector Database Semantic Map"
            )
            
            fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(template="plotly_white")
            
            fig.write_html(output_filename)
            print(f"✔ Success! Interactive map saved to: {os.path.abspath(output_filename)}")
        except Exception as e:
            print(f"An error occurred during visualization: {e}")


if __name__ == "__main__":
    indexer = LocalPDFIndexer()

    while True:
        print("\n==================================")
        print("    LOCAL VECTOR DB MANAGER       ")
        print("==================================")
        print(f"Status: Database holds {indexer.get_total_count()} active chunks.")
        print("1. Index/Process the 'data' Folder")
        print("2. List Books inside the Database")
        print("3. Inspect Sample Database Records")
        print("4. Inspect Raw Vector Math")
        print("5. Test Semantic Query Search")
        print("6. Generate 2D Universe Map (HTML)")
        print("7. Delete a Book from the Database")
        print("8. Exit")
        
        choice = input("\nChoose an option (1-8): ").strip()
        
        if choice == '1':
            indexer.process_data_folder(folder_path="data")
        elif choice == '2':
            indexer.list_indexed_books()
        elif choice == '3':
            indexer.inspect_sample_records()
        elif choice == '4':
            indexer.inspect_raw_embeddings()
        elif choice == '5':
            query = input("\nEnter your search question: ").strip()
            if query:
                indexer.query_similarity_search(query)
        elif choice == '6':
            indexer.visualize_and_save_db()
        elif choice == '7':
            active_books = indexer.list_indexed_books()
            if active_books:
                target = input("\nEnter the exact filename to delete (or press Enter to cancel): ").strip()
                if target in active_books:
                    indexer.delete_book_from_db(target)
                elif target:
                    print(f"File '{target}' not found in the database. Ensure you include the .pdf extension.")
        elif choice == '8':
            print("Exiting system. Goodbye!")
            break
        else:
            print("Invalid selection. Choose a number between 1 and 8.")