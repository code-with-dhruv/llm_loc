import os
from flask import Flask, request, jsonify, render_template, abort
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import hashlib
import json
from datetime import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
import ollama
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'pdfs'
app.config['INDEX_METADATA'] = 'index_metadata.json'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

class VectorStore:
    def __init__(self):
        self.pdf_texts = []
        self.pdf_vectors = []
        self.index = None
        self.last_indexed = None
        self.is_initialized = False
        self.indexed_files = {}
        self.index_file = 'faiss_index.index'
        self.texts_file = 'pdf_texts.json'
        self.load_persisted_data() # Store metadata about indexed files
    def load_persisted_data(self):
    # Load FAISS index
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            logger.info("Loaded FAISS index from disk")

        # Load PDF texts
        if os.path.exists(self.texts_file):
            with open(self.texts_file, 'r') as f:
                self.pdf_texts = json.load(f)
            logger.info(f"Loaded {len(self.pdf_texts)} PDF texts from disk")
        
        # Load file metadata
        self.load_metadata()
        
        # If the persisted index and texts exist, mark the vector store as initialized
        if self.index is not None and len(self.pdf_texts) > 0:
            self.is_initialized = True

            if os.path.exists(self.texts_file):
                with open(self.texts_file, 'r') as f:
                    self.pdf_texts = json.load(f)
                logger.info(f"Loaded {len(self.pdf_texts)} PDF texts from disk")
                
            self.load_metadata()
    

    def save_persisted_data(self):
        if self.index:
            faiss.write_index(self.index, self.index_file)
        with open(self.texts_file, 'w') as f:
            json.dump(self.pdf_texts, f)
        self.save_metadata()

    # 3. Initialize Llama model (add this after Flask app creation)
    llama_model = None
    llama_tokenizer = None
#HF_AUTH_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "your_hf_auth_token_here")


# def load_llama_model():
#     global llama_model, llama_tokenizer
#     try:
#         llama_tokenizer = AutoTokenizer.from_pretrained(
#             "meta-llama/Llama-2-7b-chat-hf",
#             use_auth_token=HF_AUTH_TOKEN  # Pass the token here
#         )
#         llama_model = AutoModel.from_pretrained(
#             "meta-llama/Llama-2-7b-chat-hf",
#             load_in_4bit=True,  # Use quantization for memory efficiency
#             device_map="auto",
#             use_auth_token=HF_AUTH_TOKEN  # Pass the token here as well
#         )
#         logger.info("Loaded Llama model successfully")
#     except Exception as e:
#         logger.error(f"Error loading Llama model: {e}")

# # Initialize model when app starts
#     load_llama_model()

    def clear(self):
        self.pdf_texts = []
        self.pdf_vectors = []
        self.index = None
        self.is_initialized = False
        self.indexed_files = {}

    def save_metadata(self, filename):
        try:
            with open(app.config['INDEX_METADATA'], 'w') as f:
                json.dump(self.indexed_files, f)
            logger.info(f"Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def load_metadata(self):
        try:
            if os.path.exists(app.config['INDEX_METADATA']):
                with open(app.config['INDEX_METADATA'], 'r') as f:
                    self.indexed_files = json.load(f)
                logger.info(f"Loaded metadata for {len(self.indexed_files)} files")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.indexed_files = {}

vector_store = VectorStore()

def find_keyword_matches(text: str, query: str) -> List[Dict[str, Any]]:
    """Find and highlight keyword matches in text."""
    # Split query into keywords and create pattern
    keywords = query.lower().split()
    if not keywords:
        return []
    
    # Find all matches with context
    matches = []
    context_window = 100  # Characters before and after match
    
    text_lower = text.lower()
    for keyword in keywords:
        start = 0
        while True:
            pos = text_lower.find(keyword, start)
            if pos == -1:
                break
                
            # Get context around match
            context_start = max(0, pos - context_window)
            context_end = min(len(text), pos + len(keyword) + context_window)
            
            # Get the actual text with original casing
            context = text[context_start:context_end]
            keyword_start = pos - context_start
            keyword_end = keyword_start + len(keyword)
            
            matches.append({
                "context": context,
                "highlight": {
                    "start": keyword_start,
                    "end": keyword_end,
                    "keyword": text[pos:pos + len(keyword)]
                }
            })
            
            start = pos + len(keyword)
            
            # Limit to first few matches per keyword
            if len(matches) >= 3:
                break
    
    return matches

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_file_metadata(file_path):
    """Get file metadata including hash and last modified time"""
    stat = os.stat(file_path)
    return {
        'hash': calculate_file_hash(file_path),
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'indexed_at': datetime.now().isoformat()
    }

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        page_breaks = []
        current_position = 0
        
        for page in reader.pages:
            page_text = page.extract_text()
            text += page_text
            current_position += len(page_text)
            page_breaks.append(current_position)
            
        return text.strip(), page_breaks
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        return None, []
# def get_embeddings(text, chunk_size=512):
#     inputs = llama_tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=chunk_size,
#         padding=True
#     ).to(llama_model.device)

#     with torch.no_grad():
#         outputs = llama_model(**inputs)
    
#     # Mean pooling for embeddings
#     embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
#     return embeddings


def get_embeddings(text):
    """Generate embeddings using the SentenceTransformer model for indexing."""
    try:
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embeddings with SentenceTransformer: {e}")
        return None
def index_pdfs(folder_path):
    """Index PDFs, skipping already indexed unchanged files."""
    try:
        # Load existing metadata
        if not vector_store.indexed_files:
            vector_store.load_metadata()

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        new_files, unchanged_files, updated_files = [], [], []

        # Identify new, updated, or unchanged files
        for filename in pdf_files:
            pdf_path = os.path.join(folder_path, filename)
            current_metadata = get_file_metadata(pdf_path)

            if filename not in vector_store.indexed_files:
                new_files.append(filename)
            elif vector_store.indexed_files[filename]['hash'] != current_metadata['hash']:
                updated_files.append(filename)
            else:
                unchanged_files.append(filename)

        if not new_files and not updated_files:
            logger.info("No new or modified PDFs found.")
            return True

        total_files = len(new_files) + len(updated_files)
        logger.info(f"Processing {total_files} files (New: {len(new_files)}, Updated: {len(updated_files)})")

        new_pdf_texts = []
        new_pdf_vectors = []

        # Process new and updated PDFs
        for idx, filename in enumerate(new_files + updated_files):
            pdf_path = os.path.join(folder_path, filename)
            text, page_breaks = extract_text_from_pdf(pdf_path)

            if text:
                chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
                chunk_embeddings = []

                for chunk in chunks:
                    embedding = get_embeddings(chunk)
                    if embedding is not None:
                        chunk_embeddings.append(embedding)

                if chunk_embeddings:
                    doc_embedding = np.mean(chunk_embeddings, axis=0)

                    current_metadata = get_file_metadata(pdf_path)
                    vector_store.indexed_files[filename] = current_metadata

                    new_pdf_texts.append({
                        'filename': filename,
                        'content': text,
                        'page_breaks': page_breaks,
                        'indexed_at': current_metadata['indexed_at']
                    })
                    new_pdf_vectors.append(doc_embedding)

            # Print progress
            logger.info(f"Indexed {idx + 1}/{total_files} PDFs...")

        # Preserve unchanged PDFs
        for filename in unchanged_files:
            idx = next((i for i, item in enumerate(vector_store.pdf_texts) if item['filename'] == filename), None)
            if idx is not None:
                new_pdf_texts.append(vector_store.pdf_texts[idx])
                new_pdf_vectors.append(vector_store.pdf_vectors[idx])

        # Update vector store
        vector_store.pdf_texts = new_pdf_texts
        vector_store.pdf_vectors = new_pdf_vectors

        if new_pdf_vectors:
            dimension = len(new_pdf_vectors[0])
            vector_store.index = faiss.IndexFlatL2(dimension)
            vector_store.index.add(np.array(new_pdf_vectors))

            vector_store.last_indexed = datetime.now()
            vector_store.is_initialized = True

            vector_store.save_metadata(app.config['INDEX_METADATA'])

            logger.info(f"Finished indexing {len(new_pdf_texts)} PDFs. (Unchanged: {len(unchanged_files)})")
            return True

        return False

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return False
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def index_folder():
    try:
        folder_path = request.json.get('folder_path', app.config['UPLOAD_FOLDER'])
        
        if index_pdfs(folder_path):
            stats = {
                'total_indexed': len(vector_store.pdf_texts),
                'last_indexed': vector_store.last_indexed.isoformat() if vector_store.last_indexed else None
            }
            return jsonify({
                "message": f"Index updated. Total PDFs indexed: {stats['total_indexed']}",
                "stats": stats,
                "status": "success"
            }), 200
        else:
            return jsonify({
                "error": "No PDFs found or indexing failed",
                "status": "error",
                "details": "Please add PDF files to the 'pdfs' folder and try again"
            }), 400
    except Exception as e:
        logger.error(f"Error in index_folder: {e}")
        return jsonify({
            "error": str(e),
            "status": "error",
            "details": "An unexpected error occurred during indexing"
        }), 500

@app.route('/search', methods=['POST'])
def search_pdfs():
    try:
        if not vector_store.is_initialized:
            return jsonify({
                "error": "Search index not initialized",
                "status": "error",
                "details": "Please index PDFs first using the 'Index PDFs' button"
            }), 400

        query = request.json.get('query')
        if not query:
            return jsonify({
                "error": "Query is required",
                "status": "error",
                "details": "Please provide a search query"
            }), 400

        # Use SentenceTransformer for query embedding
        query_vector = model.encode(query)

        # Retrieve the single most relevant document
        k = 1
        distances, indices = vector_store.index.search(np.array([query_vector]), k=k)
        top_index = indices[0][0]

        if top_index >= len(vector_store.pdf_texts):
            return jsonify({
                "error": "Invalid index returned",
                "status": "error",
                "details": "Search index returned an invalid index"
            }), 500

        # Get the most relevant document
        doc = vector_store.pdf_texts[top_index]
        content = doc['content']

        # Now use Ollama only for generating the answer based on the best match
        ollama_response = ollama.chat(
            model="llama2",
            messages=[{
                "role": "user",
                "content": f"Based on the following document, answer the query '{query}':\n\n{content[:2000]}"
            }]
        )
        answer = ollama_response["message"]["content"]

        return jsonify({
            "answer": answer,
            "filename": doc['filename'],
            "distance": float(distances[0][0]),
            "indexed_at": doc['indexed_at'],
            "query_time": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in search_pdfs: {e}")
        return jsonify({
            "error": str(e),
            "status": "error",
            "details": "An unexpected error occurred during search"
        }), 500
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large", "status": "error"}), 413

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)