import os
import json
import re
import faiss
import torch
import hashlib
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, abort
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import ollama

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'pdfs'
app.config['INDEX_METADATA'] = 'index_metadata.json'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use SentenceTransformer for indexing and query encoding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables to track indexing progress and recent chats
INDEX_PROGRESS = {"total": 0, "indexed": 0}
RECENT_CHATS = []  # Each entry: { query, filename, answer, timestamp }

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
        self.load_persisted_data()

    def load_persisted_data(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            logger.info("Loaded FAISS index from disk")
        if os.path.exists(self.texts_file):
            with open(self.texts_file, 'r') as f:
                self.pdf_texts = json.load(f)
            logger.info(f"Loaded {len(self.pdf_texts)} PDF texts from disk")
        self.load_metadata()
        if self.index is not None and len(self.pdf_texts) > 0:
            self.is_initialized = True

    def save_persisted_data(self):
        if self.index:
            faiss.write_index(self.index, self.index_file)
        with open(self.texts_file, 'w') as f:
            json.dump(self.pdf_texts, f)
        self.save_metadata()

    def clear(self):
        self.pdf_texts = []
        self.pdf_vectors = []
        self.index = None
        self.is_initialized = False
        self.indexed_files = {}

    def save_metadata(self):
        try:
            with open(app.config['INDEX_METADATA'], 'w') as f:
                json.dump(self.indexed_files, f)
            logger.info("Metadata saved successfully")
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

def calculate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_file_metadata(file_path):
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

# Use SentenceTransformer for indexing embeddings
def get_embeddings(text):
    try:
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

def index_pdfs(folder_path):
    try:
        if not vector_store.indexed_files:
            vector_store.load_metadata()
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        new_files, unchanged_files, updated_files = [], [], []

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
        INDEX_PROGRESS["total"] = total_files  # Set progress total
        INDEX_PROGRESS["indexed"] = 0
        logger.info(f"Processing {total_files} files (New: {len(new_files)}, Updated: {len(updated_files)})")

        new_pdf_texts = []
        new_pdf_vectors = []

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
            INDEX_PROGRESS["indexed"] += 1  # Update progress after each file
            logger.info(f"Indexed {idx + 1}/{total_files} PDFs...")

        for filename in unchanged_files:
            idx = next((i for i, item in enumerate(vector_store.pdf_texts) if item['filename'] == filename), None)
            if idx is not None:
                new_pdf_texts.append(vector_store.pdf_texts[idx])
                new_pdf_vectors.append(vector_store.pdf_vectors[idx])

        vector_store.pdf_texts = new_pdf_texts
        vector_store.pdf_vectors = new_pdf_vectors
        if new_pdf_vectors:
            dimension = len(new_pdf_vectors[0])
            vector_store.index = faiss.IndexFlatL2(dimension)
            vector_store.index.add(np.array(new_pdf_vectors))
            vector_store.last_indexed = datetime.now()
            vector_store.is_initialized = True
            vector_store.save_metadata()
            logger.info(f"Finished indexing {len(new_pdf_texts)} PDFs. (Unchanged: {len(unchanged_files)})")
            return True
        return False
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return False

@app.route('/')
def index():
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
        k = 1  # Only fetch the top result
        distances, indices = vector_store.index.search(np.array([query_vector]), k=k)
        top_index = indices[0][0]

        if top_index >= len(vector_store.pdf_texts):
            return jsonify({
                "error": "Invalid index returned",
                "status": "error",
                "details": "Search index returned an invalid index"
            }), 500

        doc = vector_store.pdf_texts[top_index]
        content = doc['content']

        # Use Ollama only for generating the answer on the most relevant document.
        ollama_response = ollama.chat(
            model="llama2",
            messages=[{
                "role": "user",
                "content": f"Based on the following document, answer the query '{query}':\n\n{content[:2000]}"
            }]
        )
        answer = ollama_response["message"]["content"]

        # Prepare result and store the chat session
        result = {
            "filename": doc['filename'],
            "content_snippet": answer,
            "distance": float(distances[0][0]),
            "indexed_at": doc['indexed_at'],
            "query_time": datetime.now().isoformat()
        }

        RECENT_CHATS.append({
            "query": query,
            "filename": doc['filename'],
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        if len(RECENT_CHATS) > 10:
            RECENT_CHATS.pop(0)

        return jsonify({
            "results": [result],
            "status": "success",
            "query_time": datetime.now().isoformat(),
            "total_results": 1
        }), 200

    except Exception as e:
        logger.error(f"Error in search_pdfs: {e}")
        return jsonify({
            "error": str(e),
            "status": "error",
            "details": "An unexpected error occurred during search"
        }), 500

@app.route('/progress', methods=['GET'])
def progress():
    return jsonify(INDEX_PROGRESS)

@app.route('/chats', methods=['GET'])
def chats():
    return jsonify(RECENT_CHATS)

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large", "status": "error"}), 413

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
