# ingest.py (MODIFIED VERSION)
import os
import sys
from pathlib import Path
from typing import List, Set
import chromadb
from chromadb.config import Settings
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings as LlamaSettings
from document_tracker import DocumentTracker
from document_loaders import load_document, get_supported_extensions

sys.path.append(os.path.dirname(__file__))
from config import (
    RAW_DOCS_DIR, VECTOR_DB_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, OLLAMA_BASE_URL, TRACKING_DB_PATH
)

def load_documents() -> List[Document]:
    """Load documents from the raw data directory, skipping already ingested ones."""
    documents = []
    tracker = DocumentTracker(TRACKING_DB_PATH)
    
    raw_path = Path(RAW_DOCS_DIR)
    print(f"\n Loading documents from: {raw_path}")
    
    # Get all files using supported extensions
    supported_extensions = get_supported_extensions()
    file_patterns = [f"*{ext}" for ext in supported_extensions]
    files = []
    for pattern in file_patterns:
        files.extend(raw_path.glob(pattern))
    
    if not files:
        print(f"  No documents found in {RAW_DOCS_DIR}")
        print(f"   Supported formats: {', '.join(supported_extensions)}")
        return documents
    
    print(f" Found {len(files)} files")

    # Track Stats
    skipped_count = 0
    loaded_count = 0

    for file_path in files:
        # CHECK IF ALREADY INGESTED
        if tracker.is_document_ingested(file_path):
            print(f"   ⏭️  Skipping (already ingested): {file_path.name}")
            skipped_count += 1
            continue
        
        try:
            print(f"   📥 Loading: {file_path.name}")
            
            # Use the new loader system
            text = load_document(file_path)
            
            if text is None:
                print(f"   ⚠️  Skipped (could not load): {file_path.name}")
                continue
            
            if text.strip():
                doc = Document(
                    text=text,
                    metadata={
                        "filename": file_path.name,
                        "file_type": file_path.suffix,
                        "file_path": str(file_path)
                    }
                )
                documents.append(doc)
                loaded_count += 1
            else:
                print(f"   ⚠️  Skipped (empty): {file_path.name}")
                
        except Exception as e:
            print(f"   ❌ Error loading {file_path.name}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   ✅ New documents: {loaded_count}")
    print(f"   ⏭️  Already ingested: {skipped_count}")
    
    return documents

def create_vector_store():
    """Initialize ChromaDB vector store."""
    print("\n Initializing ChromaDB...")
    
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    # Get or create (DON'T DELETE!)
    chroma_collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )    
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    print(f" ChromaDB initialized at: {VECTOR_DB_DIR}")
    
    return vector_store, chroma_collection # todo modify: do i need to comment out chroma_collection??

def ingest_documents():  # MODIFIED: Removed reset parameter
    # Main ingestion pipeline.
    print("=" * 60)
    print("STARTING DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    tracker = DocumentTracker(TRACKING_DB_PATH)

    # Show current stats
    stats = tracker.get_ingestion_stats()
    print(f"\n Current database stats:")
    print(f"   Total documents ingested: {stats['total_documents']}")
    print(f"   Total chunks: {stats['total_chunks']}")

    # Load NEW documents only  
    # 1. Load documents
    documents = load_documents()
    if not documents:
        print("\n✅ No new documents to ingest. Database is up to date.")
        return
    
    # 2. Configure embedding model
    print(f"\n Configuring embedding model: {EMBED_MODEL}")
    try:
        embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        LlamaSettings.embed_model = embed_model
    except Exception as e:
        print(f"\n❌ ERROR: Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        print(f"   Make sure Ollama is running: ollama serve")
        print(f"   Error: {e}")
        return
    
    # 3. Create vector store
    vector_store, chroma_collection = create_vector_store() # MODIFIED
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 5. Configure chunking
    print(f"\n Configuring text splitter:")
    print(f"   Chunk size: {CHUNK_SIZE} tokens")
    print(f"   Chunk overlap: {CHUNK_OVERLAP} tokens")
    
    text_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # 6. Create index
    print(f"\n  Processing documents...")
    print(f"   - Splitting into chunks")
    print(f"   - Generating embeddings")
    print(f"   - Storing in vector database")
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[text_splitter],
        show_progress=True
    )
    
    # After successful indexing, mark documents as ingested
    print("\n Updating tracking database...")
    
    # Get actual chunk counts from ChromaDB instead of from the nodes (old = "nodes = index.docstore.docs")
    doc_chunk_counts = {}
    try:
        # Query ChromaDB for all items and count by source file
        all_items = chroma_collection.get(include=['metadatas'])
        if all_items and all_items['metadatas']:
            for metadata in all_items['metadatas']:
                source_file = metadata.get('file_path')
                if source_file:
                    doc_chunk_counts[source_file] = doc_chunk_counts.get(source_file, 0) + 1
    except Exception as e:
        print(f"   ⚠️  Could not get chunk counts from ChromaDB: {e}")
        # Fallback to estimation if ChromaDB query fails
        for doc in documents:
            doc_chunk_counts[str(Path(doc.metadata['file_path']))] = len(doc.text) // CHUNK_SIZE
    
    # Mark each document with accurate chunk count
    for doc in documents:
        file_path = Path(doc.metadata['file_path'])
        actual_chunks = doc_chunk_counts.get(str(file_path), 0)
        tracker.mark_document_ingested(file_path, num_chunks=actual_chunks)
        print(f"   ✓ {file_path.name}: {actual_chunks} chunks")
    
    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE!")
    print("=" * 60)
    print(f" Statistics:")
    print(f"   Documents processed: {len(documents)}")
    print(f"   Vector store: {VECTOR_DB_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"\n Next step: Run query.py to test retrieval")
    print("=" * 60)

    # Show updated stats
    stats = tracker.get_ingestion_stats()
    print(f" Updated stats:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Total chunks: {stats['total_chunks']}")


if __name__ == "__main__":    
    ingest_documents()