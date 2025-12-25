# query.py
import os
import sys
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings as LlamaSettings, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import get_response_synthesizer

# Import config
sys.path.append(os.path.dirname(__file__))
from config import (
    VECTOR_DB_DIR, COLLECTION_NAME, TOP_K,
    LLM_MODEL, LLM_FALLBACK, EMBED_MODEL, OLLAMA_BASE_URL,
    SIMILARITY_THRESHOLD, LLM_TEMPERATURE, LLM_CONTEXT_WINDOW,
    LLM_REQUEST_TIMEOUT, LLM_NUM_PREDICT, GPU_LAYERS, NUM_GPU,
    MAX_CHUNKS_IN_CONTEXT
)


def initialize_rag_system():
    """Initialize the RAG system with vector store and LLM."""
    print("🔧 Initializing RAG system...")
    
    # 1. Configure embedding model
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    # 2. Configure LLM with fallback
    llm = None
    try:
        print(f"   Attempting to use primary LLM: {LLM_MODEL}")
        llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=LLM_REQUEST_TIMEOUT,
            temperature=LLM_TEMPERATURE,
            context_window=LLM_CONTEXT_WINDOW,
            additional_kwargs={
                "num_predict": LLM_NUM_PREDICT,
                "num_gpu": NUM_GPU,
            }
        )
        print(f"   ✅ Using {LLM_MODEL}")
    except Exception as e:
        print(f"   ⚠️  Primary LLM {LLM_MODEL} unavailable: {str(e)[:100]}")
        print(f"   Falling back to: {LLM_FALLBACK}")
        try:
            llm = Ollama(
                model=LLM_FALLBACK,
                base_url=OLLAMA_BASE_URL,
                request_timeout=LLM_REQUEST_TIMEOUT,
                temperature=LLM_TEMPERATURE,
                context_window=LLM_CONTEXT_WINDOW,
                additional_kwargs={
                    "num_predict": LLM_NUM_PREDICT,
                    "num_gpu": NUM_GPU,
                }
            )
            print(f"   ✅ Using fallback {LLM_FALLBACK}")
        except Exception as e2:
            print(f"   ❌ Fallback also failed: {str(e2)[:100]}")
            print(f"   Please ensure Ollama is running and models are available")
            raise
    
    # Set global settings
    LlamaSettings.embed_model = embed_model
    LlamaSettings.llm = llm
    
    # 3. Load vector store
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"\n❌ Error: Vector store not found!")
        print(f"   Please run ingest.py first to create the vector database.")
        print(f"   Collection '{COLLECTION_NAME}' does not exist in {VECTOR_DB_DIR}")
        sys.exit(1)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 4. Create index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    
    print(f"✅ RAG system initialized")
    print(f"   Embedding model: {EMBED_MODEL}")
    print(f"   LLM: {LLM_MODEL}")
    print(f"   Context window: {LLM_CONTEXT_WINDOW} tokens")
    print(f"   GPU layers: {GPU_LAYERS if GPU_LAYERS > 0 else 'auto'}")
    print(f"   Top-K retrieval: {TOP_K}")
    
    return index


def create_query_engine(index):
    """Create a query engine with retriever and response synthesis."""
    
    # Configure retriever with dynamic top-k based on context window
    # For larger context windows, we can retrieve more chunks
    effective_top_k = min(TOP_K, MAX_CHUNKS_IN_CONTEXT)
    
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=effective_top_k,
    )
    
    # Add similarity threshold filter to remove irrelevant chunks
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=SIMILARITY_THRESHOLD)
    ]
    
    # Custom prompt template for better responses
    qa_prompt_template = (
        "You are an AI assistant answering questions based on provided context documents.\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above, answer the following question in a clear, comprehensive, and well-structured manner.\n"
        "If the context doesn't contain enough information to fully answer the question, say so explicitly.\n"
        "Provide specific details and examples from the context when possible.\n"
        "Format your response with:\n"
        "1. A direct answer to the question\n"
        "2. Supporting details from the context\n"
        "3. Any relevant implications or considerations\n\n"
        "Question: {query_str}\n"
        "Answer: "
    )
    
    qa_prompt = PromptTemplate(qa_prompt_template)
    
    # OLD METHOD (commented out - less reliable):
    # query_engine = RetrieverQueryEngine(
    #     retriever=retriever,
    #     node_postprocessors=node_postprocessors,
    # )
    # query_engine.update_prompts(
    #     {"response_synthesizer:text_qa_template": qa_prompt}
    # )
    
    # NEW METHOD (recommended - guarantees {context_str} and {query_str} population):
    # Create response synthesizer with custom prompt
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_prompt,
        response_mode="compact"  # Use compact mode for better responses
    )
    
    # Create query engine with custom response synthesizer
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
    )
    
    return query_engine


def format_response(response) -> Dict:
    """Format the response with retrieved context."""
    raw_answer = str(response)
    if not raw_answer or raw_answer.strip().lower() == "empty response":
        answer = (
            "I could not find enough relevant information in the indexed documents "
            "to answer that question. Try rephrasing or asking something more "
            "directly related to the ingested documents."
        )
    else:
        answer = raw_answer

    result = {
        "answer": answer,
        "sources": []
    }
    
    # Extract source information
    if hasattr(response, 'source_nodes'):
        for idx, node in enumerate(response.source_nodes, 1):
            # Show more context in preview (300 chars instead of 200)
            preview_text = node.node.text[:300]
            if len(node.node.text) > 300:
                preview_text += "..."
            
            source_info = {
                "chunk_id": idx,
                "text": preview_text,
                "score": node.score,
                "metadata": node.node.metadata
            }
            result["sources"].append(source_info)
    
    return result


def query_interactive(query_engine):
    """Interactive query loop."""
    print("\n" + "=" * 60)
    print("💬 RAG QUERY INTERFACE")
    print("=" * 60)
    print("Type your questions below (or 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print(f"\n🔍 Retrieving relevant context...")
            print(f"🤖 Generating response...\n")
            
            # Query the system with fallback on OOM error
            try:
                response = query_engine.query(question)
                result = format_response(response)
            except Exception as query_error:
                error_msg = str(query_error)
                # Check if it's an OOM error
                if "system memory" in error_msg or "status code: 500" in error_msg:
                    print(f"⚠️  Primary model failed (insufficient memory)")
                    print(f"🔄 Retrying with fallback model: {LLM_FALLBACK}...\n")
                    
                    # Reinitialize with fallback model
                    from llama_index.llms.ollama import Ollama
                    from llama_index.core import Settings as LlamaSettings
                    
                    fallback_llm = Ollama(
                        model=LLM_FALLBACK,
                        base_url=OLLAMA_BASE_URL,
                        request_timeout=120.0,
                        temperature=0.1,
                    )
                    LlamaSettings.llm = fallback_llm
                    
                    # Recreate query engine with fallback
                    query_engine_fallback = create_query_engine(query_engine._index)
                    response = query_engine_fallback.query(question)
                    result = format_response(response)
                else:
                    raise
            
            # Display answer
            print("📝 Answer:")
            print("-" * 60)
            print(result["answer"])
            print("-" * 60)
            
            # Display sources
            if result["sources"]:
                print(f"\n📚 Sources (Top {len(result['sources'])} chunks):")
                for source in result["sources"]:
                    print(f"\n   [{source['chunk_id']}] Score: {source['score']:.3f}")
                    print(f"   File: {source['metadata'].get('filename', 'Unknown')}")
                    print(f"   Preview: {source['text']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def query_single(query_engine, question: str):
    """Query with a single question and return."""
    print(f"\n❓ Question: {question}")
    print(f"\n🔍 Retrieving relevant context...")
    print(f"🤖 Generating response...\n")
    
    response = query_engine.query(question)
    result = format_response(response)
    
    print("📝 Answer:")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    
    if result["sources"]:
        print(f"\n📚 Sources (Top {len(result['sources'])} chunks):")
        for source in result["sources"]:
            print(f"\n   [{source['chunk_id']}] Score: {source['score']:.3f}")
            print(f"   File: {source['metadata'].get('filename', 'Unknown')}")
            print(f"   Preview: {source['text']}")
    
    return result


def main():
    """Main entry point."""
    # Initialize RAG system
    index = initialize_rag_system()
    query_engine = create_query_engine(index)
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        # Single query mode
        question = " ".join(sys.argv[1:])
        query_single(query_engine, question)
    else:
        # Interactive mode
        query_interactive(query_engine)


if __name__ == "__main__":
    main()