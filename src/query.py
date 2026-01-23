# query.py
# Updated to support both Ollama and LM Studio for LLM inference
# Embeddings ALWAYS use Ollama (LM Studio limitation)

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
from llama_index.core import get_response_synthesizer
from custom_retriever import create_filtered_retriever

# LM Studio uses OpenAI-compatible API
try:
    from llama_index.llms.openai_like import OpenAILike
    HAS_OPENAI_LIKE = True
except ImportError:
    HAS_OPENAI_LIKE = False
    print("Warning: llama_index.llms.openai_like not available. Install with:")
    print("  pip install llama-index-llms-openai-like")

# Import config
sys.path.append(os.path.dirname(__file__))
from config import (
    # Backend selection
    LLM_BACKEND,
    # Ollama settings (embeddings + optional LLM)
    OLLAMA_BASE_URL, EMBED_MODEL, OLLAMA_LLM_MODEL, OLLAMA_LLM_FALLBACK,
    OLLAMA_NUM_THREAD, OLLAMA_NUM_BATCH,
    # LM Studio settings (LLM only)
    LMSTUDIO_BASE_URL, LMSTUDIO_LLM_MODEL,
    # Shared LLM settings
    LLM_TEMPERATURE, LLM_CONTEXT_WINDOW, LLM_REQUEST_TIMEOUT, LLM_NUM_PREDICT,
    # Hardware
    GPU_LAYERS, NUM_GPU,
    # RAG settings
    VECTOR_DB_DIR, COLLECTION_NAME, TOP_K, SIMILARITY_THRESHOLD,
    MAX_CHUNKS_IN_CONTEXT, EMBED_BATCH_SIZE, USE_RERANKING, RERANK_TOP_N,
    # Backward compatibility
    LLM_MODEL, LLM_FALLBACK
)


def _create_ollama_llm(model_name: str) -> Ollama:
    """Create an Ollama LLM instance with full configuration."""
    return Ollama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        request_timeout=LLM_REQUEST_TIMEOUT,
        temperature=LLM_TEMPERATURE,
        context_window=LLM_CONTEXT_WINDOW,
        additional_kwargs={
            "num_predict": LLM_NUM_PREDICT,
            "num_gpu": NUM_GPU,
            "num_gqa": GPU_LAYERS,
            "num_thread": OLLAMA_NUM_THREAD,
            "num_batch": OLLAMA_NUM_BATCH,
            "num_ctx": LLM_CONTEXT_WINDOW,
        }
    )


def _create_lmstudio_llm(model_name: str) -> "OpenAILike":
    """
    Create an LM Studio LLM instance using OpenAI-compatible API.
    
    LM Studio serves models on http://localhost:1234/v1 by default.
    It implements OpenAI's chat completions API.
    
    IMPORTANT: You must:
    1. Load the model in LM Studio GUI first
    2. Start the local server in LM Studio (Server tab → Start Server)
    """
    if not HAS_OPENAI_LIKE:
        raise ImportError(
            "llama-index-llms-openai-like required for LM Studio. "
            "Install with: pip install llama-index-llms-openai-like"
        )
    
    return OpenAILike(
        model=model_name,
        api_base=LMSTUDIO_BASE_URL,
        api_key="lm-studio",  # LM Studio ignores API key but requires non-empty value
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_NUM_PREDICT,
        context_window=LLM_CONTEXT_WINDOW,
        timeout=LLM_REQUEST_TIMEOUT,
        is_chat_model=True,  # LM Studio serves chat completions
    )


def initialize_rag_system():
    """
    Initialize the RAG system with vector store and LLM.
    
    Architecture:
    - Embeddings: ALWAYS use Ollama (port 11434) with bge-m3
    - LLM: Use Ollama OR LM Studio based on LLM_BACKEND config
    
    This split is necessary because LM Studio cannot serve both
    LLM and embedding models simultaneously.
    """
    print("Initializing RAG system...")
    print(f"   LLM Backend: {LLM_BACKEND.upper()}")
    
    # =========================================================================
    # 1. Configure embedding model (ALWAYS Ollama)
    # =========================================================================
    # NOTE: embed_batch_size for heavy machine (128GB RAM). Comment out if not on heavy machine. Processes 64 texts per batch (128GB RAM) currently.
    print(f"   Embeddings: Ollama @ {OLLAMA_BASE_URL}")
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        embed_batch_size=EMBED_BATCH_SIZE,
    )
    
    # =========================================================================
    # 2. Configure LLM (Ollama OR LM Studio based on config)
    # =========================================================================
    llm = None
    
    if LLM_BACKEND == "lmstudio":
        # --- LM Studio Backend ---
        print(f"   LLM: LM Studio @ {LMSTUDIO_BASE_URL}")
        try:
            llm = _create_lmstudio_llm(LMSTUDIO_LLM_MODEL)
            print(f"   Using LM Studio model: {LMSTUDIO_LLM_MODEL}")
        except Exception as e:
            print(f"   LM Studio failed: {str(e)[:100]}")
            print(f"   Falling back to Ollama: {OLLAMA_LLM_FALLBACK}")
            try:
                llm = _create_ollama_llm(OLLAMA_LLM_FALLBACK)
                print(f"   Using Ollama fallback: {OLLAMA_LLM_FALLBACK}")
            except Exception as e2:
                print(f"   Fallback also failed: {str(e2)[:100]}")
                raise
    else:
        # --- Ollama Backend (default) ---
        print(f"   LLM: Ollama @ {OLLAMA_BASE_URL}")
        try:
            llm = _create_ollama_llm(OLLAMA_LLM_MODEL)
            print(f"   Using Ollama model: {OLLAMA_LLM_MODEL} (GPU layers: {GPU_LAYERS})")
        except Exception as e:
            print(f"   Primary LLM {OLLAMA_LLM_MODEL} unavailable: {str(e)[:100]}")
            print(f"   Falling back to: {OLLAMA_LLM_FALLBACK}")
            try:
                llm = _create_ollama_llm(OLLAMA_LLM_FALLBACK)
                print(f"   Using fallback: {OLLAMA_LLM_FALLBACK}")
            except Exception as e2:
                print(f"   Fallback also failed: {str(e2)[:100]}")
                print(f"   Please ensure Ollama is running and models are available")
                raise
    
    # Set global settings
    LlamaSettings.embed_model = embed_model
    LlamaSettings.llm = llm
    
    # =========================================================================
    # 3. Load vector store (ChromaDB)
    # =========================================================================
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"\n Error: Vector store not found!")
        print(f"   Please run ingest.py first to create the vector database.")
        print(f"   Collection '{COLLECTION_NAME}' does not exist in {VECTOR_DB_DIR}")
        sys.exit(1)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # =========================================================================
    # 4. Create index from existing vector store
    # =========================================================================
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    
    # Print summary
    print(f"\n RAG system initialized")
    print(f"   Embedding model: {EMBED_MODEL} (via Ollama)")
    print(f"   LLM: {LMSTUDIO_LLM_MODEL if LLM_BACKEND == 'lmstudio' else OLLAMA_LLM_MODEL} (via {LLM_BACKEND.upper()})")
    print(f"   Context window: {LLM_CONTEXT_WINDOW} tokens")
    if LLM_BACKEND == "ollama":
        print(f"   GPU layers: {GPU_LAYERS if GPU_LAYERS > 0 else 'auto'}")
    print(f"   Top-K retrieval: {TOP_K}")
    
    return index


def create_query_engine(index):
    """Create a query engine with retriever and response synthesis."""
    
    # Configure retriever with dynamic top-k based on context window
    effective_top_k = min(TOP_K, MAX_CHUNKS_IN_CONTEXT)
    
    # Create base retriever
    base_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=effective_top_k,
    )
    
    # Wrap with custom FilteredRetriever for similarity filtering at retrieval time
    retriever = create_filtered_retriever(
        base_retriever=base_retriever,
        similarity_threshold=SIMILARITY_THRESHOLD,
        verbose=True
    )
    
    # Configure postprocessors - reranking if enabled
    node_postprocessors = []
    
    if USE_RERANKING:
        try:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            reranker = SentenceTransformerRerank(
                model="BAAI/bge-reranker-v2-m3",
                top_n=RERANK_TOP_N
            )
            node_postprocessors.append(reranker)
            print(f"   Reranking enabled: top {RERANK_TOP_N} results")
        except ImportError:
            print("   Warning: SentenceTransformerRerank not available, skipping reranking")
    
    # Custom prompt template
    qa_prompt_template = (
        "You are a knowledgeable AI assistant. Answer the user's question based on the provided context documents when available.\n\n"
        "Context information from indexed documents:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Instructions:\n"
        "1. If the context contains relevant information, use it to answer the question comprehensively.\n"
        "2. If the context is empty or insufficient, provide a helpful answer based on your general knowledge.\n"
        "3. Be direct, clear, and professional in your responses.\n"
        "4. Never fabricate information or claim something is in the documents when it isn't.\n"
        "5. Provide accurate, well-structured answers that directly address the user's question.\n\n"
        "Question: {query_str}\n"
        "Answer:"
    )
    
    qa_prompt = PromptTemplate(qa_prompt_template)
    
    # Create response synthesizer with custom prompt
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_prompt,
        response_mode="compact"
    )
    
    # Create query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
    )
    
    # Store references for fallback queries
    query_engine._llm = LlamaSettings.llm
    query_engine._retriever = retriever
    query_engine._response_synthesizer = response_synthesizer
    
    return query_engine


def create_streaming_query_engine(index):
    """
    Create a query engine with STREAMING enabled for real-time response.
    This is a streaming-enabled version of create_query_engine().
    """
    # Configure retriever with dynamic top-k based on context window
    effective_top_k = min(TOP_K, MAX_CHUNKS_IN_CONTEXT)
    
    # Create base retriever
    base_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=effective_top_k,
    )
    
    # Wrap with custom FilteredRetriever for similarity filtering at retrieval time
    retriever = create_filtered_retriever(
        base_retriever=base_retriever,
        similarity_threshold=SIMILARITY_THRESHOLD,
        verbose=True
    )
    
    # Configure postprocessors - reranking if enabled
    node_postprocessors = []
    
    if USE_RERANKING:
        try:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            reranker = SentenceTransformerRerank(
                model="BAAI/bge-reranker-v2-m3",
                top_n=RERANK_TOP_N
            )
            node_postprocessors.append(reranker)
            print(f"   Reranking enabled: top {RERANK_TOP_N} results")
        except ImportError:
            print("   Warning: SentenceTransformerRerank not available, skipping reranking")
    
    # Custom prompt template (same as non-streaming)
    qa_prompt_template = (
        "You are a knowledgeable AI assistant. Answer the user's question based on the provided context documents when available.\n\n"
        "Context information from indexed documents:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Instructions:\n"
        "1. If the context contains relevant information, use it to answer the question comprehensively.\n"
        "2. If the context is empty or insufficient, provide a helpful answer based on your general knowledge.\n"
        "3. Be direct, clear, and professional in your responses.\n"
        "4. Never fabricate information or claim something is in the documents when it isn't.\n"
        "5. Provide accurate, well-structured answers that directly address the user's question.\n\n"
        "Question: {query_str}\n"
        "Answer:"
    )
    
    qa_prompt = PromptTemplate(qa_prompt_template)
    
    # Create response synthesizer with STREAMING ENABLED
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_prompt,
        response_mode="compact",
        streaming=True  # <<< STREAMING ENABLED
    )
    
    # Create query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
    )
    
    # Store references for fallback queries
    query_engine._llm = LlamaSettings.llm
    query_engine._retriever = retriever
    query_engine._response_synthesizer = response_synthesizer
    
    return query_engine


def query_with_fallback(query_engine, question: str):
    """
    Query with fallback to general knowledge when no relevant documents found.
    """
    nodes = query_engine._retriever.retrieve(question)
    
    if not nodes:
        # No relevant documents - call LLM directly
        llm = query_engine._llm
        prompt = (
            f"Question: {question}\n\n"
            "Please provide a helpful, accurate, and well-structured answer to this question. "
            "Be informative, clear, and professional in your response.\n\n"
            "Answer:"
        )
        response_text = llm.complete(prompt).text
        
        class MockResponse:
            def __init__(self, text):
                self._text = text
                self.source_nodes = []
            def __str__(self):
                return self._text
        
        return MockResponse(response_text)
    else:
        # Documents found - synthesize response
        response = query_engine._response_synthesizer.synthesize(question, nodes)
        response.source_nodes = nodes
        return response


def query_with_fallback_streaming(query_engine, question: str):
    """
    STREAMING version of query_with_fallback.
    Yields text chunks and source nodes as they're generated.
    Returns a generator that yields (text_chunk, source_nodes) tuples.
    """
    nodes = query_engine._retriever.retrieve(question)
    
    if not nodes:
        # No relevant documents - call LLM directly with streaming
        llm = query_engine._llm
        prompt = (
            f"Question: {question}\n\n"
            "Please provide a helpful, accurate, and well-structured answer to this question. "
            "Be informative, clear, and professional in your response.\n\n"
            "Answer:"
        )
        # Use stream_complete for streaming
        stream_response = llm.stream_complete(prompt)
        for chunk in stream_response:
            yield (chunk.delta, [])  # No source nodes for general knowledge
    else:
        # Documents found - synthesize response with streaming
        # The response_synthesizer.synthesize returns StreamingResponse when streaming=True
        streaming_response = query_engine._response_synthesizer.synthesize(question, nodes)
        
        # Stream the response text
        for text_chunk in streaming_response.response_gen:
            yield (text_chunk, nodes)  # Yield text chunks with source nodes


def format_response(response) -> Dict:
    """Format the response with retrieved context."""
    raw_answer = str(response)
    
    if not raw_answer or raw_answer.strip().lower() == "empty response":
        answer = "I don't have any indexed documents to reference for this query, but I can help based on my general knowledge. Please re-ask your question and I'll do my best to assist."
    else:
        answer = raw_answer

    result = {
        "answer": answer,
        "sources": []
    }
    
    if hasattr(response, 'source_nodes'):
        for idx, node in enumerate(response.source_nodes, 1):
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
    print("RAG QUERY INTERFACE")
    print("=" * 60)
    print("Type your questions below (or 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            print(f"\nRetrieving relevant context...")
            print(f"Generating response...\n")
            
            try:
                response = query_engine.query(question)
                result = format_response(response)
            except Exception as query_error:
                error_msg = str(query_error)
                if "system memory" in error_msg or "status code: 500" in error_msg:
                    print(f"Primary model failed (insufficient memory)")
                    print(f"Retrying with fallback model: {OLLAMA_LLM_FALLBACK}...\n")
                    
                    fallback_llm = _create_ollama_llm(OLLAMA_LLM_FALLBACK)
                    LlamaSettings.llm = fallback_llm
                    
                    query_engine_fallback = create_query_engine(query_engine._index)
                    response = query_engine_fallback.query(question)
                    result = format_response(response)
                else:
                    raise
            
            # Display answer
            print("Answer:")
            print("-" * 60)
            print(result["answer"])
            print("-" * 60)
            
            # Display sources
            if result["sources"]:
                print(f"\nSources (Top {len(result['sources'])} chunks):")
                for source in result["sources"]:
                    print(f"\n   [{source['chunk_id']}] Score: {source['score']:.3f}")
                    print(f"   File: {source['metadata'].get('filename', 'Unknown')}")
                    print(f"   Preview: {source['text']}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def query_single(query_engine, question: str):
    """Query with a single question and return."""
    print(f"\nQuestion: {question}")
    print(f"\nRetrieving relevant context...")
    print(f"Generating response...\n")
    
    response = query_engine.query(question)
    result = format_response(response)
    
    print("Answer:")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    
    if result["sources"]:
        print(f"\nSources (Top {len(result['sources'])} chunks):")
        for source in result["sources"]:
            print(f"\n   [{source['chunk_id']}] Score: {source['score']:.3f}")
            print(f"   File: {source['metadata'].get('filename', 'Unknown')}")
            print(f"   Preview: {source['text']}")
    
    return result


def main():
    """Main entry point."""
    index = initialize_rag_system()
    query_engine = create_query_engine(index)
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        query_single(query_engine, question)
    else:
        query_interactive(query_engine)


if __name__ == "__main__":
    main()
