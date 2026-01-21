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
# REPLACED: SimilarityPostprocessor with custom retriever that filters at retrieval time (from similarity_filter import filter_by_similarity)
# from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import get_response_synthesizer
from custom_retriever import create_filtered_retriever

# Import config
sys.path.append(os.path.dirname(__file__))
from config import (
    VECTOR_DB_DIR, COLLECTION_NAME, TOP_K,
    LLM_MODEL, LLM_FALLBACK, EMBED_MODEL, OLLAMA_BASE_URL,
    SIMILARITY_THRESHOLD, LLM_TEMPERATURE, LLM_CONTEXT_WINDOW,
    LLM_REQUEST_TIMEOUT, LLM_NUM_PREDICT, GPU_LAYERS, NUM_GPU,
    MAX_CHUNKS_IN_CONTEXT, OLLAMA_NUM_THREAD, OLLAMA_NUM_BATCH,
    EMBED_BATCH_SIZE, USE_RERANKING, RERANK_TOP_N
)

# added (OLLAMA_NUM_THREAD, OLLAMA_NUM_BATCH, EMBED_BATCH_SIZE) above (added in from the original for version) - reference note

def initialize_rag_system():
    """Initialize the RAG system with vector store and LLM."""
    print("Initializing RAG system...")
    
    # 1. Configure embedding model (with batch processing - for heavy machine, comment out embed_batch_size if not on heavy machine)!
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        embed_batch_size=EMBED_BATCH_SIZE,  # Process 64 texts per batch (128GB RAM)
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
                "num_gqa": GPU_LAYERS,  # GPU layers to offload (999 = all layers)
                "num_thread": OLLAMA_NUM_THREAD,  # 16 threads for Zen 5 cores (comment out if not on heavy machine)
                "num_batch": OLLAMA_NUM_BATCH,    # 512 batch for prompt processing (comment out if not on heavy machine)
                "num_ctx": LLM_CONTEXT_WINDOW,    # Explicit context size (comment out if not on heavy machine)
            }
        )
        print(f"   Using {LLM_MODEL} (GPU layers: {GPU_LAYERS})")
    except Exception as e:
        print(f"   Primary LLM {LLM_MODEL} unavailable: {str(e)[:100]}")
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
                    "num_gqa": GPU_LAYERS,  # GPU layers to offload (999 = all layers)
                    "num_thread": OLLAMA_NUM_THREAD,  # 16 threads for Zen 5 cores (comment out if not on heavy machine)
                    "num_batch": OLLAMA_NUM_BATCH,    # 512 batch for prompt processing (comment out if not on heavy machine)
                    "num_ctx": LLM_CONTEXT_WINDOW,    # Explicit context size (comment out if not on heavy machine)
                }
            )
            print(f"   Using fallback {LLM_FALLBACK} (GPU layers: {GPU_LAYERS})")
        except Exception as e2:
            print(f"   Fallback also failed: {str(e2)[:100]}")
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
        print(f"\n Error: Vector store not found!")
        print(f"   Please run ingest.py first to create the vector database.")
        print(f"   Collection '{COLLECTION_NAME}' does not exist in {VECTOR_DB_DIR}")
        sys.exit(1)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 4. Create index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    
    print(f" RAG system initialized")
    print(f" Embedding model: {EMBED_MODEL}")
    print(f" LLM: {LLM_MODEL}")
    print(f" Context window: {LLM_CONTEXT_WINDOW} tokens")
    print(f" GPU layers: {GPU_LAYERS if GPU_LAYERS > 0 else 'auto'}")
    print(f" Top-K retrieval: {TOP_K}")
    
    return index


def create_query_engine(index):
    """Create a query engine with retriever and response synthesis."""
    
    # Configure retriever with dynamic top-k based on context window
    # For larger context windows, we can retrieve more chunks
    effective_top_k = min(TOP_K, MAX_CHUNKS_IN_CONTEXT)
    
    # Create base retriever
    base_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=effective_top_k,
    )
    
    # REPLACED: LlamaIndex SimilarityPostprocessor with custom pure Python filtering
    # OLD CODE (commented for reference):
    # node_postprocessors = [
    #     SimilarityPostprocessor(similarity_cutoff=SIMILARITY_THRESHOLD)
    # ]
    
    # NEW CODE: We'll apply filtering manually in a custom retriever wrapper
    # This gives us full transparency over the filtering logic
    # For now, we pass empty postprocessors and handle filtering in format_response
    
    # CRITICAL FOR RAG QUALITY: Apply similarity filtering at RETRIEVAL TIME
    # This ensures the LLM only sees high-quality, relevant chunks
    # 
    # OLD APPROACH (WRONG - commented for reference):
    # - Used node_postprocessors with SimilarityPostprocessor
    # - OR filtered at display time in format_response
    # - Problem: LLM already saw low-quality chunks when generating answer
    # 
    # NEW APPROACH (CORRECT):
    # - Wrap retriever with custom FilteredRetriever
    # - Filtering happens BEFORE chunks reach the LLM
    # - LLM only sees chunks with score >= SIMILARITY_THRESHOLD
    # - Improves answer quality by preventing irrelevant context pollution
    
    retriever = create_filtered_retriever(
        base_retriever=base_retriever,
        similarity_threshold=SIMILARITY_THRESHOLD,
        verbose=True  # Show filtering stats for transparency
    )
    
    # Configure postprocessors - reranking if enabled. however, filtering is already done at retrieval time so no postprocessors are needed and will fail gracefully as it will be skipped with a warning
    node_postprocessors = []
    
    if USE_RERANKING:
        try:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            reranker = SentenceTransformerRerank(
                model="BAAI/bge-reranker-v2-m3",  # Could also use "BAAI/bge-reranker-base" or "BAAI/bge-reranker-large" as it Matches bge-m3 family Matches bge-m3 embedding model family for better semantic alignment. however the one now is specifically designed for bge-m3 - but is the best quality/bigger version. switched from the original of "cross-encoder/ms-marco-MiniLM-L-2-v2",  which was Fast, accurate reranker but not as good as bge-reranker-v2-m3 nor optimal for our use case
                top_n=RERANK_TOP_N
            )
            node_postprocessors.append(reranker)
            print(f"   Reranking enabled: top {RERANK_TOP_N} results (using BGE reranker to reorder results by relevance for final ranking via cross encoder scoring performed by the BGE reranker)")
        except ImportError:
            print("   Warning: SentenceTransformerRerank not available, skipping reranking")
            print("   Install with: pip install sentence-transformers")
    
    # Custom prompt template for better responses
    # UPDATED: Natural responses without exposing internal prompt mechanics - also allows general knowledge fallback when context is insufficient
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
    
    # OLD Instructions:
    #        "1. If the context contains relevant information to answer the question, use it and cite the source.\n"
    #        "2. If the context is empty, irrelevant, or insufficient, you MAY answer from your general knowledge.\n"
    #        "3. When answering from general knowledge (not from documents), clearly prefix your response with: '[General Knowledge]'\n"
    #        "4. When answering from the documents, prefix with: '[From Documents]'\n"
    #        "5. Be helpful and informative - do NOT refuse to answer just because documents lack the information.\n"
    #        "6. Never fabricate document content or claim something is in the documents when it isn't.\n\n"


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
    
    # Store references for fallback queries and direct synthesis
    query_engine._llm = LlamaSettings.llm
    query_engine._retriever = retriever
    query_engine._response_synthesizer = response_synthesizer
    
    return query_engine


def query_with_fallback(query_engine, question: str):
    """
    Query with fallback to general knowledge when no relevant documents found.
    
    This fixes the LlamaIndex "Empty Response" issue where the response synthesizer
    short-circuits when retriever returns no nodes, never calling the LLM at all.
    
    Pipeline verification:
    - Retriever queries ChromaDB via: FilteredRetriever → VectorIndexRetriever → ChromaVectorStore → ChromaDB
    - Same VECTOR_DB_DIR and COLLECTION_NAME used by ingest.py
    - Same EMBED_MODEL (bge-m3) for query embedding as used during ingestion
    """
    # First, retrieve nodes from ChromaDB (via FilteredRetriever chain)
    nodes = query_engine._retriever.retrieve(question)
    
    if not nodes:
        # No relevant documents found in ChromaDB after similarity filtering
        # Call LLM directly for general knowledge answer
        llm = query_engine._llm
        prompt = (
            f"Question: {question}\n\n"
            "Please provide a helpful, accurate, and well-structured answer to this question. "
            "Be informative, clear, and professional in your response.\n\n"
            "Answer:"
        )
        response_text = llm.complete(prompt).text

        # OLD prompt (above - now commented out below for reference):
        #            f"The user asked: {question}\n\n"
        #            "I searched the indexed documents but found no relevant information for this question. "
        #            "Please provide a helpful answer based on your general knowledge. "
        #            "Be informative and accurate. Start your response with '[General Knowledge]' "
        #            "to indicate this answer is not from the indexed documents.\n\n"
        
        # Create a mock response object for format_response compatibility
        class MockResponse:
            def __init__(self, text):
                self._text = text
                self.source_nodes = []
            def __str__(self):
                return self._text
        
        return MockResponse(response_text)
    else:
        # Documents found in ChromaDB - synthesize response using retrieved nodes
        # Use response synthesizer directly with already-retrieved nodes to avoid double retrieval
        response = query_engine._response_synthesizer.synthesize(question, nodes)
        # Attach source_nodes for format_response compatibility
        response.source_nodes = nodes
        return response


def format_response(response) -> Dict:
    """Format the response with retrieved context."""
    raw_answer = str(response)
    # OLD: Hard-coded refusal message caused "learned helplessness"
    # if not raw_answer or raw_answer.strip().lower() == "empty response":
    #     answer = "I could not find enough relevant information..."
    # NEW: Let the LLM handle empty context gracefully via the prompt template
    # The prompt now instructs LLM to use general knowledge when docs are insufficient
    
    # Handle empty responses
    # OLD: answer = "[General Knowledge] I don't have any indexed documents to reference for this query, but I can help based on my general knowledge. Please re-ask your question and I'll do my best to assist." 
    if not raw_answer or raw_answer.strip().lower() == "empty response":
        answer = "I don't have any indexed documents to reference for this query, but I can help based on my general knowledge. Please re-ask your question and I'll do my best to assist."
    else:
        answer = raw_answer
    
    # PROFESSIONAL QUALITY: Strip prompt artifacts from response
    # Remove common prefix markers that shouldn't be visible to users
    import re
    artifacts_to_remove = [
        r'^\[General Knowledge\]\s*',
        r'^\[From Documents\]\s*',
        r'^\[Background\]\s*',
        r'^\[Context\]\s*',
        r'^Answer:\s*',
        r'^Response:\s*',
    ]
    
    for pattern in artifacts_to_remove:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
    
    # Clean up any remaining bracketed prefixes at the start
    answer = re.sub(r'^\[[^\]]+\]\s*', '', answer)
    answer = answer.strip()

    result = {
        "answer": answer,
        "sources": []
    }
    
    # Extract source information
    # Note: Filtering already happened at retrieval time via FilteredRetriever
    # All chunks here have already passed the similarity threshold
    # This ensures the LLM only saw high-quality context when generating the answer
    if hasattr(response, 'source_nodes'):
        source_nodes = response.source_nodes
        
        # CUSTOM FILTERING: Apply our pure Python similarity filter
        # This replaces the LlamaIndex SimilarityPostprocessor
        # if apply_similarity_filter:
        #    original_count = len(source_nodes)
        #    source_nodes = filter_by_similarity(source_nodes, SIMILARITY_THRESHOLD)
        #    filtered_count = len(source_nodes)
            
            # Note: This filtering happens at display time rather than retrieval time
            # Both approaches are valid - this gives us more flexibility to adjust
            # the threshold without re-querying
        #    if filtered_count < original_count:
        #        print(f"   [Filter] Removed {original_count - filtered_count} chunks below similarity threshold {SIMILARITY_THRESHOLD}")
        
        for idx, node in enumerate(source_nodes, 1):
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
            
            # Query the system with fallback on OOM error
            try:
                response = query_engine.query(question)
                result = format_response(response)
            except Exception as query_error:
                error_msg = str(query_error)
                # Check if it's an OOM error
                if "system memory" in error_msg or "status code: 500" in error_msg:
                    print(f"Primary model failed (insufficient memory)")
                    print(f"Retrying with fallback model: {LLM_FALLBACK}...\n")
                    
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