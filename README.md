# Phase 0 RAG Project - "Dry Run" / Study / Proof of Concept

## Overview

This project is a local and secure AI platform that is powered by a RAG (Retrieval Augmented Generation) LLM engine system using LlamaIndex and Ollama.

### Project Context

- I have a v1 MVP prototype of a local RAG LLM system built for my low-power laptop (limited RAM/GPU, e.g., 16GB RAM, integrated graphics). Its source code is the basis for this project which we will build and improve on - to accommodate the improved hardware we have access to now (see below for details).
- Hardware constraints in v1 MVP: Small models (7B-13B quantized), limited context (4K-8K tokens), batch size 1, CPU-only or basic GPU offload, handling ~10-50GB data max.
- Performance: ~10-20 tokens/sec inference, basic RAG faithfulness, no advanced routing/multi-agent.

## Hardware Specifications

**GMKtec EVO-X2 AI Mini PC** with:

- AMD Ryzen™ AI Max+ 395 (Strix Halo APU)
- 16 Zen 5 CPU cores / 32 threads
- Radeon 8060S integrated GPU (40 RDNA 3.5 CUs, performance comparable to RTX 4060–4070 laptop)
- XDNA2 NPU (~50 TOPS)
- 64–128GB LPDDR5X-8000 unified memory (huge shared pool for CPU/GPU/NPU)
- 1–2TB fast PCIe 4.0 SSD
- Full ROCm support on Windows (as of late 2025)
- Physical performance mode button for sustained higher TDP

This hardware can comfortably run 70B–120B+ quantized models at high speed, support 32K–128K context windows, handle hundreds of GB of ingested documents, and perform fast embedding generation and retrieval.

## Current MVP (V1) Architecture

### Technology Stack

**Core Framework:**

- **LlamaIndex** - RAG orchestration and query engine
- **ChromaDB** - Persistent vector storage
- **Ollama** - Local LLM inference server
- **Flask** - Web application server
- **SQLite** - Document tracking database

**Models:**

- **LLM Primary:** llama3:latest (via Ollama)
- **LLM Fallback:** deepseek-r1:latest (for OOM scenarios)
- **Embeddings:** nomic-embed-text (via OllamaEmbedding)

**Configuration:**

- Chunk size: 1024 tokens
- Chunk overlap: 128 tokens
- Top-K retrieval: 5 chunks
- Similarity threshold: 0.3

### MVP Components

**Core Modules:**

- `app.py` - Flask web server with HTTP Basic Auth, structured logging, API endpoints
- `query.py` - RAG query engine with custom prompts, similarity filtering, LLM fallback handling
- `ingest.py` - Document ingestion pipeline with incremental updates (skip already-ingested files)
- `document_tracker.py` - SQLite-based tracking system to prevent duplicate ingestion
- `document_loaders.py` - Multi-format document loaders with extensible architecture
- `db_manager.py` - CLI utility for database inspection and management
- `config.py` - Centralized configuration management
- `index.html` - Web UI frontend for queries

**Supported Document Formats:**

- Text: `.txt`, `.md`
- Documents: `.pdf`, `.docx`
- Data: `.csv`, `.json`, `.xlsx`, `.xls`
- Web: `.html`, `.htm`

### MVP Features

**Production-Ready Capabilities:**

- ✅ Multi-user authentication (JSON file or environment-based)
- ✅ Structured logging with rotation (5MB files, 10 backups)
- ✅ Incremental document ingestion (hash-based deduplication)
- ✅ Web UI with real-time query interface
- ✅ Source attribution with similarity scores
- ✅ Custom QA prompt templates
- ✅ Similarity threshold filtering
- ✅ LLM fallback on OOM errors
- ✅ Database management CLI tools
- ✅ Request tracking and performance metrics

**Current Limitations:**

- Single-threaded ingestion
- No async query processing
- Basic vector search (no hybrid search or reranking)
- No query caching
- No evaluation metrics or monitoring dashboard
- Limited to CPU/basic GPU (no ROCm optimization yet)
- No multi-hop reasoning or agent capabilities

## Software Environment

**Base:**

- Windows 11
- Python 3.13.1
- Ollama (local inference server)

## V2 Upgrade Path

### Performance & Scale Goals

**Hardware Utilization:**

- ✅ Migrate to GMKtec EVO-X2 (64-128GB unified memory, Radeon 8060S iGPU, XDNA2 NPU)
- 🎯 Enable ROCm acceleration for LLM inference (40-80+ tokens/sec target)
- 🎯 Leverage NPU for embedding generation (XDNA2 ~50 TOPS)
- 🎯 Utilize unified memory for large context windows (32K-128K tokens)
- 🎯 Support 3-5 concurrent users without degradation

**Model Upgrades:**

- 🎯 Deploy 70B-120B quantized models (e.g., Qwen2.5 72B Q4/Q5, Llama 3.1 70B)
- 🎯 Evaluate frontier models for RAG faithfulness vs. speed tradeoffs
- 🎯 Test specialized embedding models (e.g., bge-large, e5-mistral)

**Data Scale:**

- 🎯 Scale from ~10-50GB to 100-500GB+ document corpus
- 🎯 Optimize ChromaDB for large-scale retrieval
- 🎯 Implement batch embedding generation for faster ingestion

### Advanced RAG Techniques

**Retrieval Improvements:**

- 🎯 **Hybrid search**: Combine dense (vector) + sparse (BM25/keyword) retrieval
- 🎯 **Reranking**: Add cross-encoder reranker (e.g., bge-reranker) after initial retrieval
- 🎯 **Multi-query retrieval**: Generate multiple query variations for better recall
- 🎯 **Hypothetical document embeddings (HyDE)**: Generate hypothetical answers, embed them, retrieve
- 🎯 **Parent-document retrieval**: Retrieve small chunks, return larger parent context
- 🎯 **Query routing**: Route queries to specialized indices or models based on intent

**Chunking Strategy:**

- 🎯 Implement semantic chunking (split on topic boundaries, not fixed tokens)
- 🎯 Add hierarchical chunking (summaries + detailed chunks)
- 🎯 Experiment with sliding window vs. sentence-aware splitting

**Context Enhancement:**

- 🎯 Add metadata filtering (date, document type, source)
- 🎯 Implement citation tracking and provenance
- 🎯 Add multi-hop reasoning for complex queries

### Production Features

**Performance:**

- 🎯 Async ingestion pipeline (background workers)
- 🎯 Query result caching (Redis or in-memory)
- 🎯 Concurrent query processing
- 🎯 Streaming responses for better UX

**Monitoring & Evaluation:**

- 🎯 RAG evaluation metrics (faithfulness, relevance, answer quality)
- 🎯 Performance dashboard (latency, throughput, cache hit rate)
- 🎯 Query analytics (popular queries, failure modes)
- 🎯 Cost tracking (tokens used, compute time)

**UI/UX Improvements:**

- 🎯 Modern React-based UI (replace basic HTML)
- 🎯 Conversation history and context management
- 🎯 Document upload via web interface
- 🎯 Real-time ingestion status
- 🎯 Source document viewer with highlighting

**Security & Multi-tenancy:**

- 🎯 Role-based access control (RBAC)
- 🎯 Per-user document collections
- 🎯 API rate limiting
- 🎯 Audit logging

### Technical Implementation Priorities

#### Phase 1: Hardware Optimization (Immediate)

1. Set up ROCm on Windows for Ollama
2. Benchmark 70B models (Qwen2.5, Llama 3.1, DeepSeek)
3. Test NPU acceleration for embeddings
4. Optimize for unified memory (large context windows)

#### Phase 2: Advanced RAG (Short-term)

1. **Hybrid search**: Implement BM25Retriever + QueryFusionRetriever with reciprocal rank fusion (implement hybrid search, dense)
2. **Reranking**: Add bge-reranker-v2-m3 cross-encoder (retrieve 10, rerank to 3) (add reranking layer)
3. **Semantic chunking**: Use SemanticSplitterNodeParser for topic-aware splitting (semantic splitting to improve chunking strategy)
4. **Parent-document retrieval**: Retrieve small chunks (512), return large context (2048)
5. **HyDE**: Add HyDEQueryTransform for complex queries
6. **Metadata filtering**: Enable date/type/source filtering in ChromaDB
7. Query routing and multi-query retrieval (??? redundant???)

#### Phase 3: Production Features (Medium-term)

1. **Async ingestion**: RQ (Redis Queue) for background document processing (with progress tracking)
2. **Query caching**: diskcache for persistent caching (simpler than Redis) (query caching layer)
3. **Streaming responses**: LlamaIndex native streaming for better UX
4. **Evaluation**: RAGAS framework (faithfulness, relevance, precision, recall) (evaluation framework and metrics)
5. **Monitoring**: Streamlit dashboard for latency/throughput/quality metrics
6. **Query routing**: Route to specialized indices based on document type (if needed)

#### Phase 4: Scale & Polish (Long-term)

1. **Multi-user RBAC**: Flask-Security-Too for role-based access control
2. **Modern UI**: HTMX (simpler, Python-focused) or Svelte (if rich UI needed) as opposed to original idea (modern react ui)
3. **Conversation management**: ChatMemoryBuffer with SQLite storage
4. **Advanced analytics**: Query patterns, failure modes, user behavior tracking

### Principles

- ✅ Keep everything 100% local, open-source, and private
- ✅ No external API dependencies or cloud services
- ✅ Optimize specifically for AMD hardware (ROCm, NPU, unified memory)
- ✅ Maintain low operational cost
- ✅ Prioritize RAG quality and faithfulness over speed
