# System Architecture

## Overview

Serverless RAG pipeline on AWS that processes NIAHO Standards PDFs and answers questions using vector embeddings and LLM generation.

## Architecture Diagram

```
┌─────────────────┐
│   PDF Upload    │
│   (S3/raw/)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Lambda: PDF Processor  │
│  • Extract text         │
│  • Detect chapters      │
│  • Chunk (500-1500 tok) │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  S3: chunks/    │
│  (JSON files)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Lambda: Embedding Gen   │
│  • Titan V1 (1536-dim)  │
│  • Generate embeddings  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  S3: embeddings/        │
│  (embeddings_index.json)│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Lambda: Query Handler  │
│  ┌─────────────────────┐│
│  │ Citation Mode       ││
│  │ • Exact match       ││
│  │ • Fallback: vector  ││
│  └─────────────────────┘│
│  ┌─────────────────────┐│
│  │ Q&A Mode            ││
│  │ • Vector search     ││
│  │ • Claude 3.5 Haiku  ││
│  └─────────────────────┘│
└────────┬────────────────┘
         │
         ▼
    Response
```

## Components

### 1. PDF Processor (Lambda)
- **Input**: PDF from S3 `raw/` folder
- **Process**: 
  - Extract text using `pdfplumber`
  - Detect chapter boundaries via regex
  - Split into chunks (500-1500 tokens)
  - Preserve metadata (document, section, chapter)
- **Output**: JSON chunks in S3 `chunks/` folder
- **Config**: 3008 MB, 900s timeout

### 2. Embedding Generator (Lambda)
- **Input**: Chunks from S3 `chunks/`
- **Process**:
  - Generate 1536-dim embeddings (Titan V1)
  - Skip existing embeddings (idempotent)
  - Create consolidated index
- **Output**: 
  - Updated chunks with embeddings
  - `embeddings_index.json` for fast loading
- **Config**: 512 MB, 900s timeout

### 3. Query Handler (Lambda)
- **Input**: User query + bucket name
- **Modes**:
  - **Citation**: Exact chapter text (no AI)
  - **Q&A**: RAG with Claude 3.5 Haiku
- **Process**:
  - Detect query type
  - Vector similarity search (cosine)
  - Generate answer with citations
- **Output**: JSON response with answer/citations
- **Config**: 512 MB, 300s timeout

## Data Flow

1. **Setup Phase** (one-time):
   - PDF → PDF Processor → Chunks → Embedding Generator → Embeddings Index

2. **Query Phase** (on-demand):
   - Query → Query Handler → Vector Search → LLM → Response

## Storage

- **S3 Structure**:
  ```
  bucket/
  ├── raw/              # Original PDF
  ├── chunks/           # Processed chunks (JSON)
  └── embeddings/       # Embeddings index (JSON)
  ```

## Models

- **Embeddings**: Amazon Titan Embeddings G1 (`amazon.titan-embed-text-v1`)
  - 1536 dimensions
  - Max 8192 tokens input
  
- **LLM**: Claude 3.5 Haiku (`us.anthropic.claude-3-5-haiku-20241022-v1:0`)
  - Via inference profile
  - Used for Q&A mode only

## Security

- IAM roles with least privilege
- No credentials in code (env vars only)
- S3 bucket policies
- CloudWatch logging

## Scalability

- **Serverless**: Auto-scales with demand
- **Stateless**: Each Lambda invocation independent
- **Idempotent**: Safe to re-run embedding generation
- **Cost-effective**: Pay per invocation

## Cost

- Setup: ~$0.27 (one-time)
- Per query: ~$0.0002-0.0011
- 50 queries: ~$0.03
- **Total: <$0.30** (within $5 budget)

