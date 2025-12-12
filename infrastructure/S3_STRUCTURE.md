# S3 Bucket Structure

## Bucket Naming Convention

**Pattern**: `rag-pipeline-{project}-{account-id}-{region}`

**Example**: `rag-pipeline-niah-123456789012-us-east-1`

- `rag-pipeline`: Prefix indicating RAG pipeline
- `{project}`: Project identifier (e.g., "niah" for NIAHO)
- `{account-id}`: AWS Account ID (ensures uniqueness)
- `{region}`: AWS region (e.g., "us-east-1")

**Optional Suffix**: Can add `-{suffix}` for additional uniqueness (e.g., `-prod`, `-dev`)

## Folder Structure

```
bucket-name/
├── raw/                          # Original PDF files
│   └── niaho_standards.pdf      # Source document
│
├── chunks/                       # Processed text chunks
│   ├── chunk_001.json           # Chunk with metadata
│   ├── chunk_002.json
│   └── ...
│
└── embeddings/                   # Vector embeddings
    └── embeddings_index.json     # Consolidated index (all embeddings)
```

## Object Structure

### Chunk JSON (`chunks/chunk_XXX.json`)

```json
{
  "chunk_id": "001",
  "text": "Full chunk text content...",
  "metadata": {
    "document": "NIAHO Standards",
    "section": "Quality Management",
    "chapter": "QM.1"
  },
  "token_count": 1250,
  "embedding": [0.123, -0.456, ...]  // 1536 dimensions
}
```

### Embeddings Index (`embeddings/embeddings_index.json`)

```json
{
  "embeddings": [
    {
      "chunk_id": "001",
      "vector": [0.123, -0.456, ...],
      "metadata": {
        "document": "NIAHO Standards",
        "section": "Quality Management",
        "chapter": "QM.1"
      }
    },
    ...
  ],
  "total_chunks": 170,
  "embedding_dimension": 1536,
  "model": "amazon.titan-embed-text-v1"
}
```

## Naming Conventions

- **Chunks**: `chunk_{ID}.json` where ID is zero-padded 3 digits (001, 002, ..., 170)
- **PDFs**: `{document_name}.pdf` in `raw/` folder
- **Index**: `embeddings_index.json` in `embeddings/` folder

## Access Patterns

- **Read**: All Lambda functions read from `chunks/` and `embeddings/`
- **Write**:
  - PDF Processor writes to `chunks/`
  - Embedding Generator writes to `chunks/` and `embeddings/`
- **Delete**: PDF Processor clears `chunks/` before processing new PDF
