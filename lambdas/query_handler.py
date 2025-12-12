"""
Query Handler Lambda Function

This Lambda function handles user queries with two modes:
1. Citation Mode: Returns exact text for a specific chapter (no AI generation)
2. Q&A Mode: Answers questions using RAG (Retrieval Augmented Generation)

Citation Mode Flow:
- Detects chapter ID in query
- Searches chunks by metadata.chapter (exact match)
- Returns verbatim text with source metadata
- Falls back to vector search if chapter not found

Q&A Mode Flow:
- Generates query embedding using Titan
- Performs cosine similarity search
- Retrieves top 3 most relevant chunks
- Passes context to Claude 3.5 Haiku for answer generation
- Returns answer with citations and confidence score

Lambda Configuration:
- Memory: 512 MB (sufficient for single query processing)
- Timeout: 300 seconds (5 minutes) - allows time for RAG processing
"""

import json
import re
import math
import logging
import os
import boto3
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')

# Configuration from environment variables (with defaults)
# Model configurations
TITAN_EMBEDDING_MODEL_ID = os.getenv('TITAN_EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v1')  # For query embeddings
CLAUDE_INFERENCE_PROFILE_ID = os.getenv('CLAUDE_INFERENCE_PROFILE_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')  # For Q&A answers
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '1536'))  # Titan V1 produces 1536-dimensional vectors

# Regex pattern to match chapter IDs in queries
# Pattern handles: base prefixes (QM.1), psychiatric (PH-GR.1), sub-numbers (QM.1.2)
CHAPTER_PATTERN = re.compile(r'\b([A-Z]{2,4}(?:-[A-Z]{2})?)\.(\d+(?:\.\d+)?)\b')

_BASE_PREFIXES = [
    'QM', 'GB', 'CE', 'MS', 'NS', 'SM', 'MM', 'SS', 'AS', 'OB',
    'LS', 'RC', 'MI', 'NM', 'RS', 'ES', 'OS', 'DS', 'PR', 'IC',
    'MR', 'DC', 'UR', 'PE', 'TO', 'SB', 'TD', 'PC', 'RR', 'FS',
    'RN', 'PH'
]
_PSYCH_PREFIXES = [
    'PH-GR', 'PH-MR', 'PH-E', 'PH-NE', 'PH-TP', 'PH-PN', 'PH-DP',
    'PH-PR', 'PH-MS', 'PH-NS', 'PH-PS', 'PH-SS', 'PH-PA', 'PH-TA'
]
VALID_CHAPTER_PREFIXES = set(_BASE_PREFIXES + _PSYCH_PREFIXES)


def detect_chapter_id(query: str) -> Optional[str]:
    """
    Extracts chapter ID from a query string.
    
    Searches for chapter ID patterns (e.g., "QM.1", "LS.2", "PH-GR.1")
    and validates against known chapter prefixes.
    
    Args:
        query: User query string
        
    Returns:
        Chapter ID if found (e.g., "QM.1"), None otherwise
    """
    for match in CHAPTER_PATTERN.finditer(query.upper()):
        prefix, number = match.group(1), match.group(2)
        # Only accept valid chapter prefixes (not random text)
        if prefix in VALID_CHAPTER_PREFIXES:
            return f"{prefix}.{number}"
    return None


def is_citation_mode(query: str) -> Tuple[bool, Optional[str]]:
    """
    Determines if a query is requesting citation mode (exact chapter text).
    
    Citation mode is triggered when:
    1. Query contains a valid chapter ID AND
    2. Query matches citation patterns (e.g., "show me chapter QM.1")
    3. OR query is very short (<5 words) with a chapter ID
    
    Args:
        query: User query string
        
    Returns:
        Tuple of (is_citation_mode: bool, chapter_id: Optional[str])
    """
    chapter_id = detect_chapter_id(query)
    if chapter_id:
        # Common citation request patterns
        citation_patterns = [
            r'what\s+(does|is)\s+' + re.escape(chapter_id),  # "What does QM.1 say?"
            r'show\s+(me\s+)?' + re.escape(chapter_id),      # "Show me QM.1"
            r'cite\s+' + re.escape(chapter_id),               # "Cite QM.1"
            r'^' + re.escape(chapter_id) + r'\s*$',          # Just "QM.1"
            r'get\s+' + re.escape(chapter_id),                # "Get QM.1"
            r'retrieve\s+' + re.escape(chapter_id),          # "Retrieve QM.1"
            r'chapter\s+' + re.escape(chapter_id),           # "Chapter QM.1"
        ]
        query_lower = query.lower()
        if any(re.search(p, query_lower, re.IGNORECASE) for p in citation_patterns):
            return True, chapter_id
        # Short queries with chapter ID are likely citation requests
        if len(query.split()) <= 5:
            return True, chapter_id
    return False, chapter_id


def load_embeddings_index(bucket: str) -> Dict[str, Any]:
    """
    Loads the consolidated embeddings index from S3.
    
    The index contains all chunk embeddings in a single file for efficient
    similarity search. This avoids loading individual chunk files.
    
    Args:
        bucket: S3 bucket name
        
    Returns:
        Embeddings index dictionary with embeddings array and metadata
    """
    response = s3_client.get_object(Bucket=bucket, Key="embeddings/embeddings_index.json")
    return json.loads(response['Body'].read().decode('utf-8'))


def load_chunk_by_chapter(bucket: str, embeddings_index: Dict, chapter_id: str) -> Optional[Dict]:
    """
    Finds and loads a chunk by exact chapter ID match.
    
    Searches the embeddings index for a chunk with matching chapter ID.
    Uses case-insensitive matching. Returns the first matching chunk.
    
    Args:
        bucket: S3 bucket name
        embeddings_index: Loaded embeddings index dictionary
        chapter_id: Chapter ID to search for (e.g., "QM.1")
        
    Returns:
        Chunk dictionary if found, None otherwise
    """
    chapter_id_upper = chapter_id.upper()
    for entry in embeddings_index.get('embeddings', []):
        metadata = entry.get('metadata', {})
        # Case-insensitive chapter ID matching
        if metadata.get('chapter', '').upper() == chapter_id_upper:
            chunk_id = entry.get('chunk_id')
            if not chunk_id:
                continue
            # Pad chunk ID to 3 digits (e.g., "1" -> "001")
            chunk_id_padded = f"{int(chunk_id):03d}" if chunk_id.isdigit() else chunk_id
            try:
                response = s3_client.get_object(Bucket=bucket, Key=f"chunks/chunk_{chunk_id_padded}.json")
                return json.loads(response['Body'].read().decode('utf-8'))
            except Exception as e:
                logger.error(f"Error loading chunk {chunk_id_padded}: {e}", exc_info=True)
                continue
    return None


def generate_query_embedding(query: str) -> List[float]:
    """
    Generates a vector embedding for the user query.
    
    Converts the query text into a 1536-dimensional vector using Titan Embeddings.
    This embedding is used for cosine similarity search against chunk embeddings.
    
    Args:
        query: User query string (truncated if > 40000 chars)
        
    Returns:
        1536-dimensional vector as list of floats
        
    Raises:
        ValueError if embedding dimension mismatch
        Exception if Bedrock API call fails
    """
    # Truncate very long queries (shouldn't happen in practice)
    if len(query) > 40000:
        query = query[:40000]
        logger.warning("Query truncated to 40000 characters")
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId=TITAN_EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": query})
        )
        embedding = json.loads(response['body'].read()).get('embedding', [])
        if not embedding or len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(f"Invalid embedding: expected {EMBEDDING_DIMENSION} dimensions, got {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}", exc_info=True)
        raise


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates cosine similarity between two vectors.
    
    Cosine similarity measures the angle between vectors, returning a value
    between -1 and 1. Higher values indicate more similar content.
    Formula: cos(θ) = (A·B) / (||A|| * ||B||)
    
    Args:
        vec1: First vector (query embedding)
        vec2: Second vector (chunk embedding)
        
    Returns:
        Cosine similarity score (0.0 to 1.0, where 1.0 = identical)
    """
    if len(vec1) != len(vec2) or len(vec1) == 0:
        return 0.0
    # Dot product: sum of element-wise multiplication
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    # Vector magnitudes (L2 norm)
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def find_top_chunks(query_embedding: List[float], embeddings_index: Dict, top_k: int = 3) -> List[Dict]:
    """
    Finds the top K most similar chunks using cosine similarity.
    
    Performs in-memory similarity search across all chunk embeddings.
    Returns chunks sorted by relevance (highest similarity first).
    
    Args:
        query_embedding: 1536-dimensional query vector
        embeddings_index: Loaded embeddings index dictionary
        top_k: Number of top chunks to return (default: 3)
        
    Returns:
        List of chunk dictionaries with similarity scores, sorted by relevance
    """
    similarities = []
    # Calculate similarity for each chunk embedding
    for entry in embeddings_index.get('embeddings', []):
        vector = entry.get('vector', [])
        if vector:
            similarities.append({
                'chunk_id': entry.get('chunk_id'),
                'metadata': entry.get('metadata', {}),
                'similarity': cosine_similarity(query_embedding, vector)
            })
    # Sort by similarity (highest first) and return top K
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]


def load_chunk_text(bucket: str, chunk_id: str) -> str:
    """
    Loads the text content of a chunk from S3.
    
    Args:
        bucket: S3 bucket name
        chunk_id: Chunk ID (e.g., "1" or "001")
        
    Returns:
        Chunk text content, or empty string if error
    """
    # Pad chunk ID to 3 digits for S3 key format
    chunk_id_padded = f"{int(chunk_id):03d}" if chunk_id.isdigit() else chunk_id
    try:
        response = s3_client.get_object(Bucket=bucket, Key=f"chunks/chunk_{chunk_id_padded}.json")
        return json.loads(response['Body'].read().decode('utf-8')).get('text', '')
    except Exception as e:
        logger.error(f"Error loading chunk text for {chunk_id_padded}: {e}", exc_info=True)
        return ""


def build_rag_context(bucket: str, top_chunks: List[Dict]) -> str:
    """
    Builds RAG context string from top similar chunks.
    
    Formats chunks with chapter/section headers for Claude to understand context.
    Separates chunks with clear delimiters.
    
    Args:
        bucket: S3 bucket name
        top_chunks: List of top similar chunks with metadata
        
    Returns:
        Formatted context string with all chunk texts
    """
    context_parts = []
    for chunk in top_chunks:
        metadata = chunk['metadata']
        text = load_chunk_text(bucket, chunk['chunk_id'])
        if text:
            # Format: [Chapter QM.1 - Section Name]\nText content
            context_parts.append(f"[Chapter {metadata.get('chapter', 'Unknown')} - {metadata.get('section', 'Unknown')}]\n{text}")
    return "\n\n---\n\n".join(context_parts)  # Clear separator between chunks


def call_claude_for_answer(query: str, context: str) -> Dict[str, Any]:
    """
    Calls Claude 3.5 Haiku to generate an answer using RAG context.
    
    Uses a carefully crafted system prompt to ensure:
    - Answers are based only on provided context
    - Citations are included for all references
    - Missing information is explicitly stated
    - Information is synthesized across multiple chunks
    
    Args:
        query: User's question
        context: Formatted context string with relevant chunks
        
    Returns:
        Dictionary with 'answer' (str) and 'confidence' (str: "low"/"medium"/"high")
    """
    # System prompt defines Claude's behavior and constraints
    system_prompt = """You are a helpful assistant specializing in NIAHO (National Integrated Accreditation for Healthcare Organizations) Standards. 
Your role is to answer questions accurately based on the provided context from the NIAHO Standards document.

Guidelines:
1. ONLY answer based on the provided context - do not use external knowledge
2. You may summarize and synthesize information across multiple chunks to provide a comprehensive answer
3. REQUIRED: Cite specific chapters/sections (e.g., "According to QM.1..." or "As stated in Chapter IC.3...") when referencing standards
4. If the context doesn't contain enough information to answer the question, explicitly state: "The provided context does not contain sufficient information to answer this question"
5. Be concise but thorough
6. Maintain accuracy - if information is unclear or conflicting across chunks, note this in your response"""

    user_prompt = f"""Context from NIAHO Standards (multiple chunks may be provided):
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the context provided above
- You may synthesize information from multiple chunks if relevant
- MUST cite specific chapters/sections (e.g., "According to QM.1..." or "Chapter IC.3 states...")
- If the context does not contain enough information, clearly state this
- Provide a clear, accurate answer with proper citations"""

    response = bedrock_runtime.invoke_model(
        modelId=CLAUDE_INFERENCE_PROFILE_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.3
        })
    )
    
    response_body = json.loads(response['body'].read())
    answer = response_body.get('content', [{}])[0].get('text', '') if response_body.get('content') else ''
    
    return {"answer": answer, "confidence": determine_confidence(answer)}


def determine_confidence(answer: str) -> str:
    """
    Determines confidence level based on answer content.
    
    Analyzes answer text for phrases indicating certainty or uncertainty.
    Used to help users understand answer reliability.
    
    Args:
        answer: Generated answer text from Claude
        
    Returns:
        Confidence level: "low", "medium", or "high"
    """
    answer_lower = answer.lower()
    # Phrases indicating low confidence (missing information)
    low_phrases = ["i don't have enough information", "the context doesn't", "i cannot find", 
                   "not mentioned in the context", "unclear from the context", "i'm not certain"]
    # Phrases indicating high confidence (specific citations)
    high_phrases = ["according to chapter", "the standard states", "as specified in", "clearly indicates"]
    
    if any(phrase in answer_lower for phrase in low_phrases):
        return "low"
    if any(phrase in answer_lower for phrase in high_phrases):
        return "high"
    return "medium"


def handle_citation_mode(bucket: str, chapter_id: str, query: str = None) -> Dict[str, Any]:
    """
    Handles citation mode: returns exact chapter text without AI generation.
    
    Flow:
    1. Tries exact match by chapter ID in metadata
    2. If found: returns verbatim text with source metadata
    3. If not found: falls back to vector search and returns similar chunks
    
    Args:
        bucket: S3 bucket name
        chapter_id: Chapter ID to retrieve (e.g., "QM.1")
        query: Original user query (optional, for response metadata)
        
    Returns:
        Response dictionary with exact_text, source, disclaimer, etc.
    """
    try:
        embeddings_index = load_embeddings_index(bucket)
        # Step 1: Try exact match by chapter ID
        chunk = load_chunk_by_chapter(bucket, embeddings_index, chapter_id)
        
        if chunk:
            metadata = chunk.get('metadata', {})
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "query": query or f"Show me chapter {chapter_id}",
                    "query_type": "citation",
                    "chapter": chapter_id,
                    "exact_text": chunk.get('text', ''),
                    "source": {
                        "document": metadata.get('document', os.getenv('DOCUMENT_NAME', 'NIAHO Standards')),
                        "section": metadata.get('section', 'Unknown'),
                        "chapter": chapter_id,
                        "chunk_id": chunk.get('chunk_id', 'Unknown')
                    },
                    "disclaimer": f"Exact text from {os.getenv('DOCUMENT_NAME', 'NIAHO Standards')} document - retrieved {timestamp}"
                }, ensure_ascii=False)
            }
        else:
            # Step 2: Fallback to vector search if exact match fails
            logger.info(f"Chapter {chapter_id} not found via exact match, falling back to vector search...")
            query_text = query or f"chapter {chapter_id}"
            query_embedding = generate_query_embedding(query_text)
            top_chunks = find_top_chunks(query_embedding, embeddings_index, top_k=3)
            
            if not top_chunks:
                return {
                    "statusCode": 404,
                    "body": json.dumps({
                        "query_type": "citation",
                        "chapter": chapter_id,
                        "error": f"Chapter {chapter_id} not found in the index",
                        "message": "The requested chapter could not be found, and no similar content was found"
                    })
                }
            
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            fallback_results = []
            for chunk_info in top_chunks:
                chunk_id = chunk_info['chunk_id']
                chunk_text = load_chunk_text(bucket, chunk_id)
                metadata = chunk_info.get('metadata', {})
                fallback_results.append({
                    "chunk_id": chunk_id,
                    "chapter": metadata.get('chapter', 'Unknown'),
                    "section": metadata.get('section', 'Unknown'),
                    "text": chunk_text,
                    "relevance_score": round(chunk_info['similarity'], 4)
                })
            
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "query": query or f"Show me chapter {chapter_id}",
                    "query_type": "citation",
                    "chapter": chapter_id,
                    "exact_match": False,
                    "message": f"Chapter {chapter_id} not found via exact match. Returning most relevant content using vector search:",
                    "fallback_results": fallback_results,
                    "source": {
                        "document": os.getenv('DOCUMENT_NAME', 'NIAHO Standards'),
                        "retrieval_method": "vector_search_fallback"
                    },
                    "disclaimer": f"Content retrieved via similarity search - {timestamp}"
                }, ensure_ascii=False)
            }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e), "message": "Failed to retrieve citation"})}


def handle_qa_mode(bucket: str, query: str) -> Dict[str, Any]:
    """
    Handles Q&A mode: answers questions using RAG (Retrieval Augmented Generation).
    
    Flow:
    1. Generates query embedding using Titan
    2. Performs cosine similarity search to find top 3 relevant chunks
    3. Builds context string from chunks
    4. Calls Claude 3.5 Haiku to generate answer with citations
    5. Returns answer with citations and confidence score
    
    Args:
        bucket: S3 bucket name
        query: User's question
        
    Returns:
        Response dictionary with answer, citations, confidence
    """
    try:
        # Step 1: Generate embedding for the query
        logger.info("Generating query embedding...")
        query_embedding = generate_query_embedding(query)
        
        # Step 2: Load embeddings index for similarity search
        logger.info("Loading embeddings index...")
        embeddings_index = load_embeddings_index(bucket)
        
        # Step 3: Find top 3 most similar chunks
        logger.info("Finding similar chunks...")
        top_chunks = find_top_chunks(query_embedding, embeddings_index, top_k=3)
        
        if not top_chunks:
            logger.warning("No relevant chunks found for query")
            return {
                "statusCode": 404,
                "body": json.dumps({
                    "query_type": "question",
                    "error": "No relevant chunks found",
                    "message": "Could not find relevant context for your question"
                })
            }
        
        # Step 4: Build context string from chunks
        logger.info("Building RAG context...")
        context = build_rag_context(bucket, top_chunks)
        
        # Step 5: Generate answer using Claude with RAG context
        logger.info("Calling Claude for answer...")
        claude_response = call_claude_for_answer(query, context)
        
        # Step 6: Format citations with relevance scores
        default_document = os.getenv('DOCUMENT_NAME', 'NIAHO Standards')
        citations = [{
            "chunk_id": chunk['chunk_id'],
            "document": chunk['metadata'].get('document', default_document),
            "section": chunk['metadata'].get('section', 'Unknown'),
            "chapter": chunk['metadata'].get('chapter', 'Unknown'),
            "relevance_score": round(chunk['similarity'], 4)  # Cosine similarity score
        } for chunk in top_chunks]
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "query": query,
                "query_type": "question",
                "answer": claude_response['answer'],
                "citations": citations,
                "confidence": claude_response['confidence']
            }, ensure_ascii=False)
        }
        
    except Exception as e:
        logger.error(f"Error in Q&A mode: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e), "message": "Failed to process question"})}


def lambda_handler(event: dict, context) -> dict:
    """
    Lambda handler for query processing.
    
    Main entry point that routes queries to appropriate mode:
    - Citation Mode: Returns exact chapter text (no AI)
    - Q&A Mode: Answers questions using RAG with Claude
    
    Expected event format:
    {
        "bucket": "my-bucket-name",
        "query": "What are the requirements for quality management?"
    }
    
    Lambda Configuration:
    - Memory: 512 MB (sufficient for single query processing)
    - Timeout: 300 seconds (5 minutes) - allows time for RAG processing
    
    Args:
        event: Lambda event dictionary
        context: Lambda context object
        
    Returns:
        Response dictionary with statusCode and body
    """
    try:
        bucket = event.get('bucket')
        query = event.get('query', '').strip()
        
        if not bucket:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing 'bucket' parameter"})}
        if not query:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing 'query' parameter"})}
        
        logger.info(f"Processing query: {query}")
        # Detect query type (citation vs Q&A)
        citation_mode, chapter_id = is_citation_mode(query)
        
        if citation_mode and chapter_id:
            # Route to citation mode: return exact text
            logger.info(f"Citation mode detected for chapter: {chapter_id}")
            return handle_citation_mode(bucket, chapter_id, query)
        else:
            # Route to Q&A mode: generate answer with RAG
            logger.info("Q&A mode detected")
            return handle_qa_mode(bucket, query)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e), "message": "Failed to process query"})}

