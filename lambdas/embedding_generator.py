"""
Embedding Generator Lambda Function

This Lambda function generates vector embeddings for all chunks:
1. Loads chunks from S3 chunks/ folder
2. Generates 1536-dimensional embeddings using Amazon Titan Embeddings G1
3. Stores embeddings back in chunk JSONs
4. Creates consolidated embeddings_index.json for fast loading

Lambda Configuration:
- Memory: 512 MB (current) | Recommended: 1024-2048 MB for embedding operations
  Note: 512 MB works for current workload, but 1024-2048 MB is recommended
  for better performance and to handle larger batches of embeddings
- Timeout: 900 seconds (15 minutes) - allows time for processing all chunks
"""

import json
import logging
import os
import boto3
from typing import List, Dict, Any

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')

# Configuration from environment variables (with defaults)
# Amazon Titan Embeddings G1 - Text model (1536 dimensions)
TITAN_EMBEDDING_MODEL_ID = os.getenv('TITAN_EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v1')
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '1536'))  # Titan V1 produces 1536-dimensional vectors


def list_chunk_files(bucket: str) -> List[str]:
    """
    Lists all chunk JSON files in the S3 chunks/ folder.
    
    Uses pagination to handle large numbers of chunks.
    Returns sorted list of S3 keys for processing.
    
    Args:
        bucket: S3 bucket name
        
    Returns:
        Sorted list of S3 keys (e.g., ['chunks/chunk_001.json', ...])
    """
    chunk_keys = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix='chunks/'):
        if 'Contents' in page:
            chunk_keys.extend([obj['Key'] for obj in page['Contents'] if obj['Key'].endswith('.json')])
    return sorted(chunk_keys)


def load_chunk_from_s3(bucket: str, key: str) -> Dict[str, Any]:
    """
    Loads a chunk JSON file from S3.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key (e.g., 'chunks/chunk_001.json')
        
    Returns:
        Chunk dictionary with text, metadata, chunk_id, etc.
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))


def generate_embedding(text: str) -> List[float]:
    """
    Generates a vector embedding for text using Amazon Titan Embeddings.
    
    Calls the Bedrock API to convert text into a 1536-dimensional vector.
    The embedding captures semantic meaning for similarity search.
    
    Args:
        text: Text to generate embedding for (should be < 4500 tokens)
        
    Returns:
        1536-dimensional vector as list of floats
        
    Raises:
        Exception if embedding generation fails or dimension mismatch
    """
    response = bedrock_runtime.invoke_model(
        modelId=TITAN_EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    embedding = json.loads(response['body'].read()).get('embedding', [])
    if len(embedding) != EMBEDDING_DIMENSION:
        logger.warning(f"Expected {EMBEDDING_DIMENSION} dimensions, got {len(embedding)}")
    return embedding


def count_tokens(text: str) -> int:
    """
    Estimates token count from word count.
    
    Uses approximation: 1 word ≈ 0.75 tokens (average).
    This is sufficient for truncation purposes.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated token count
    """
    words = len(text.split())
    return int(words / 0.75)

def truncate_text_for_embedding(text: str, max_tokens: int = 4500) -> str:
    """
    Truncates text to fit within Titan embedding model limits.
    
    Titan Embeddings G1 has a limit of ~8192 tokens, but we use a conservative
    limit of 4500 tokens to ensure reliability. Truncates by words to preserve
    word boundaries.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed (default: 4500)
        
    Returns:
        Truncated text (or original if within limit)
    """
    tokens = count_tokens(text)
    if tokens > max_tokens:
        words = text.split()
        target_words = int(max_tokens * 0.75)  # Convert tokens back to words
        truncated_words = words[:target_words]
        truncated_text = ' '.join(truncated_words)
        logger.info(f"Truncated text from {tokens} estimated tokens to ~{max_tokens} tokens ({len(text)} to {len(truncated_text)} chars)")
        return truncated_text
    return text


def save_chunk_with_embedding(bucket: str, chunk: Dict[str, Any]) -> None:
    """
    Saves chunk back to S3 with the generated embedding included.
    
    Updates the chunk JSON file to include the 'embedding' field.
    This allows chunks to be self-contained with their embeddings.
    
    Args:
        bucket: S3 bucket name
        chunk: Chunk dictionary with embedding field added
    """
    s3_client.put_object(
        Bucket=bucket,
        Key=f"chunks/chunk_{chunk['chunk_id']}.json",
        Body=json.dumps(chunk, ensure_ascii=False),
        ContentType='application/json'
    )


def create_embeddings_index(chunks_with_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates a consolidated embeddings index for fast loading.
    
    The index contains all embeddings in a single file for efficient
    similarity search. Includes metadata for each chunk.
    
    Structure:
    {
        "embeddings": [
            {"chunk_id": "001", "vector": [...], "metadata": {...}},
            ...
        ],
        "total_chunks": 170,
        "embedding_dimension": 1536,
        "model": "amazon.titan-embed-text-v1"
    }
    
    Args:
        chunks_with_embeddings: List of chunks with embedding vectors
        
    Returns:
        Index dictionary ready for S3 storage
    """
    return {
        "embeddings": [{
            "chunk_id": chunk["chunk_id"],
            "vector": chunk.get("embedding", []),
            "metadata": chunk.get("metadata", {})
        } for chunk in chunks_with_embeddings],
        "total_chunks": len(chunks_with_embeddings),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "model": TITAN_EMBEDDING_MODEL_ID
    }


def save_embeddings_index(bucket: str, index: Dict[str, Any]) -> None:
    """
    Saves the consolidated embeddings index to S3.
    
    Stores the index at embeddings/embeddings_index.json for fast loading
    during query processing. This avoids loading individual chunk files.
    
    Args:
        bucket: S3 bucket name
        index: Embeddings index dictionary
    """
    s3_client.put_object(
        Bucket=bucket,
        Key="embeddings/embeddings_index.json",
        Body=json.dumps(index, ensure_ascii=False),
        ContentType='application/json'
    )


def lambda_handler(event: dict, context) -> dict:
    """
    Lambda handler for embedding generation.
    
    Processes all chunks sequentially:
    1. Lists all chunk files from S3
    2. For each chunk:
       - Skips if embedding already exists (idempotent)
       - Truncates text if too long
       - Generates embedding with retry logic
       - Saves chunk back to S3 with embedding
    3. Creates consolidated embeddings_index.json
    
    Expected event format:
    {
        "bucket": "my-bucket-name"
    }
    
    Lambda Configuration:
    - Memory: 512 MB (current) | Recommended: 1024-2048 MB for embedding operations
      Note: Higher memory improves performance for batch embedding generation
    - Timeout: 900 seconds (15 minutes) - allows time for all chunks
    
    Args:
        event: Lambda event dictionary
        context: Lambda context object
        
    Returns:
        Response dictionary with processing statistics
    """
    try:
        bucket = event.get('bucket')
        if not bucket:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing 'bucket' parameter"})}
        
        logger.info(f"Generating embeddings for chunks in bucket: {bucket}")
        chunk_keys = list_chunk_files(bucket)
        
        if not chunk_keys:
            logger.warning("No chunks found in S3")
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "No chunks found", "message": "Please run the PDF processor first"})
            }
        
        logger.info(f"Found {len(chunk_keys)} chunks to process")
        
        chunks_with_embeddings = []
        processed_count = 0
        error_count = 0
        
        # Process each chunk sequentially
        for key in chunk_keys:
            try:
                chunk = load_chunk_from_s3(bucket, key)
                chunk_id = chunk.get('chunk_id', 'unknown')
                
                # Skip if embedding already exists (allows re-running safely)
                if 'embedding' in chunk and chunk.get('embedding'):
                    logger.debug(f"Skipping chunk {chunk_id} - embedding already exists")
                    chunks_with_embeddings.append(chunk)
                    processed_count += 1
                    continue
                
                logger.info(f"Processing chunk {chunk_id}...")
                text = chunk.get('text', '')
                if not text:
                    logger.warning(f"Chunk {chunk_id} has no text, skipping")
                    continue
                
                # Truncate if text is too long for embedding model
                text = truncate_text_for_embedding(text)
                embedding = None
                max_retries = 3
                
                # Retry logic for Bedrock API calls (handles transient errors)
                for attempt in range(max_retries):
                    try:
                        embedding = generate_embedding(text)
                        if embedding and len(embedding) > 0:
                            break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Retry {attempt + 1}/{max_retries} for chunk {chunk_id}: {str(e)}")
                            continue
                        else:
                            raise
                
                if not embedding or len(embedding) == 0:
                    logger.warning(f"Chunk {chunk_id} got empty embedding, skipping")
                    error_count += 1
                    continue
                
                # Save chunk with embedding back to S3
                chunk['embedding'] = embedding
                save_chunk_with_embedding(bucket, chunk)
                chunks_with_embeddings.append(chunk)
                processed_count += 1
                logger.info(f"Successfully processed chunk {chunk_id}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing chunk {key}: {str(e)}", exc_info=True)
                continue
        
        # Create consolidated index for fast loading during queries
        if chunks_with_embeddings:
            embeddings_index = create_embeddings_index(chunks_with_embeddings)
            save_embeddings_index(bucket, embeddings_index)
            logger.info(f"Created embeddings index with {len(chunks_with_embeddings)} entries")
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Embeddings generated successfully",
                "total_chunks": len(chunk_keys),
                "processed": processed_count,
                "errors": error_count,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "model": TITAN_EMBEDDING_MODEL_ID,
                "bucket": bucket
            })
        }
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e), "message": "Failed to generate embeddings"})}

