"""
PDF Processor Lambda Function

This Lambda function processes NIAHO Standards PDF documents:
1. Downloads PDF from S3
2. Extracts text using pdfplumber
3. Detects chapter boundaries using regex patterns
4. Splits text into semantically meaningful chunks (500-1500 tokens)
5. Stores chunks in S3 with metadata (document, section, chapter)

Lambda Configuration:
- Memory: 3008 MB (required for large PDF processing - 450+ pages)
- Timeout: 900 seconds (15 minutes) - allows time for full PDF processing
"""

import json
import re
import os
import logging
import boto3
import pdfplumber

# Configure logging for CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

# Regex pattern to match chapter IDs (e.g., "QM.1", "LS.2", "PH-GR.1")
# Pattern: [A-Z]{2,4} = 2-4 uppercase letters (chapter prefix)
#          \. = literal dot
#          \d+(?:\.\d+)? = number with optional sub-number (e.g., "1.2")
CHAPTER_PATTERN = re.compile(r'\b([A-Z]{2,4})\.(\d+(?:\.\d+)?)\b')

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


def download_pdf_from_s3(bucket: str, key: str, local_path: str) -> str:
    """
    Downloads a PDF file from S3 to local filesystem.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key (path to PDF)
        local_path: Local file path to save the PDF
        
    Returns:
        Local file path where PDF was saved
    """
    s3_client.download_file(bucket, key, local_path)
    return local_path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from all pages of a PDF file.
    
    Uses pdfplumber library which preserves text layout and formatting better
    than other PDF libraries. Processes each page sequentially.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Complete text content of the PDF as a single string
    """
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


def detect_chapters(text: str) -> list:
    """
    Detects chapter boundaries in the PDF text.
    
    Scans through the text line by line, looking for chapter ID patterns.
    Filters out Table of Contents entries and duplicate chapter IDs.
    Returns list of (chapter_id, position) tuples sorted by position.
    
    Args:
        text: Full PDF text content
        
    Returns:
        List of tuples: [(chapter_id, byte_position), ...]
        Example: [("QM.1", 1234), ("QM.2", 5678), ...]
    """
    chapters = []
    lines = text.split('\n')
    current_pos = 0
    
    # Scan each line for chapter ID patterns
    for line in lines:
        match = CHAPTER_PATTERN.match(line.strip())
        if match:
            prefix, number = match.group(1), match.group(2)
            # Only accept valid chapter prefixes (not random text)
            if prefix in VALID_CHAPTER_PREFIXES:
                chapter_id = f"{prefix}.{number}"
                # Skip Table of Contents entries (they have page numbers/dots)
                if not is_toc_entry(line.strip()):
                    chapters.append((chapter_id, current_pos))
        current_pos += len(line) + 1
    
    # Remove duplicate chapter IDs (keep first occurrence)
    seen = set()
    return [(cid, pos) for cid, pos in chapters if not (cid in seen or seen.add(cid))]


def is_toc_entry(line: str) -> bool:
    """
    Determines if a line is a Table of Contents entry.
    
    TOC entries typically have:
    - Dots/ellipsis before page numbers
    - Page numbers at the end
    - Short lines (< 200 chars)
    
    Args:
        line: Text line to check
        
    Returns:
        True if line appears to be a TOC entry, False otherwise
    """
    return ('...' in line or '. . .' in line or 
            re.search(r'\.{2,}\s*\d+\s*$', line) or
            (re.search(r'\s+\d{1,3}\s*$', line) and len(line) < 200))


def count_tokens(text: str) -> int:
    """
    Estimates token count by counting words.
    
    Uses simple word count as approximation (1 word ≈ 0.75 tokens).
    This is sufficient for chunking purposes.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated token count (word count)
    """
    return len(text.split())


def split_into_chunks(text: str, chapters: list) -> list:
    """
    Splits PDF text into semantically meaningful chunks.
    
    Strategy:
    1. Split by chapter boundaries (preserves semantic meaning)
    2. Target chunk size: 500-1500 tokens (optimal for embeddings)
    3. Merge small chunks (<300 tokens) with neighbors
    4. Re-split large chunks (>1800 tokens) by paragraphs
    5. Handle preamble text before first chapter
    
    Args:
        text: Full PDF text content
        chapters: List of (chapter_id, position) tuples from detect_chapters()
        
    Returns:
        List of chunk dictionaries with keys: chapter, text, section
    """
    chunks = []
    # Target range: 500-1500 tokens per chunk (optimal for embeddings)
    MAX_TOKENS_PER_CHUNK = 1800  # Upper bound - chunks larger than this get re-split
    MIN_TOKENS_PER_CHUNK = 300   # Lower bound - chunks smaller than this get merged
    
    if not chapters:
        return split_text_by_size(text, "FULL", MAX_TOKENS_PER_CHUNK)
    
    sorted_chapters = sorted(chapters, key=lambda x: x[1])
    
    for i, (chapter_id, start_pos) in enumerate(sorted_chapters):
        end_pos = sorted_chapters[i + 1][1] if i + 1 < len(sorted_chapters) else len(text)
        chunk_text = text[start_pos:end_pos].strip()
        
        if not chunk_text or len(chunk_text) < 50:
            continue
        
        section = extract_section_name(chunk_text.split('\n')[0] if chunk_text else "", chapter_id)
        token_count = count_tokens(chunk_text)
        
        if token_count > MAX_TOKENS_PER_CHUNK:
            sub_chunks = split_text_by_size(chunk_text, chapter_id, MAX_TOKENS_PER_CHUNK)
            for j, sub_chunk in enumerate(sub_chunks):
                sub_chunk["section"] = f"{section} (Part {j+1})"
                chunks.append(sub_chunk)
        else:
            chunks.append({"chapter": chapter_id, "text": chunk_text, "section": section})
    
    if sorted_chapters and sorted_chapters[0][1] > 0:
        preamble_text = text[:sorted_chapters[0][1]].strip()
        if preamble_text and len(preamble_text) > 100:
            token_count = count_tokens(preamble_text)
            if token_count > MAX_TOKENS_PER_CHUNK:
                sub_chunks = split_text_by_size(preamble_text, "INTRO", MAX_TOKENS_PER_CHUNK)
                for j, sub_chunk in enumerate(sub_chunks):
                    sub_chunk["section"] = f"Introduction (Part {j+1})"
                    chunks.insert(j, sub_chunk)
            else:
                chunks.insert(0, {"chapter": "INTRO", "text": preamble_text, "section": "Introduction/Preamble"})
    
    # Step 3: Merge tiny chunks with neighbors to hit the 500-1500 token window
    # This ensures we don't have fragments that are too small for meaningful embeddings
    merged = []
    for chunk in chunks:
        if merged and count_tokens(chunk["text"]) < MIN_TOKENS_PER_CHUNK:
            prev = merged[-1]
            # Only merge if same chapter; otherwise keep as is (preserve chapter boundaries)
            if prev["chapter"] == chunk["chapter"]:
                prev["text"] = prev["text"] + "\n\n" + chunk["text"]
                prev["section"] = prev["section"].split(" (Part")[0]  # Remove part suffix if merging
                continue
        merged.append(chunk)

    # Step 4: If chunks are still too large, re-split by paragraphs
    # This handles cases where a single chapter is very long
    final_chunks = []
    for chunk in merged:
        tokens = count_tokens(chunk["text"])
        if tokens > MAX_TOKENS_PER_CHUNK:
            # Re-split large chunks by paragraphs to stay within token limit
            sub_chunks = split_text_by_size(chunk["text"], chunk["chapter"], MAX_TOKENS_PER_CHUNK)
            for j, sub_chunk in enumerate(sub_chunks):
                sub_chunk["section"] = f"{chunk['section']} (Part {j+1})"
                final_chunks.append(sub_chunk)
        else:
            final_chunks.append(chunk)

    return final_chunks


def split_text_by_size(text: str, chapter_id: str, max_tokens: int) -> list:
    """
    Splits text into smaller chunks when a chapter exceeds token limit.
    
    Uses paragraph boundaries to split, preserving semantic meaning.
    Accumulates paragraphs until token limit is reached, then creates a new chunk.
    
    Args:
        text: Text to split
        chapter_id: Chapter identifier (e.g., "QM.1")
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of chunk dictionaries with chapter, text, and section fields
    """
    chunks = []
    paragraphs = text.split('\n\n')  # Split by double newlines (paragraph boundaries)
    current_chunk = []
    current_tokens = 0
    part_num = 1
    
    # Accumulate paragraphs until we hit the token limit
    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current_chunk:
            # Current chunk is full, save it and start a new one
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({"chapter": chapter_id, "text": chunk_text, "section": f"Part {part_num}"})
            current_chunk = [para]
            current_tokens = para_tokens
            part_num += 1
        else:
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_tokens += para_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        if len(chunk_text.strip()) > 50:  # Only include if substantial content
            chunks.append({"chapter": chapter_id, "text": chunk_text, "section": f"Part {part_num}" if part_num > 1 else "Full"})
    
    return chunks


def extract_section_name(first_line: str, chapter_id: str) -> str:
    """
    Extracts section name from the first line of a chapter.
    
    Removes the chapter ID and cleans up formatting characters.
    Used to provide context in chunk metadata.
    
    Args:
        first_line: First line of the chapter text
        chapter_id: Chapter identifier (e.g., "QM.1")
        
    Returns:
        Section name (truncated to 100 chars) or "General" if not found
    """
    section = first_line.replace(chapter_id, "").strip()
    section = re.sub(r'^[\s\-:]+', '', section)  # Remove leading whitespace, dashes, colons
    return section[:100] if section else "General"


def format_chunk_json(chunk_data: dict, chunk_index: int) -> dict:
    """
    Formats chunk data into the standard JSON structure.
    
    Creates a chunk with:
    - chunk_id: Zero-padded 3-digit ID (e.g., "001", "042")
    - text: The actual chunk text content
    - metadata: Document hierarchy (document > section > chapter)
    - token_count: Estimated token count for the chunk
    
    Args:
        chunk_data: Dictionary with keys: chapter, text, section
        chunk_index: Sequential chunk number (1-based)
        
    Returns:
        Formatted chunk dictionary ready for S3 storage
    """
    return {
        "chunk_id": f"{chunk_index:03d}",  # Zero-padded: 001, 002, ..., 170
        "text": chunk_data["text"],
        "metadata": {
            "document": os.getenv('DOCUMENT_NAME', 'NIAHO Standards'),
            "section": chunk_data["section"],
            "chapter": chunk_data["chapter"]
        },
        "token_count": count_tokens(chunk_data["text"])
    }


def upload_chunk_to_s3(bucket: str, chunk: dict) -> None:
    """
    Uploads a single chunk to S3.
    
    Stores chunks in the 'chunks/' prefix with naming pattern:
    chunks/chunk_001.json, chunks/chunk_002.json, etc.
    
    Args:
        bucket: S3 bucket name
        chunk: Chunk dictionary to upload
    """
    s3_client.put_object(
        Bucket=bucket,
        Key=f"chunks/chunk_{chunk['chunk_id']}.json",
        Body=json.dumps(chunk, indent=2, ensure_ascii=False),
        ContentType='application/json'
    )


def clear_existing_chunks(bucket: str) -> None:
    """
    Deletes all existing chunks from S3 before processing new PDF.
    
    This ensures we don't have stale chunks from previous runs.
    Uses pagination to handle large numbers of objects.
    
    Args:
        bucket: S3 bucket name
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix='chunks/'):
        if 'Contents' in page:
            objects = [{'Key': obj['Key']} for obj in page['Contents']]
            if objects:
                s3_client.delete_objects(Bucket=bucket, Delete={'Objects': objects})


def lambda_handler(event: dict, context) -> dict:
    """
    Lambda handler for PDF processing.
    
    Main entry point that orchestrates the PDF processing pipeline:
    1. Downloads PDF from S3
    2. Extracts text
    3. Detects chapters
    4. Splits into chunks
    5. Uploads chunks to S3
    
    Expected event format:
    {
        "bucket": "my-bucket-name",
        "pdf_key": "raw/niaho_standards.pdf"  # optional, defaults to raw/niaho_standards.pdf
    }
    
    Lambda Configuration:
    - Memory: 3008 MB (required for processing large PDFs - 450+ pages)
    - Timeout: 900 seconds (15 minutes) - allows time for full PDF processing
    
    Args:
        event: Lambda event dictionary
        context: Lambda context object
        
    Returns:
        Response dictionary with statusCode and body
    """
    try:
        bucket = event.get('bucket')
        if not bucket:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing 'bucket' parameter"})}
        
        # Get PDF key from event or environment variable (default: raw/niaho_standards.pdf)
        pdf_key = event.get('pdf_key', os.getenv('PDF_KEY', 'raw/niaho_standards.pdf'))
        local_pdf_path = '/tmp/document.pdf'
        
        # Step 1: Download PDF from S3
        logger.info(f"Processing PDF: s3://{bucket}/{pdf_key}")
        download_pdf_from_s3(bucket, pdf_key, local_pdf_path)
        logger.info("PDF downloaded successfully")
        
        # Step 2: Extract text from PDF
        text = extract_text_from_pdf(local_pdf_path)
        logger.info(f"Extracted {len(text)} characters from PDF")
        
        # Step 3: Detect chapter boundaries
        chapters = detect_chapters(text)
        logger.info(f"Detected {len(chapters)} unique chapters")
        
        # Step 4: Split text into chunks (500-1500 tokens each)
        raw_chunks = split_into_chunks(text, chapters)
        logger.info(f"Created {len(raw_chunks)} raw chunks")
        
        # Step 5: Clear old chunks and upload new ones
        clear_existing_chunks(bucket)
        logger.info("Cleared existing chunks")
        
        processed_chunks = []
        for i, chunk_data in enumerate(raw_chunks, start=1):
            chunk = format_chunk_json(chunk_data, i)
            upload_chunk_to_s3(bucket, chunk)
            processed_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "chapter": chunk["metadata"]["chapter"],
                "token_count": chunk["token_count"]
            })
        
        logger.info(f"Uploaded {len(processed_chunks)} chunks to S3")
        
        # Cleanup: Remove local PDF file
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "PDF processed successfully",
                "total_chunks": len(processed_chunks),
                "chunks_summary": processed_chunks,
                "bucket": bucket,
                "source_pdf": pdf_key
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e), "message": "Failed to process PDF"})}

