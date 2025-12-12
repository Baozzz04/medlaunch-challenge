# NIAHO Standards RAG Pipeline

Serverless RAG (Retrieval Augmented Generation) pipeline on AWS for processing NIAHO Standards PDFs and answering questions.

## Time Log

Approximate hours spent on each component:

- **Setup & Infrastructure**: 30 minutes

  - AWS account setup, IAM roles, S3 bucket configuration
  - Lambda function creation and layer setup
  - Environment variables configuration

- **PDF Processor**: 1 hour

  - PDF text extraction implementation
  - Chapter detection and chunking logic
  - S3 integration and error handling

- **Embedding Generator**: 1 hour

  - Titan Embeddings integration
  - Batch processing and retry logic
  - Embeddings index creation

- **Query Handler**: 2 hours

  - Citation mode implementation
  - Q&A mode with RAG pipeline
  - Claude 3.5 Haiku integration
  - Vector similarity search
  - Fallback mechanisms

- **Debugging**: 1 hour

**Total Development Time**: ~6 hours

## Architecture

- **PDF Processor** (Lambda): Extracts text, chunks by chapter (500-1500 tokens), stores in S3
- **Embedding Generator** (Lambda): Generates 1536-dim embeddings using Amazon Titan V1
- **Query Handler** (Lambda): Handles queries in two modes:
  - **Citation Mode**: Returns exact chapter text (no AI)
  - **Q&A Mode**: Answers questions using Claude 3.5 Haiku with RAG

## Prerequisites

- AWS CLI configured with appropriate credentials
- AWS account with access to:
  - Lambda, S3, Bedrock (Titan V1, Claude 3.5 Haiku)
- Python 3.9+ (for local testing)

## Setup

### 1. Create S3 Bucket

```bash
aws s3 mb s3://rag-pipeline-niah-bucket-us-east-1 --region us-east-1
```

### 2. Upload PDF

```bash
aws s3 cp your-pdf.pdf s3://rag-pipeline-niah-bucket-us-east-1/raw/niaho_standards.pdf --region us-east-1
```

### 3. Deploy Lambda Functions

```bash
# Package functions
cd lambdas
zip pdf_processor.zip pdf_processor.py
zip embedding_generator.zip embedding_generator.py
zip query_handler.zip query_handler.py

# Upload to Lambda (update function names as needed)
aws lambda update-function-code --function-name pdf-processor --zip-file fileb://pdf_processor.zip --region us-east-1
aws lambda update-function-code --function-name embedding-generator --zip-file fileb://embedding_generator.zip --region us-east-1
aws lambda update-function-code --function-name query-handler --zip-file fileb://query_handler.zip --region us-east-1
```

### 4. Configure Environment Variables

```bash
# PDF Processor
aws lambda update-function-configuration \
  --function-name pdf-processor \
  --environment Variables="{DOCUMENT_NAME=NIAHO Standards,PDF_KEY=raw/niaho_standards.pdf}" \
  --region us-east-1

# Embedding Generator
aws lambda update-function-configuration \
  --function-name embedding-generator \
  --environment Variables="{TITAN_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1,EMBEDDING_DIMENSION=1536}" \
  --region us-east-1

# Query Handler
aws lambda update-function-configuration \
  --function-name query-handler \
  --environment Variables="{TITAN_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1,CLAUDE_INFERENCE_PROFILE_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0,EMBEDDING_DIMENSION=1536,DOCUMENT_NAME=NIAHO Standards}" \
  --region us-east-1
```

### 5. Set Lambda Memory & Timeout

- **pdf-processor**: 3008 MB, 900s
- **embedding-generator**: 512 MB, 900s
- **query-handler**: 512 MB, 300s

### 6. Run Pipeline

```bash
# Step 1: Process PDF
aws lambda invoke --function-name pdf-processor \
  --region us-east-1 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"bucket":"rag-pipeline-niah-bucket-us-east-1"}' \
  /tmp/pdf_response.json

# Step 2: Generate Embeddings
aws lambda invoke --function-name embedding-generator \
  --region us-east-1 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"bucket":"rag-pipeline-niah-bucket-us-east-1"}' \
  /tmp/embedding_response.json
```

## Usage

### Query via Script

```bash
./query.sh "Show me chapter QM.1"           # Citation mode
./query.sh "What are quality requirements?" # Q&A mode
```

Results are saved to `results/` folder with timestamps.

### Query via AWS CLI

```bash
aws lambda invoke --function-name query-handler \
  --region us-east-1 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"bucket":"rag-pipeline-niah-bucket-us-east-1","query":"Show me chapter QM.1"}' \
  /tmp/response.json
```

## Lambda Layers

- **pdfplumber**: Required for `pdf-processor`
  - Create layer with `pdfplumber` package
  - Attach to `pdf-processor` function

## IAM Permissions

Lambda execution role needs:

- S3 read/write access to bucket
- Bedrock `InvokeModel` for:
  - `amazon.titan-embed-text-v1`
  - `us.anthropic.claude-3-5-haiku-20241022-v1:0` (inference profile)
  - Foundation model ARNs in us-east-1, us-east-2, us-west-2

## Cost Estimate

- Setup (one-time): ~$0.27
- Per query: ~$0.0002-0.0011
- 50 test queries: ~$0.03
- **Total: <$0.30** (well within $5 budget)

## View Logs

```bash
# CloudWatch Logs
aws logs tail /aws/lambda/query-handler --follow --region us-east-1
aws logs tail /aws/lambda/pdf-processor --follow --region us-east-1
aws logs tail /aws/lambda/embedding-generator --follow --region us-east-1
```

Or via AWS Console: CloudWatch → Log groups → `/aws/lambda/{function-name}`

## Project Structure

```
.
├── lambdas/
│   ├── pdf_processor.py       # PDF processing & chunking
│   ├── embedding_generator.py # Embedding generation
│   └── query_handler.py       # Query processing (Citation & Q&A)
├── query.sh                   # Query script
└── results/                    # Query results (JSON)
```
