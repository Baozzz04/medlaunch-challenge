#!/bin/bash
# Simple query script - saves JSON to results/ folder
# Usage: ./query.sh "your query here"

BUCKET="rag-pipeline-niah-bucket-us-east-1"
REGION="us-east-1"
QUERY="$1"

if [ -z "$QUERY" ]; then
    echo "Usage: $0 'your query here'"
    exit 1
fi

# Create outputs folder if it doesn't exist
mkdir -p outputs

# Generate easy-to-read filename from query (sanitize for filesystem)
FILENAME=$(echo "$QUERY" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_\|_$//g' | cut -c1-60)
OUTPUT_FILE="outputs/${FILENAME}.json"

# Invoke Lambda and save to results folder
aws lambda invoke \
    --function-name query-handler \
    --cli-binary-format raw-in-base64-out \
    --payload "{\"bucket\":\"$BUCKET\",\"query\":\"$QUERY\"}" \
    --region $REGION \
    /tmp/lambda-response.json > /dev/null 2>&1

# Extract and format the actual response body
python3 << PYTHON > "$OUTPUT_FILE"
import json
import sys

try:
    with open('/tmp/lambda-response.json', 'r') as f:
        lambda_response = json.load(f)
    
    # Extract the actual response body
    if lambda_response.get('statusCode') == 200:
        body = json.loads(lambda_response.get('body', '{}'))
        # Save the clean, formatted response
        print(json.dumps(body, indent=2, ensure_ascii=False))
    else:
        # Save error response
        error_body = json.loads(lambda_response.get('body', '{}'))
        error_response = {
            "statusCode": lambda_response.get('statusCode'),
            "error": error_body.get('error', 'Unknown error'),
            "message": error_body.get('message', '')
        }
        print(json.dumps(error_response, indent=2, ensure_ascii=False))
except Exception as e:
    # Fallback: save raw response
    with open('/tmp/lambda-response.json', 'r') as f:
        print(f.read())
PYTHON

echo "Query: $QUERY"
echo "Output saved to: $OUTPUT_FILE"
cat "$OUTPUT_FILE" | python3 -m json.tool
