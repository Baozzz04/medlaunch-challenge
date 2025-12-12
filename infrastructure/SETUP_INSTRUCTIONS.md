# Infrastructure Setup Instructions

## Option 1: CloudFormation (Recommended)

### Deploy Stack

```bash
# Deploy with auto-generated unique bucket name
aws cloudformation create-stack \
  --stack-name rag-pipeline-infrastructure \
  --template-body file://cloudformation-template.yaml \
  --parameters ParameterKey=ProjectName,ParameterValue=niah \
  --region us-east-1

# Or with custom suffix
aws cloudformation create-stack \
  --stack-name rag-pipeline-infrastructure \
  --template-body file://cloudformation-template.yaml \
  --parameters ParameterKey=ProjectName,ParameterValue=niah \
               ParameterKey=BucketNameSuffix,ParameterValue=prod \
  --region us-east-1
```

**Note**: Bucket name will be auto-generated as: `rag-pipeline-{project}-{account-id}-{region}`

### Check Status

```bash
aws cloudformation describe-stacks \
  --stack-name rag-pipeline-infrastructure \
  --region us-east-1
```

### Get Outputs

```bash
aws cloudformation describe-stacks \
  --stack-name rag-pipeline-infrastructure \
  --query 'Stacks[0].Outputs' \
  --region us-east-1
```

## Option 2: Manual Setup

### 1. Create S3 Bucket

```bash
# Get your account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-east-1"
PROJECT="niah"

# Create bucket with unique name
BUCKET_NAME="rag-pipeline-${PROJECT}-${ACCOUNT_ID}-${REGION}"
aws s3 mb s3://${BUCKET_NAME} --region ${REGION}
```

### 2. Create IAM Role

```bash
# Create role (use unique name to avoid conflicts)
ROLE_NAME="rag-lambda-role-$(date +%s)"
aws iam create-role \
  --role-name ${ROLE_NAME} \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'
```

### 3. Attach Basic Execution Policy

```bash
aws iam attach-role-policy \
  --role-name ${ROLE_NAME} \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### 4. Create Custom Policy

```bash
# IMPORTANT: Update iam-policy.json with your actual bucket name first!
# Replace "YOUR-BUCKET-NAME" with your bucket name from step 1

# Create policy from iam-policy.json
aws iam put-role-policy \
  --role-name ${ROLE_NAME} \
  --policy-name rag-lambda-policy \
  --policy-document file://iam-policy.json
```

**Note**: Update `iam-policy.json` and replace `YOUR-BUCKET-NAME` with your actual bucket name before running step 4.

### 5. Create S3 Folder Structure

```bash
# Use the bucket name from step 1
# Create folders (S3 doesn't need explicit folders, but this documents structure)
aws s3api put-object --bucket ${BUCKET_NAME} --key raw/ --region ${REGION}
aws s3api put-object --bucket ${BUCKET_NAME} --key chunks/ --region ${REGION}
aws s3api put-object --bucket ${BUCKET_NAME} --key embeddings/ --region ${REGION}
```

## Verify Setup

```bash
# Check bucket exists
aws s3 ls s3://${BUCKET_NAME} --region ${REGION}

# Check IAM role exists
aws iam get-role --role-name ${ROLE_NAME}

# Check policy attached
aws iam get-role-policy \
  --role-name ${ROLE_NAME} \
  --policy-name rag-lambda-policy
```

## Next Steps

After infrastructure is set up:

1. Upload PDF to S3: `raw/niaho_standards.pdf`
2. Create Lambda functions (see README.md)
3. Attach IAM role to Lambda functions
4. Configure environment variables
5. Run pipeline
