# Challenges & Solutions

## Challenge 1: PDF Library Compatibility Issue

**Problem:**

- Initial implementation used PyMuPDF (fitz) for PDF processing
- Lambda runtime (Linux) couldn't load macOS-compiled binaries
- Error: `Runtime.ImportModuleError: /opt/python/pymupdf/_extra.so: invalid ELF header`

**Solution:**

- Switched to `pdfplumber` (pure Python library)
- No binary dependencies, works across all platforms
- Created Lambda layer with `pdfplumber` and dependencies
- Result: Reliable PDF text extraction on Lambda

## Challenge 2: Claude 3.5 Haiku Access via Inference Profile

**Problem:**

- Attempted direct model invocation: `anthropic.claude-3-5-haiku-20241022-v1:0`
- Error: `ValidationException: Invocation of model ID ... with on-demand throughput isn't supported. Retry your request with the ID or ARN of an inference profile.`
- Claude 3.5 Haiku requires inference profile, not direct model access

**Solution:**

- Used system-defined inference profile: `us.anthropic.claude-3-5-haiku-20241022-v1:0`
- Updated IAM policy to include:
  - Inference profile ARN: `arn:aws:bedrock:*::inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0`
  - Foundation model ARNs in us-east-1, us-east-2, us-west-2 (inference profile routes to multiple regions)
- Result: Successful Claude 3.5 Haiku integration for Q&A mode
