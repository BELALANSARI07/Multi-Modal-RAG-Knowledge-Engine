# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark: |
| < 1.1   | :x:                |

## Reporting a Vulnerability

We take the security of the **Multi-Modal RAG Knowledge Engine** seriously. If you discover a security vulnerability, please do NOT create a public issue.

Instead, please send an email to:
**belalansari000078@gmail.com**

Include the following details in your report:
- Type of vulnerability (e.g. path traversal, injection, authentication bypass, denial of service).
- Step-by-step instructions or Proof-of-Concept to reproduce the vulnerability.
- Affected endpoints or files.
- Potential impact.

### Response Timeline
- **Initial Response:** Within 24-48 hours.
- **Status Update:** Within 5 business days with triage assessment.
- **Fix Delivery:** As soon as possible depending on severity.

## Security Practices in this Codebase
- **Magic Bytes Validation:** Files are verified for true `%PDF-` header signatures rather than relying only on file extensions.
- **Path Traversal Shielding:** All uploaded filenames are stripped of relative directories, control characters, and normalized using deterministic UUID/hashes.
- **Error Shielding:** Internal database errors, file paths, and provider exception traces are sanitized before returning HTTP 500 responses to API clients.
- **API Key & Rate Limiting:** Optional token-based authentication and token-bucket rate limiting prevent resource exhaustion attacks against downstream VLM APIs.
