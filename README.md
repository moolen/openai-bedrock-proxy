# openai-bedrock-proxy

Codex-first OpenAI-compatible proxy to AWS Bedrock.

## Runtime Configuration

The `internal/config` package reads runtime settings from environment variables:

- `LISTEN_ADDR` (default: `0.0.0.0`)
- `LISTEN_PORT` (default: `8080`)
- `AWS_REGION` (optional)
- `LOG_LEVEL` (default: `info`)
