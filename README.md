# openai-bedrock-proxy

Codex-first OpenAI-compatible proxy to AWS Bedrock.

## Runtime Configuration

The `internal/config` package reads runtime settings from environment variables:

- `LISTEN_ADDR` (default: `0.0.0.0`)
- `LISTEN_PORT` (default: `8080`)
- `AWS_REGION` (optional)
- `LOG_LEVEL` (default: `info`)

## Run

The proxy uses the AWS SDK default credential chain, including standard `AWS_*`
environment variables and shared AWS config files.

```bash
go run ./cmd/openai-bedrock-proxy
```

For local usage against Bedrock with environment credentials:

```bash
AWS_REGION=us-west-2 \
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
go run ./cmd/openai-bedrock-proxy
```

## Supported Endpoints

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `GET /v1/models`
- `GET /v1/models/{id}`
- `GET /health`

## Parity Scope

This proxy is intended for local Bedrock usage with OpenAI-compatible clients.

- Uses the AWS SDK default credential chain instead of proxy-managed auth.
- Does not add an inbound API-key layer.
- Does not add a CORS layer.
- Focuses on local runtime compatibility rather than AWS deployment workflows.
