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
- `GET /v1/usage`
- `GET /v1/usage/{session_id}`
- `GET /health`

## Measuring Usage

- Non-streaming `POST /v1/chat/completions` responses include OpenAI-style `usage` when Bedrock returns token metadata.
- Streaming `POST /v1/chat/completions` responses can include a final usage chunk with `stream_options.include_usage`.
- `GET /v1/usage` returns process-wide in-memory request and token totals.
- `GET /v1/usage/{session_id}` returns totals for one tracked session.

Session tracking rules:

- `POST /v1/responses` automatically groups a `previous_response_id` chain into one session.
- A new `/v1/responses` conversation uses its first OpenAI response id as the implicit session id.
- `POST /v1/chat/completions` and `POST /v1/embeddings` participate in per-session totals when the request includes `X-Session-ID`.

Example:

```bash
curl -s http://localhost:8080/v1/usage

curl -s -H 'X-Session-ID: demo-session' \
  -H 'Content-Type: application/json' \
  -d '{"model":"amazon.nova-pro-v1:0","messages":[{"role":"user","content":"hi"}]}' \
  http://localhost:8080/v1/chat/completions

curl -s http://localhost:8080/v1/usage/demo-session
```

## Parity Scope

This proxy is intended for local Bedrock usage with OpenAI-compatible clients.

- Uses the AWS SDK default credential chain instead of proxy-managed auth.
- Does not add an inbound API-key layer.
- Does not add a CORS layer.
- Focuses on local runtime compatibility rather than AWS deployment workflows.
