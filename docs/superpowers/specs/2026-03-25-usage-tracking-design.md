# Usage Tracking Design

**Goal:** Add built-in token usage measurement to the local Bedrock proxy so users can inspect request, session, and process-wide token totals without external instrumentation.

## Scope

- Capture Bedrock token usage for non-streaming and streaming chat completions.
- Capture Bedrock token usage for `/v1/responses` requests, including streaming.
- Preserve existing embeddings usage and feed it into the same tracker.
- Support overall process totals.
- Support per-session totals.
- Keep all tracking in memory.

## Session Model

- Requests may opt into explicit session tracking with the `X-Session-ID` header.
- `/v1/responses` requests also gain implicit sessions through `previous_response_id`.
- For a new `/v1/responses` conversation without `X-Session-ID`, the stored session id is the first OpenAI response id.
- Continuations inherit the stored session id from the previous response record.
- Chat completions and embeddings only participate in per-session totals when `X-Session-ID` is provided.

## API Surface

- Add `GET /v1/usage` for process-wide totals.
- Add `GET /v1/usage/{session_id}` for per-session totals.
- Keep chat completion streaming `usage` chunks unchanged.
- Add non-streaming chat completion `usage` fields when Bedrock provides them.
- Do not change the current `/v1/responses` payload shape.

## Data Model

- Bedrock conversation responses carry normalized token usage internally.
- Conversation records store `SessionID` so `previous_response_id` chains can resolve to one session.
- A new in-memory tracker stores:
  - overall request count and token totals
  - per-session request count and token totals

## Error Handling

- Missing usage from Bedrock is treated as "not recordable" and does not create synthetic counts.
- Unknown session ids on `GET /v1/usage/{session_id}` return a not-found error.

## Testing

- Verify non-stream chat completions include `usage`.
- Verify responses and streamed responses record usage in the tracker.
- Verify chat and embeddings record usage under explicit `X-Session-ID`.
- Verify `GET /v1/usage` and `GET /v1/usage/{session_id}` return stable JSON payloads.
