# OpenAI Bedrock Proxy Design

**Date:** 2026-03-24

**Goal:** Build a Go proxy that exposes OpenAI-compatible endpoints for Codex-first usage and forwards supported inference requests to Amazon Bedrock using the AWS default credential chain.

## Scope

The first milestone targets Codex compatibility through the OpenAI Responses API.

Included in the current baseline:

- `POST /v1/responses`
- Streaming responses over Server-Sent Events
- `GET /v1/models`
- Direct use of the incoming `model` value as the Bedrock `modelId`
- AWS authentication via the AWS SDK for Go v2 default credential chain
- No inbound API-key enforcement

Excluded from the current baseline:

- Alias maps between OpenAI-facing model names and Bedrock model IDs
- Custom AWS credential refresh endpoints
- Realtime / WebSocket support
- Non-inference OpenAI APIs that Bedrock cannot realistically back for Codex-first usage

## Compatibility Roadmap

The next compatibility work should proceed in this order:

1. Structured `input` items and `previous_response_id`
2. Function tools and tool call / tool output items
3. Richer Responses streaming event fidelity, then Realtime / WebSocket support

This ordering is intentional. Codex-first compatibility is currently blocked more by item-based request shapes and incremental response continuation than by tools or richer stream event taxonomies.

## Product Constraints

- The proxy must ignore inbound `Authorization` bearer tokens.
- The proxy must rely on standard AWS configuration sources such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, shared config files, and other providers supported by the AWS SDK default chain.
- The proxy should behave predictably for Codex clients, prioritizing correctness over broad but partial API emulation.
- Unsupported OpenAI fields should either be ignored safely or rejected explicitly when dropping them would materially change behavior.

## Architecture

The project should be structured as a small Go service with three main layers:

1. `cmd/openai-bedrock-proxy`
   - Process startup
   - HTTP server wiring
   - Configuration loading
   - Graceful shutdown

2. `internal/openai`
   - OpenAI-compatible request parsing
   - Responses API validation
   - SSE serialization for streaming output
   - Error response shaping

3. `internal/bedrock`
   - AWS SDK client construction
   - Request translation from OpenAI Responses payloads to Bedrock Converse / ConverseStream calls
   - Response translation from Bedrock events back to OpenAI-style JSON / SSE events

Supporting packages may include:

- `internal/config` for runtime configuration such as listen address and AWS region override
- `internal/httpserver` for routing and handlers
- `internal/translate` if request and response mapping grows large enough to justify a neutral translation package

The translation logic should remain independent from HTTP transport so Realtime / WebSocket support can be added later without duplicating Bedrock mapping logic.

## Endpoint Behavior

### `POST /v1/responses`

The handler accepts an OpenAI Responses API request and translates the supported subset into Bedrock Converse inputs.

Supported request concepts in the current baseline:

- Text input
- Streaming mode
- Basic inference controls such as token limits and temperature when available

Supported request concepts in compatibility slice 1:

- Text input supplied either as a raw string or as structured message items
- Conversation history that can be represented as `user`, `assistant`, `system`, and `developer` text messages
- `previous_response_id` continuation using proxy-managed in-memory state
- Streaming mode
- Basic inference controls such as token limits and temperature when available

Expected behavior:

- Validate the request body and reject malformed payloads with `400`
- Use the incoming `model` field directly as the Bedrock `modelId`
- Build Bedrock system and conversation messages from the OpenAI request body
- Call `Converse` for non-streaming requests
- Call `ConverseStream` for streaming requests
- Translate Bedrock output into an OpenAI-style Responses payload

#### Structured Input and Continuation

Compatibility slice 1 expands request handling from plain string input to the subset of item-based Responses input most likely to be emitted by Codex clients.

Accepted forms:

- `input` as a plain string
- `input` as a single message object
- `input` as an array of message objects
- message objects in either "easy message" form such as `{ "role": "user", "content": "hello" }` or explicit item form such as `{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hello" }] }`
- message `role` values of `user`, `assistant`, `system`, or `developer`
- content supplied either as a raw string or as an array of text blocks
- accepted text block types:
  - `input_text` for `user`, `system`, and `developer`
  - `output_text` for `assistant`
  - raw string content for any accepted role

Normalization rules:

- `developer` is normalized as Bedrock system context, the same as `system`
- `input_text` and `output_text` blocks are flattened to plain text in encounter order
- mixed text blocks are concatenated with no synthetic separators
- any non-text block type is rejected in slice 1

Concrete accepted examples:

```json
{
  "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
  "input": "Find the bug in parseConfig()"
}
```

```json
{
  "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
  "input": {
    "type": "message",
    "role": "user",
    "content": [
      { "type": "input_text", "text": "Find the bug in parseConfig()" }
    ]
  }
}
```

```json
{
  "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
  "previous_response_id": "resp_123",
  "instructions": "Be concise.",
  "input": [
    { "role": "assistant", "content": "The parser fails on empty sections." },
    {
      "type": "message",
      "role": "user",
      "content": [
        { "type": "input_text", "text": "Patch it without changing behavior." }
      ]
    }
  ]
}
```

Rejected in slice 1:

- multimodal content
- function call items
- function call output items
- tool definitions
- `tool_choice`
- `parallel_tool_calls`

`previous_response_id` should be implemented with an internal response store. When present, the proxy loads the stored normalized conversation snapshot for the referenced response, appends only the new request input, rebuilds the current turn's system context from the new request, and sends the merged result to Bedrock.

If `previous_response_id` is unknown, the proxy should return `400` instead of silently dropping history.

#### System Context Precedence

System context should be deterministic and turn-local.

Rules:

- Only `user` and `assistant` conversation turns are persisted for `previous_response_id` continuation
- `system`, `developer`, and `instructions` values are not inherited from the previously stored response
- For a new request, normalized `system` and `developer` messages from the incoming `input` are collected in encounter order
- If `instructions` is present, it is appended after those messages as the final system block, giving it highest precedence in the current turn
- The Bedrock request's `System` field is built only from the current request's `system`, `developer`, and `instructions` inputs

### Streaming

For streaming requests, the proxy should emit SSE events compatible with OpenAI Responses streaming semantics as closely as practical.

The proxy should:

- Flush partial text as it arrives from Bedrock stream events
- Emit only text-oriented events in the current baseline and slice 1
- Terminate the stream cleanly with completion events
- Surface stream-time translation failures as structured error events where possible, and otherwise terminate the stream

Tool-shaped streaming events remain deferred to slice 2 and richer event fidelity remains deferred to slice 3.

### `GET /v1/models`

The models endpoint should be intentionally minimal in v1.

Preferred initial behavior:

- Return an OpenAI-compatible list payload
- Provide enough structure for clients that probe the endpoint
- Avoid introducing model-discovery logic until a concrete client requires it

This endpoint may return a conservative static payload instead of querying Bedrock model catalogs in v1.

## Translation Strategy

The main translation problem is converting OpenAI Responses request items into Bedrock Converse messages and converting Bedrock outputs back into OpenAI Responses structures.

Rules for the current implementation:

- Support only the subset required for Codex-first use
- Prefer explicit unsupported-field errors over lossy silent coercion
- Keep translation rules small, testable, and table-driven

Translation responsibilities after compatibility slice 1:

- Normalize raw string input and structured `message` items into a single internal conversation model
- Extract `system` and `developer` messages from structured input and merge them with turn-local `instructions`
- Convert `user` and `assistant` turns into Bedrock message content blocks
- Load and append prior normalized conversation state from `previous_response_id`
- Map Bedrock stop reasons into OpenAI completion / response status fields
- Build stable SSE event sequences from Bedrock streaming events

The translation layer should introduce a normalized conversation representation that is independent from both the raw OpenAI JSON payload and the Bedrock SDK types. That model becomes the boundary between request parsing, state persistence, and Bedrock translation.

## Response State

Compatibility slice 1 requires process-local response state.

The proxy should define a small response-store interface and provide an in-memory implementation keyed by OpenAI-style response ID.

Each stored record should include:

- response ID
- requested model ID
- normalized conversation snapshot after the assistant reply, containing only persisted `user` and `assistant` turns
- creation timestamp

Persistence rules:

- Store the snapshot only after a successful non-streaming response
- Store the snapshot only after a streaming response has produced a coherent assistant result and completed cleanly
- Do not store partial or failed turns
- Keep the store bounded with a fixed-capacity FIFO policy in slice 1
- Use a default maximum of 4096 stored responses and evict the oldest entry on overflow

This store is intentionally best-effort for now. A process restart loses state, and subsequent `previous_response_id` lookups should fail explicitly with `400`.

## Validation Policy

Slice 1 should use a strict supported-subset policy with predictable wire behavior:

- Explicitly reject known unsupported fields or item types that would materially change behavior
- Accept and translate only the documented slice-1 request subset in this spec
- Ignore unknown top-level JSON fields that are not used by translation, matching Go's normal JSON decoding behavior
- Reject unknown structured `input` item shapes and unknown content block types with `400`

Explicit reject list for slice 1:

- `tools`
- `tool_choice`
- `parallel_tool_calls`
- non-message input items
- non-text content blocks
- multimodal inputs

## Configuration

The service should require minimal configuration.

Environment variables or flags should cover:

- Listen address
- Listen port
- Optional AWS region override
- Optional log level

The service should load AWS credentials with `config.LoadDefaultConfig` from the AWS SDK for Go v2.

## Error Handling

Errors should be classified consistently:

- `400 Bad Request`
  - Invalid JSON
  - Missing required fields
  - Unsupported request features that would change semantics if ignored
  - Unknown `previous_response_id`
  - Structured input outside the supported slice-1 subset

- `500 Internal Server Error`
  - Local translation bugs
  - Response-store failures in local state handling
  - Unexpected runtime failures in the proxy

- `502 Bad Gateway`
  - Bedrock service failures
  - AWS authentication or signing issues after request acceptance

Error payloads should be OpenAI-compatible where practical so clients receive recognizable failure shapes.

## Testing Strategy

Implementation should follow TDD.

Priority test layers:

1. Translation unit tests
   - OpenAI request to Bedrock request mapping
   - Bedrock response to OpenAI response mapping
   - Structured message normalization
   - Easy-message and explicit-item input forms
   - `developer` and `system` normalization into Bedrock system context
   - `instructions` precedence over incoming system context
   - `previous_response_id` snapshot append semantics
   - Stop-reason conversion

2. HTTP handler tests
   - `/v1/responses` request validation
   - Non-streaming success path
   - Streaming SSE path
   - Unknown `previous_response_id`
   - `/v1/models` response shape
   - Overflow eviction behavior for the in-memory response store

3. End-to-end style tests with a mocked Bedrock interface
   - Codex-relevant request payloads
   - Item-based input payloads
   - Continuation requests using `previous_response_id`
   - Stream event sequencing
   - Error propagation

The Bedrock client should be abstracted behind an interface so tests can avoid live AWS calls.

## Deferred Work

The following are intentionally deferred until after the Responses-first milestone is stable:

- Function tools and tool call / tool output items
- Richer Responses streaming event fidelity
- Realtime / WebSocket support
- Additional OpenAI compatibility endpoints
- Dynamic model discovery
- Alias mapping between public model names and Bedrock targets
- Expanded multimodal support beyond what Codex-first scenarios require

## Open Questions

The current design intentionally leaves these implementation details flexible until coding begins:

- Whether `/v1/models` needs static placeholders or dynamic behavior for the target client
- How much of the OpenAI Responses event taxonomy should be reproduced verbatim versus approximated cleanly for Codex compatibility in slice 3
