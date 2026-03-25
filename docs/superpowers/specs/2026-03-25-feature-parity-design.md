# AWS Sample Feature Parity Design

**Date:** 2026-03-25

**Goal:** Bring this local OpenAI-compatible Bedrock proxy to practical feature parity with `aws-samples/bedrock-access-gateway` for local usage with `AWS_*` credentials, while preserving the existing Responses-first architecture.

## Scope

This parity effort targets local runtime behavior only.

Included:

- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `GET /v1/models/{id}`
- `GET /health`
- richer `GET /v1/models` model discovery
- multimodal chat inputs for supported Bedrock models
- advanced Bedrock request controls used by the AWS sample:
  - `top_p`
  - `stop`
  - `reasoning_effort`
  - `extra_body`
  - prompt caching controls
- support for foundation models, cross-region inference profiles, and application inference profiles
- preservation of the existing `POST /v1/responses` and `GET /v1/models` behavior

Excluded:

- AWS deployment templates or deployment docs
- inbound API-key enforcement
- CORS configuration
- exact duplication of AWS sample quirks that do not improve local compatibility
- replacing the existing Responses API with Chat Completions

## Product Intent

The proxy should remain a small local Bedrock adapter rather than a full clone of the AWS sample.

The parity target is:

- API compatibility for the AWS sample's documented local use cases
- Bedrock capability coverage comparable to the AWS sample
- a Go implementation that stays coherent with the current codebase

The proxy should continue to:

- rely on the AWS SDK default credential chain
- use incoming Bedrock model IDs directly by default
- keep Responses as a first-class API surface

## Architecture

### 1. Converse-First Multi-API Shim

The existing `internal/bedrock` package should remain the single Bedrock integration layer.

Chat-style inference should continue to flow through Bedrock `Converse` and `ConverseStream`. The new Chat Completions API should be added as a second OpenAI-facing surface over the same underlying Bedrock translation logic rather than forcing everything through one wire shape.

This keeps the architecture aligned with the current repository:

- `internal/openai`
  - request/response types
  - validation
  - SSE serialization
- `internal/proxy`
  - API-specific orchestration
- `internal/bedrock`
  - Bedrock request translation
  - Bedrock response translation
  - model discovery
  - embeddings adapters
- `internal/httpserver`
  - routing and HTTP handlers

### 2. Keep Responses Intact

The current Responses implementation already has conversation normalization, tool-capable Bedrock translation, streaming text deltas, and `previous_response_id` continuation. That work should stay intact.

Parity work should add adjacent paths instead of rewriting the Responses stack around Chat Completions semantics.

### 3. Add Separate Chat And Embeddings Paths

Chat Completions and Embeddings should each get dedicated OpenAI-facing types and validation. They should not be squeezed into `openai.ResponsesRequest`.

Chat requests can reuse the normalized internal Bedrock conversation model where that helps, but chat-specific validation and response shaping should remain separate because:

- Chat Completions uses `messages`, not `input`
- Chat streaming emits chunk semantics rather than Responses events
- tool call, usage, and finish-reason output shapes differ

Embeddings should use a separate Bedrock path built on `InvokeModel`, because Bedrock embedding providers do not share a single Converse-style request format.

### 4. Add A Model Catalog Layer

The current models path only lists text-capable foundation models. That is not enough for parity because multimodal checks, inference-profile support, and capability-specific validation need more metadata.

Add a small model catalog sublayer inside `internal/bedrock` that can merge:

- foundation models from `ListFoundationModels`
- system-defined inference profiles
- application inference profiles

The catalog should expose enough metadata for:

- provider display
- input modalities
- streaming support where relevant
- model ID type
- underlying foundation model resolution for capability checks

### 5. Keep Boundaries Small

The main complexity should be isolated into three bounded areas:

- chat translation and chunk shaping
- embeddings model-family adapters
- model catalog and capability checks

This keeps the plan implementable without turning `internal/bedrock` into a single large parity file.

## Endpoint Design

### `POST /v1/chat/completions`

Add OpenAI-compatible Chat Completions support with the subset needed for parity with the AWS sample.

Supported request concepts:

- `messages`
  - roles: `system`, `developer`, `user`, `assistant`, `tool`
- user content as plain string or structured text/image arrays
- `stream`
- `stream_options.include_usage`
- `temperature`
- `top_p`
- `stop`
- `max_tokens`
- `max_completion_tokens`
- `tools`
- `tool_choice`
- `reasoning_effort`
- `extra_body`

Response behavior:

- non-streaming responses return OpenAI chat completion payloads
- streaming responses return chat-completion chunk payloads over SSE
- tool use maps to `tool_calls`
- Bedrock finish reasons map to OpenAI finish reasons
- usage fields are returned when available

Chat requests should be stateless and request-scoped. Unlike Responses, Chat Completions should not introduce proxy-managed continuation state.

### `POST /v1/embeddings`

Add OpenAI-compatible embeddings support for the practical subset backed by Bedrock in the AWS sample.

Initial support should cover:

- Cohere embedding models
- Titan text embedding models
- Nova embedding models with optional `dimensions`

Response behavior:

- return OpenAI-compatible embedding list payloads
- support `encoding_format` values used by the AWS sample
- reject model/input combinations that Bedrock does not support instead of silently coercing them

### `GET /v1/models`

Upgrade the current models endpoint to return a refreshed merged catalog rather than only text foundation models.

The returned set should include:

- foundation model IDs
- cross-region inference profile IDs
- application inference profile ARNs where resolvable

### `GET /v1/models/{id}`

Add a single-model lookup endpoint backed by the same catalog used by `GET /v1/models`.

This endpoint should:

- return a model object when the ID is present
- return a client error when the model is not in the catalog

### `GET /health`

Add a minimal health endpoint for local process checks.

This endpoint should not perform a live Bedrock call.

## Request Translation

### Chat Conversation Mapping

Chat requests should translate into the existing internal Bedrock conversation representation where possible.

Normalization rules:

- `system` and `developer` messages become Bedrock system content
- `user` and `assistant` messages become Bedrock conversation messages
- `tool` messages become Bedrock `toolResult` content blocks in a synthetic `user` message
- assistant tool calls from prior messages become Bedrock `toolUse` content blocks

The implementation should preserve order across mixed content so that multimodal and tool-capable conversations survive normalization without semantic drift.

### Multimodal Input Mapping

For user messages with structured content arrays:

- text items map to Bedrock text blocks
- `image_url` items map to Bedrock image blocks

Supported image sources:

- remote image URLs
- `data:` URLs for local workflows

The proxy should validate that the resolved Bedrock model supports image input before sending the request upstream.

### Advanced Inference Controls

Expand the Bedrock request builder to support:

- `top_p`
- `stop`
- `reasoning_effort`
- `extra_body`
- prompt caching control fields

Rules:

- unsupported combinations should fail explicitly when they materially affect behavior
- provider-specific Bedrock request fields should be added through a narrow translation layer rather than leaking raw OpenAI request structs into Bedrock calls
- `extra_body` should be passed through conservatively, with local filtering for proxy-owned control keys such as prompt-caching toggles

### Reasoning

Add model-aware reasoning support aligned with the AWS sample's practical behavior:

- map `reasoning_effort` to Bedrock provider-specific request fields when the target model family supports it
- avoid silently enabling reasoning for models where the semantics are unknown
- expose reasoning output in the OpenAI-compatible response shape used by the selected API surface

The exact mapping should be capability-driven by resolved underlying model metadata rather than hard-coded against only the presented model ID string.

### Prompt Caching

Add prompt caching support for supported models.

Controls should allow:

- global runtime enablement
- per-request opt-in or opt-out via request metadata

Prompt caching should be injected as Bedrock cache points into:

- system content
- user message content

only when:

- the model supports prompt caching
- caching is enabled
- the request shape can be translated safely

## Model Catalog And Capability Checks

The model catalog should become the source of truth for capability-aware validation.

Each model entry should expose at least:

- external ID
- provider
- display name
- input modalities
- whether it is a foundation model or inference profile
- resolved underlying foundation model ID

This metadata should drive:

- multimodal validation
- prompt-caching support checks
- reasoning support checks
- profile-aware model resolution

The catalog should live behind a narrow interface so request handlers and proxy services do not need to know how Bedrock discovery is implemented.

## Embeddings Design

Embeddings should be implemented as model-family adapters under `internal/bedrock`.

Rationale:

- Cohere, Titan, and Nova use different request and response bodies
- dimensions support is provider-specific
- token accounting differs across providers

Each adapter should own:

- request-body construction
- Bedrock invocation
- response parsing
- OpenAI-compatible usage shaping

The top-level embeddings service should select the adapter based on the requested Bedrock model ID.

## Response Translation

### Chat Non-Streaming

Bedrock `Converse` output should translate into OpenAI chat completion responses:

- text content becomes `message.content`
- tool use becomes `message.tool_calls`
- finish reasons map to OpenAI finish reasons
- usage fields are populated from Bedrock usage metadata when available

### Chat Streaming

Bedrock streaming events should translate into OpenAI chat completion chunks.

The stream path should support:

- text deltas
- tool call start and tool call argument deltas
- final finish reason
- optional usage chunk when requested

Streaming fidelity should be improved without disturbing the existing Responses streaming contract.

### Responses

Responses should keep their current event and payload model.

Where Bedrock capability improvements are shared between Chat Completions and Responses, the implementation should extract common translation code underneath the API-specific response shapers.

## State And Lifecycle

### Responses State

Keep `previous_response_id` continuation only for the Responses API.

The existing in-memory response store remains the correct place for that state.

### Chat State

Chat Completions should remain stateless. No proxy-managed continuation state should be added.

### Model Catalog Refresh

The catalog may be refreshed on demand or cached briefly in memory, but it should remain behind an interface so refresh behavior can be tuned later without changing handlers.

## Error Handling

Principles:

- reject malformed requests with explicit `400` errors
- reject unsupported model/capability combinations explicitly
- preserve distinct HTTP behavior between client validation failures and upstream Bedrock failures
- do not silently drop request fields that materially change behavior

Non-goals:

- perfect wire-level replication of every AWS sample edge case
- permissive fallback behavior that hides Bedrock incompatibilities

## Testing Strategy

The implementation should stay fully unit-testable offline.

Add coverage at four layers:

- `internal/openai`
  - chat validation
  - embeddings validation
- `internal/bedrock`
  - model catalog discovery and profile resolution
  - chat request translation
  - multimodal translation
  - embeddings adapters
  - advanced request field mapping
  - chat streaming chunk shaping
- `internal/proxy`
  - orchestration
  - error handling
  - API-specific service paths
- `internal/httpserver`
  - endpoint wiring
  - request decoding
  - response serialization

Core correctness should not depend on live AWS integration tests.

## Rollout Order

Implement parity in this order:

1. shared model catalog and capability plumbing
2. Chat Completions non-streaming path
3. Chat Completions streaming chunk path
4. multimodal chat input
5. advanced Bedrock request features
6. embeddings endpoint and adapters
7. `GET /v1/models/{id}` and `/health`
8. README updates for local usage and parity notes

This order keeps the work incremental and avoids building feature-specific validation before the capability model exists.
