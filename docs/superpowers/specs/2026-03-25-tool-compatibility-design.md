# Tool Compatibility Design

**Date:** 2026-03-25

**Goal:** Add non-streaming tool support to the OpenAI-compatible `POST /v1/responses` endpoint so Codex can send tool definitions, receive model tool requests translated from Bedrock, execute those tool calls locally, and send tool results back through the proxy.

## Scope

This slice adds non-streaming tool support only.

Included:

- OpenAI Responses requests with `tools`
- OpenAI Responses requests with `tool_choice`
- OpenAI Responses follow-up requests carrying tool results
- Bedrock Converse request translation for tool-capable conversations
- Bedrock Converse response translation for assistant text plus tool-use output
- `previous_response_id` continuation for conversations that include tool calls and tool results
- Codex-focused emulation of OpenAI built-in tools through synthetic Bedrock tools

Excluded from this slice:

- Streaming tool events
- Proxy-side execution of any tool
- Exact OpenAI parity for every top-level or per-tool option
- `parallel_tool_calls` unless the behavior can be represented safely without semantic drift

## Product Intent

The proxy remains a protocol adapter, not an agent runtime.

Codex remains responsible for:

- advertising tools in the request
- executing tool calls returned by the model
- sending tool results in a later request

The proxy is responsible for:

- validating the tool-capable OpenAI request subset
- translating supported tool definitions into Bedrock-compatible tool configuration
- translating Bedrock tool-use output into OpenAI/Codex-facing output items
- preserving conversation state so the tool loop survives `previous_response_id`

## Compatibility Target

The compatibility target is Codex behavior, not exact generic OpenAI wire fidelity.

This means:

- native OpenAI `function` tools should behave as directly as possible
- OpenAI built-in tools that Bedrock does not understand should be emulated through synthetic tool specifications
- the output returned to the client should still look like a Codex-usable Responses API tool request
- unsupported configurations should fail explicitly instead of being silently weakened

## Architecture

### 1. Internal Conversation Model

The current `conversation` package stores only plain text turns. That is insufficient for tool loops because the proxy needs to persist assistant tool calls and later user tool results.

Replace the text-only internal model with a normalized block model:

- `Request`
  - `System []string`
  - `Messages []Message`
  - `Tools []ToolDefinition`
  - `ToolChoice ToolChoice`
- `Message`
  - `Role string`
  - `Blocks []Block`
- `Block`
  - discriminated union with:
    - text block
    - tool call block
    - tool result block

Suggested shape:

```go
type Request struct {
	System     []string
	Messages   []Message
	Tools      []ToolDefinition
	ToolChoice ToolChoice
}

type Message struct {
	Role   string
	Blocks []Block
}

type Block struct {
	Type       BlockType
	Text       string
	ToolCall   *ToolCall
	ToolResult *ToolResult
}
```

Only `user` and `assistant` messages are persisted across `previous_response_id`. System context remains turn-local as in the current implementation.

### 2. Tool Definition Model

Normalize incoming OpenAI tools into an internal tool-neutral definition type.

Suggested shape:

```go
type ToolDefinition struct {
	Kind        ToolKind
	Name        string
	Description string
	Schema      map[string]any
	OpenAIType  string
	Metadata    map[string]any
}
```

Rules:

- `function` tools map to `KindFunction`
- built-in OpenAI tools map to `KindSyntheticBuiltIn`
- `Name` must always be stable and unique for the Bedrock request
- `Metadata` stores the original OpenAI tool config needed to reconstruct the client-facing output shape later

### 3. Synthetic Built-In Tool Adapter

Bedrock can reason over tool specifications, but it does not natively understand OpenAI built-in tool families. For these tools, the proxy will synthesize a Bedrock tool specification:

- assign a deterministic synthetic function name
- derive a JSON-schema input contract appropriate for the tool family
- keep the original built-in tool config in metadata

Examples:

- `web_search_preview` -> synthetic tool with a query-shaped schema
- `code_interpreter` -> synthetic tool with metadata describing that the target client-side executor is code interpreter
- `computer_use_preview` -> synthetic tool with metadata describing the requested computer-use session shape
- other supported built-in OpenAI tool families follow the same pattern

The proxy does not execute these tools. It only translates them so the Bedrock model can choose them and Codex can later run them.

### 4. Bedrock Translation Layer

The Bedrock request translator should consume the normalized conversation model instead of the raw OpenAI request.

Responsibilities:

- convert system strings into Bedrock system blocks
- convert text blocks into Bedrock text content blocks
- convert tool call blocks from prior assistant turns into Bedrock tool-use content blocks
- convert tool result blocks from follow-up user turns into Bedrock tool-result content blocks
- convert normalized tool definitions into `ToolConfiguration`
- map `tool_choice` where possible

The response translator should:

- parse Bedrock assistant text blocks
- parse Bedrock tool-use blocks
- reconstruct OpenAI/Codex-facing output items
- restore original built-in tool identity from stored synthetic metadata

## Request Normalization

### Accepted Additions

In addition to the currently supported text-only message forms, accept:

- top-level `tools`
- top-level `tool_choice`
- follow-up tool result items in `input`

The first slice should continue to reject:

- malformed tool definitions
- unknown tool item shapes
- multimodal payloads unrelated to supported tool loops
- `parallel_tool_calls`

### Tool Definitions

Normalize OpenAI tool definitions as follows:

- `type:"function"` -> direct function tool
- built-in tool types -> synthetic built-in tool definitions

Built-in tool support should be table-driven by tool type so the proxy can reject unsupported built-in tools with precise errors instead of generic failures.

### Tool Result Inputs

When Codex sends tool results back in a follow-up request, normalize them into `user` messages containing tool-result blocks.

The normalizer should preserve:

- tool call id
- tool name when present
- structured output payload
- string output payload when the client sends string content

If the request includes both text and tool-result items, preserve encounter order in the normalized message blocks.

## Tool Choice Mapping

Map `tool_choice` conservatively:

- `auto` -> Bedrock auto tool choice
- explicit named tool -> Bedrock specific tool choice when the requested tool can be identified after synthetic-name mapping
- values that cannot be mapped safely -> `400 invalid_request_error`

The proxy should not silently reinterpret stricter tool-choice semantics.

## Response Translation

### Non-Streaming Response

When Bedrock returns an assistant message:

- text blocks become `output_text`
- tool-use blocks become OpenAI/Codex-facing tool-call output items
- mixed text plus tool-use output must preserve order

For native function tools:

- return a function-call-shaped output item that Codex can execute directly

For built-in synthetic tools:

- recover the original OpenAI built-in tool type from metadata
- return the Codex-facing output item shape associated with that tool family

### Continuation and Persistence

Persist the normalized post-response conversation snapshot after successful completion.

Snapshots must include:

- user text turns
- assistant text turns
- assistant tool-call blocks
- user tool-result blocks

Snapshots must not include:

- inherited system/developer/instructions state from prior turns

This preserves the current turn-local system precedence model while making tool loops replayable.

## Error Handling

Return `400 invalid_request_error` for:

- malformed tool definitions
- unsupported built-in tool types
- unsupported built-in tool configurations
- unsupported `tool_choice` values
- unmatched or malformed tool-result items
- `parallel_tool_calls`

Return `500` for:

- internal normalization bugs
- synthetic-tool metadata corruption
- Bedrock tool-use output that cannot be mapped back to a client-facing tool item despite passing request validation

Return `502` for:

- Bedrock request failures after the proxy accepts the request

## Testing Strategy

Implementation should follow TDD.

### `internal/openai`

- validation accepts supported tool definitions
- validation rejects malformed tool definitions
- validation rejects unsupported `parallel_tool_calls`
- validation rejects unmappable `tool_choice`

### `internal/conversation`

- normalization preserves mixed text and tool-result block ordering
- built-in tool definitions normalize into synthetic internal tool definitions
- function tools normalize directly
- continuation snapshots preserve tool calls and tool results
- merged continuation requests append new blocks without inheriting old system context

### `internal/bedrock`

- request translation maps function tools into Bedrock tool specs
- request translation maps built-in tools into synthetic Bedrock tool specs
- request translation maps prior assistant tool calls and user tool results into Bedrock content blocks
- response translation maps Bedrock tool-use blocks back into client-facing output items
- mixed text and tool-use responses preserve order

### `internal/proxy`

- non-streaming request with tools reaches Bedrock with tool config
- Bedrock tool-use response is returned in OpenAI/Codex-facing shape
- follow-up request with tool result is normalized and merged correctly
- `previous_response_id` continuation preserves tool history

## Implementation Order

1. Replace the internal text-only conversation model with a block-based normalized model.
2. Add failing normalization tests for tool definitions and tool-result inputs.
3. Add failing Bedrock request translation tests for tool-capable conversations.
4. Implement Bedrock request translation and tool-choice mapping.
5. Add failing response translation tests for Bedrock tool-use output.
6. Implement non-streaming response translation for mixed text and tool-use output.
7. Update snapshot persistence and continuation tests.
8. Add built-in synthetic tool adapters and table-driven validation.
9. Run the focused test suites, then the full Go test suite.

## Deferred Work

- streaming tool events
- full OpenAI Responses event taxonomy for tools
- proxy-side execution of built-in tools
- exact parity for every OpenAI built-in tool option
