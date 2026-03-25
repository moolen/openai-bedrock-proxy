# Debug Logging Design

**Date:** 2026-03-25

**Goal:** Add extensive, request-correlated debug logging to the proxy, with a `--log-level` CLI flag that defaults to `debug` and enables logging of raw prompts, instructions, and model responses for troubleshooting.

## Scope

Included in this change:

- A `--log-level` CLI flag on `openai-bedrock-proxy`
- Structured, leveled application logging
- Request-scoped correlation IDs propagated through HTTP, proxy, and Bedrock layers
- Verbose debug logs for request parsing, normalization, continuation lookup, Bedrock translation, streaming, persistence, and final responses
- Logging of raw prompt text, `instructions`, and raw assistant response text at `debug`

Excluded from this change:

- Log redaction or masking
- File-based log sinks
- Metrics, tracing, or OpenTelemetry export
- Request IDs surfaced back to clients in response headers
- Dynamic log-level reload without process restart

## Product Constraints

- `--log-level` must default to `debug`
- The CLI flag must override `LOG_LEVEL`
- If the user does not provide the flag, the process should still default to `debug`
- Existing Bedrock and HTTP behavior should remain unchanged apart from additional logging
- Raw prompt and response text should intentionally be logged at `debug`, even though that is sensitive

## Logging Model

The service should adopt the standard-library `log/slog` package as the only application logger.

Reasons:

- leveled logging is built in
- structured key/value output is easier to filter during debugging
- the package is already available in the standard library
- it avoids inventing a wrapper that the project would later have to replace

The logger should emit text logs to stderr through a single process-wide handler configured in `main`.

### Log Levels

Supported levels for the flag and env value:

- `debug`
- `info`
- `warn`
- `error`

Behavior:

- default level is `debug`
- invalid values should fail fast during startup with a clear error
- the resolved level should be logged once at startup

### Configuration Precedence

Startup configuration precedence should be:

1. `--log-level`
2. `LOG_LEVEL`
3. default `debug`

Other runtime configuration should continue to be read from environment variables as today.

## Correlation Strategy

Each inbound HTTP request should get a generated request ID before any significant work is done.

Requirements:

- the ID should be process-local and best-effort, not globally coordinated
- the request ID should be attached to the request context
- downstream layers should derive child loggers from the context and include the same `request_id`
- logs within a single response flow should therefore be traceable across:
  - HTTP decode and validation
  - continuation lookup and merge
  - Bedrock request construction
  - streaming deltas
  - persistence
  - final success or failure

The request ID does not need to be returned to the client in this change.

## Startup Logging

`cmd/openai-bedrock-proxy` should log:

- process startup
- resolved listen address
- resolved log level
- configured AWS region, if any
- in-memory response store capacity
- Bedrock client initialization success
- server listen start
- fatal startup or serve failures

The startup logger should not attempt to enumerate credentials or any secrets from the AWS default chain.

## HTTP Logging

`internal/httpserver` should be responsible for request lifecycle logs.

### `POST /v1/responses`

At `info`:

- request started
- request finished with status
- whether it was streaming
- model
- previous response ID, if present
- duration

At `debug`:

- decoded request body shape
- raw `input`
- raw `instructions`
- validation success or failure
- whether the handler routed to streaming or non-streaming execution
- final OpenAI response payload for non-streaming requests
- early streaming errors before bytes are written
- in-stream error event emission after bytes are written

### `GET /v1/models`

At `info`:

- request started
- request finished with status and duration

At `debug`:

- returned static payload

## Proxy Service Logging

`internal/proxy` should log stateful request handling decisions.

At `debug`:

- start of normalization
- normalized system block count
- normalized message count
- `previous_response_id` lookup attempt
- continuation lookup hit or miss
- number of messages loaded from prior state
- merged message count sent upstream
- persistence attempt and stored response ID
- persistence skipped because of upstream or streaming failure

At `info`:

- successful non-streaming completion with response ID
- successful streaming completion with response ID

For continuation requests, the logs should make it obvious that the inbound `model` remains authoritative even if the stored record used a different one.

## Bedrock Logging

`internal/bedrock` should log the Bedrock-facing view of the request and response.

At `debug` before calling Bedrock:

- model ID
- normalized system blocks
- normalized conversation messages
- `max_output_tokens`
- `temperature`

At `debug` after non-streaming responses:

- Bedrock response ID
- extracted raw assistant text
- stop reason if available

At `debug` during streaming:

- stream start
- each text delta
- accumulated assistant text on completion
- stop reason
- stream close

At `error`:

- Bedrock client construction failures
- Converse / ConverseStream API failures
- stream adapter failures
- SSE write failures

The implementation should avoid logging opaque AWS SDK internals unless they contribute direct debugging value. Request IDs and top-level error messages are sufficient.

## Raw Text Logging Policy

The user explicitly wants raw prompt and response text logged.

At `debug`, logs should include:

- raw `input`
- raw `instructions`
- normalized system blocks
- normalized conversation messages
- raw final assistant text
- raw streaming text deltas

At `info`, logs should stay compact and avoid dumping full text payloads.

This split keeps default behavior intentionally verbose because the default level is `debug`, while still preserving cleaner output if the operator later sets `--log-level=info`.

## Error Handling

Logging should make it easy to distinguish:

- malformed JSON requests
- validation failures
- unsupported streaming mode on an incompatible service
- unknown `previous_response_id`
- Bedrock API failures
- streaming failures before the SSE response starts
- streaming failures after the SSE response starts

The logger should not replace the existing HTTP error mapping. It should only describe what path was taken and why.

## Implementation Boundaries

The logging work should keep responsibilities separate:

- `cmd/openai-bedrock-proxy`
  - parse `--log-level`
  - construct the root logger and inject it
- `internal/config`
  - keep environment loading, but support the new default of `debug`
- `internal/httpserver`
  - request ID generation
  - request lifecycle logs
  - propagation of request-scoped logger through context
- `internal/proxy`
  - continuation and persistence logs
- `internal/bedrock`
  - upstream request and response logs

If a small internal logging helper package makes context propagation cleaner, that is acceptable, but it should stay minimal and standard-library-only.

## Testing Strategy

Tests should verify behavior, not just compile:

- config tests for `LOG_LEVEL` defaulting to `debug`
- config or flag parsing tests for CLI-over-env precedence
- HTTP handler tests that a request ID logger can be attached and used without changing response behavior
- proxy tests that continuation hit/miss paths still behave correctly with logging enabled
- Bedrock tests that stream and non-streaming paths still pass while using injected loggers
- targeted unit tests for log-level parsing failures

The tests do not need to snapshot every emitted log line, but they should prove:

- the new flag is accepted
- invalid log levels fail early
- logger injection does not change request semantics
- request-scoped context propagation works across the core path

## Risks

- defaulting to `debug` will make the service noisy by default
- raw prompt/response logging will capture sensitive user data and secrets if they appear in prompts or outputs
- logging every stream delta may be high volume for long responses
- adding logger plumbing across layers increases constructor and test setup surface area

These are accepted tradeoffs for the stated debugging goal.

## Acceptance Criteria

- starting the binary without flags logs at `debug`
- `--log-level=info` reduces output to compact informational logs
- `--log-level` overrides `LOG_LEVEL`
- invalid log-level values fail during startup
- `/v1/responses` logs request start, validation path, continuation decisions, Bedrock call details, and completion/failure with a shared `request_id`
- debug logs include raw `input`, `instructions`, and raw assistant output text
- streaming requests log deltas and terminal state
- existing tests still pass and new tests cover the logging configuration path
