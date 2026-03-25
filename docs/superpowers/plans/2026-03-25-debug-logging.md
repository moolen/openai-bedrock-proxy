# Debug Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add extensive structured debug logging with request correlation across the HTTP, proxy, and Bedrock layers, and add a `--log-level` CLI flag that defaults to `debug`.

**Architecture:** Introduce a small internal logging package that owns log-level parsing plus request-scoped logger propagation, wire a root `slog` logger from `cmd/openai-bedrock-proxy`, and inject it into `internal/httpserver`, `internal/proxy`, and `internal/bedrock`. Keep request semantics unchanged while making the full request lifecycle observable and logging raw prompts, instructions, and responses at `debug`.

**Tech Stack:** Go 1.24+, standard library `log/slog`, `flag`, `context`, `httptest`, AWS SDK for Go v2 Bedrock Runtime

---

## File Structure

Planned files and responsibilities:

- Modify: `internal/config/config.go`
  - Change the default env-derived log level from `info` to `debug`.
- Modify: `internal/config/config_test.go`
  - Lock the new default and preserve env override coverage.
- Create: `internal/logging/logging.go`
  - Parse log levels, build a root logger, and expose context logger helpers.
- Create: `internal/logging/logging_test.go`
  - Verify valid/invalid level parsing and context fallback behavior.
- Modify: `cmd/openai-bedrock-proxy/main.go`
  - Parse `--log-level`, resolve CLI-over-env precedence, build the root logger, log startup, and inject the logger into downstream services.
- Modify: `main_test.go`
  - Keep the binary build check passing after the new flag/logging wiring.
- Modify: `internal/httpserver/server.go`
  - Generate request IDs, attach request-scoped loggers to context, and log request lifecycle plus payload details.
- Modify: `internal/httpserver/server_test.go`
  - Cover logging-aware server construction and ensure request semantics stay unchanged.
- Modify: `internal/proxy/service.go`
  - Add continuation, normalization, merge, persistence, and completion logs.
- Modify: `internal/proxy/service_test.go`
  - Keep behavior coverage passing with injected loggers and add targeted request-context logging seams if needed.
- Modify: `internal/bedrock/client.go`
  - Add debug/error logging for client initialization, request translation, stream handling, deltas, and final text.
- Modify: `internal/bedrock/client_test.go`
  - Verify the Bedrock client still behaves correctly with logging enabled and cover any new helper seams.

### Task 1: Add Logging Primitives And Debug-Default Config

**Files:**
- Modify: `internal/config/config.go`
- Modify: `internal/config/config_test.go`
- Create: `internal/logging/logging.go`
- Create: `internal/logging/logging_test.go`

- [ ] **Step 1: Write the failing config and logging tests**

```go
func TestLoadConfigDefaults(t *testing.T) {
	cfg := LoadFromEnv(func(string) string { return "" })
	if cfg.LogLevel != "debug" {
		t.Fatalf("expected default log level, got %q", cfg.LogLevel)
	}
}

func TestParseLevelAcceptsKnownLevels(t *testing.T) {
	cases := []string{"debug", "info", "warn", "error"}
	for _, tc := range cases {
		if _, err := ParseLevel(tc); err != nil {
			t.Fatalf("expected %q to parse, got %v", tc, err)
		}
	}
}

func TestParseLevelRejectsUnknownValue(t *testing.T) {
	if _, err := ParseLevel("trace"); err == nil {
		t.Fatal("expected invalid level error")
	}
}

func TestFromContextFallsBackToDefaultLogger(t *testing.T) {
	logger := FromContext(context.Background())
	if logger == nil {
		t.Fatal("expected fallback logger")
	}
}
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `go test ./internal/config ./internal/logging -run 'Test(LoadConfigDefaults|ParseLevelAcceptsKnownLevels|ParseLevelRejectsUnknownValue|FromContextFallsBackToDefaultLogger)' -v`
Expected: FAIL because the config still defaults to `info` and the logging package does not exist yet

- [ ] **Step 3: Implement the minimal logging primitives and config default**

```go
func LoadFromEnv(getenv func(string) string) Config {
	return Config{
		ListenAddr: firstNonEmpty(getenv("LISTEN_ADDR"), "0.0.0.0"),
		ListenPort: firstNonEmpty(getenv("LISTEN_PORT"), "8080"),
		AWSRegion:  getenv("AWS_REGION"),
		LogLevel:   firstNonEmpty(getenv("LOG_LEVEL"), "debug"),
	}
}
```

```go
func ParseLevel(value string) (slog.Level, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "debug":
		return slog.LevelDebug, nil
	case "info":
		return slog.LevelInfo, nil
	case "warn":
		return slog.LevelWarn, nil
	case "error":
		return slog.LevelError, nil
	default:
		return 0, fmt.Errorf("invalid log level %q", value)
	}
}
```

```go
func NewLogger(level slog.Level) *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
}
```

```go
func WithLogger(ctx context.Context, logger *slog.Logger) context.Context
func FromContext(ctx context.Context) *slog.Logger
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `go test ./internal/config ./internal/logging -run 'Test(LoadConfigDefaults|ParseLevelAcceptsKnownLevels|ParseLevelRejectsUnknownValue|FromContextFallsBackToDefaultLogger)' -v`
Expected: PASS

- [ ] **Step 5: Commit the logging primitives**

```bash
git add internal/config/config.go internal/config/config_test.go internal/logging/logging.go internal/logging/logging_test.go
git commit -m "feat: add logging primitives"
```

### Task 2: Wire CLI Log-Level Resolution And Startup Logging

**Files:**
- Modify: `cmd/openai-bedrock-proxy/main.go`
- Modify: `main_test.go`

- [ ] **Step 1: Write the failing startup and flag-resolution tests**

Add a small testable helper in `main.go` and test it first.

```go
func TestResolveLogLevelUsesCLIValueFirst(t *testing.T) {
	cfg := config.Config{LogLevel: "error"}
	level, err := resolveLogLevel("debug", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if level != slog.LevelDebug {
		t.Fatalf("expected debug level, got %v", level)
	}
}

func TestResolveLogLevelFallsBackToConfigValue(t *testing.T) {
	cfg := config.Config{LogLevel: "warn"}
	level, err := resolveLogLevel("", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if level != slog.LevelWarn {
		t.Fatalf("expected warn level, got %v", level)
	}
}

func TestResolveLogLevelRejectsInvalidValue(t *testing.T) {
	cfg := config.Config{LogLevel: "debug"}
	if _, err := resolveLogLevel("loud", cfg); err == nil {
		t.Fatal("expected invalid level error")
	}
}
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `go test ./... -run 'Test(ResolveLogLevelUsesCLIValueFirst|ResolveLogLevelFallsBackToConfigValue|ResolveLogLevelRejectsInvalidValue)' -v`
Expected: FAIL because `resolveLogLevel` and CLI flag parsing do not exist yet

- [ ] **Step 3: Implement flag parsing, root logger construction, and startup logs**

```go
func resolveLogLevel(flagValue string, cfg config.Config) (slog.Level, error) {
	value := cfg.LogLevel
	if flagValue != "" {
		value = flagValue
	}
	return applog.ParseLevel(value)
}
```

```go
var logLevelFlag string
flag.StringVar(&logLevelFlag, "log-level", "", "log level: debug|info|warn|error")
flag.Parse()
```

```go
logger := applog.NewLogger(level)
logger.Info("starting proxy", "listen_addr", addr, "aws_region", cfg.AWSRegion, "log_level", level.String())
```

Use that logger when constructing the Bedrock client, proxy service, and HTTP server.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `go test ./... -run 'Test(ResolveLogLevelUsesCLIValueFirst|ResolveLogLevelFallsBackToConfigValue|ResolveLogLevelRejectsInvalidValue|TestMainServerBuilds)' -v`
Expected: PASS

- [ ] **Step 5: Commit the startup logging wiring**

```bash
git add cmd/openai-bedrock-proxy/main.go main_test.go
git commit -m "feat: add startup log level flag"
```

### Task 3: Add Request-Scoped HTTP Logging And Correlation IDs

**Files:**
- Modify: `internal/httpserver/server.go`
- Modify: `internal/httpserver/server_test.go`
- Modify: `internal/logging/logging.go`
- Modify: `internal/logging/logging_test.go`

- [ ] **Step 1: Write the failing HTTP logging tests**

Focus on request context propagation and keeping behavior unchanged.

```go
func TestResponsesHandlerUsesInjectedLoggerAndKeepsSuccessShape(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	svc := &fakeService{
		response: openai.Response{ID: "resp_1", Object: "response", Model: "model"},
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi"}`))
	rec := httptest.NewRecorder()

	NewServer(svc, logger).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if !strings.Contains(buf.String(), "request_id=") {
		t.Fatalf("expected request_id in logs, got %q", buf.String())
	}
}

func TestResponsesHandlerLogsStreamingPathWithoutBreakingSSE(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi","stream":true}`))

	NewServer(&fakeService{streamBody: "event: response.completed\ndata: {\"status\":\"ok\"}\n\n"}, logger).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if !strings.Contains(buf.String(), "stream=true") {
		t.Fatalf("expected streaming log entry, got %q", buf.String())
	}
}
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `go test ./internal/httpserver ./internal/logging -run 'TestResponsesHandler(UsesInjectedLoggerAndKeepsSuccessShape|LogsStreamingPathWithoutBreakingSSE)' -v`
Expected: FAIL because `NewServer` does not accept a logger or create request-scoped context yet

- [ ] **Step 3: Implement request IDs, context logger propagation, and HTTP lifecycle logs**

Add minimal helpers:

```go
func WithRequestID(ctx context.Context, requestID string) context.Context
func RequestIDFromContext(ctx context.Context) string
func WithRequestLogger(ctx context.Context, base *slog.Logger, requestID string, attrs ...any) context.Context
```

Update the server shape:

```go
func NewServer(svc Service, logger *slog.Logger) http.Handler
```

Inside handlers:

- generate a request ID before decode
- derive a request logger with `request_id`, `method`, and `path`
- attach it to the request context
- log request start/end at `info`
- log raw `input`, raw `instructions`, streaming routing, validation failures, and response payloads at `debug`

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `go test ./internal/httpserver ./internal/logging -run 'TestResponsesHandler(UsesInjectedLoggerAndKeepsSuccessShape|LogsStreamingPathWithoutBreakingSSE)' -v`
Expected: PASS

- [ ] **Step 5: Commit the HTTP logging layer**

```bash
git add internal/httpserver/server.go internal/httpserver/server_test.go internal/logging/logging.go internal/logging/logging_test.go
git commit -m "feat: add request scoped http logging"
```

### Task 4: Add Proxy-Service Debug Logs For Continuation And Persistence

**Files:**
- Modify: `internal/proxy/service.go`
- Modify: `internal/proxy/service_test.go`

- [ ] **Step 1: Write the failing proxy logging tests**

Keep the tests behavior-oriented and only assert on a few high-value log markers.

```go
func TestServiceRespondLogsContinuationLookupAndCompletion(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))

	store := newRecordingStore()
	store.Save(conversation.Record{
		ResponseID: "resp_prev",
		Messages: []conversation.Message{
			{Role: "user", Text: "hi"},
			{Role: "assistant", Text: "hello"},
		},
	})
	client := &fakeBedrock{respondResp: bedrock.ConverseResponse{ResponseID: "1", Text: "ok"}}
	svc := NewService(client, store, logger)

	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model: "model",
		Input: "next",
		PreviousResponseID: "resp_prev",
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !strings.Contains(buf.String(), "previous_response_id=resp_prev") {
		t.Fatalf("expected continuation log, got %q", buf.String())
	}
	if !strings.Contains(buf.String(), "response_id=resp_1") {
		t.Fatalf("expected completion log, got %q", buf.String())
	}
}

func TestServiceStreamLogsSkippedPersistenceOnError(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))

	svc := NewService(&fakeBedrock{streamErr: errors.New("boom")}, newRecordingStore(), logger)
	err := svc.Stream(context.Background(), openai.ResponsesRequest{Model: "model", Input: "hi"}, httptest.NewRecorder())
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(buf.String(), "persistence_skipped") {
		t.Fatalf("expected skip log, got %q", buf.String())
	}
}
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `go test ./internal/proxy -run 'TestService(RespondLogsContinuationLookupAndCompletion|StreamLogsSkippedPersistenceOnError)' -v`
Expected: FAIL because the proxy service does not accept a logger or emit those log entries yet

- [ ] **Step 3: Implement proxy logging**

Update the constructor:

```go
func NewService(client BedrockConversation, store conversation.Store, logger *slog.Logger) *Service
```

In `Respond` and `Stream`, log:

- normalization start and counts
- previous-response lookup attempt and outcome
- incoming model versus stored model when continuing
- merged message counts
- response ID on success
- persistence save or persistence skip

Use `applog.FromContext(ctx)` inside request methods so request IDs flow through from the HTTP layer.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `go test ./internal/proxy -run 'TestService(RespondLogsContinuationLookupAndCompletion|StreamLogsSkippedPersistenceOnError)' -v`
Expected: PASS

- [ ] **Step 5: Commit the proxy logging changes**

```bash
git add internal/proxy/service.go internal/proxy/service_test.go
git commit -m "feat: log proxy continuation lifecycle"
```

### Task 5: Add Bedrock Request/Response/Streaming Logs

**Files:**
- Modify: `internal/bedrock/client.go`
- Modify: `internal/bedrock/client_test.go`

- [ ] **Step 1: Write the failing Bedrock logging tests**

```go
func TestClientRespondConversationLogsModelAndResponseText(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberText{Value: "normalized reply"},
					},
				},
			},
		},
	}
	client := &Client{runtime: runtime, logger: logger}

	_, err := client.RespondConversation(context.Background(), "model-id", conversation.Request{
		System: []string{"be precise"},
		Messages: []conversation.Message{{Role: "user", Text: "hello"}},
	}, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "model_id=model-id") {
		t.Fatalf("expected model log, got %q", buf.String())
	}
	if !strings.Contains(buf.String(), "normalized reply") {
		t.Fatalf("expected response text in logs, got %q", buf.String())
	}
}

func TestClientStreamConversationLogsDeltas(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	// build fake stream with "hello"
	// expect log output to contain the delta and end_turn
}
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `go test ./internal/bedrock -run 'TestClient(RespondConversationLogsModelAndResponseText|StreamConversationLogsDeltas)' -v`
Expected: FAIL because the Bedrock client does not carry a logger or emit request/stream logs yet

- [ ] **Step 3: Implement Bedrock logging**

Extend the client:

```go
type Client struct {
	runtime       RuntimeAPI
	streamAdapter streamAdapterFunc
	logger        *slog.Logger
}
```

Use the logger for:

- client initialization success/failure
- outbound translated request metadata and raw normalized messages
- non-streaming response IDs and raw text
- streaming start, per-delta logs, stop reason, final accumulated text, and upstream errors

Keep the log-writing outside the core translation logic where possible so translation tests stay simple.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `go test ./internal/bedrock -run 'TestClient(RespondConversationLogsModelAndResponseText|StreamConversationLogsDeltas)' -v`
Expected: PASS

- [ ] **Step 5: Commit the Bedrock logging changes**

```bash
git add internal/bedrock/client.go internal/bedrock/client_test.go
git commit -m "feat: add bedrock debug logging"
```

### Task 6: Run Full Verification And Stabilize Wiring

**Files:**
- Modify as needed based on verification failures:
  - `cmd/openai-bedrock-proxy/main.go`
  - `internal/httpserver/server.go`
  - `internal/proxy/service.go`
  - `internal/bedrock/client.go`
  - related test files

- [ ] **Step 1: Run package-level verification after the wiring is complete**

Run: `go test ./internal/... -v`
Expected: PASS

- [ ] **Step 2: Run full repository verification**

Run: `go test ./... -v`
Expected: PASS

- [ ] **Step 3: Run a fresh build**

Run: `go build ./cmd/openai-bedrock-proxy`
Expected: PASS

- [ ] **Step 4: Manually verify CLI behavior**

Run: `go run ./cmd/openai-bedrock-proxy --log-level=info`
Expected: startup succeeds and logs at info level

Run: `go run ./cmd/openai-bedrock-proxy --log-level=invalid`
Expected: startup exits early with a clear invalid-level error

- [ ] **Step 5: Commit verification fixes if needed**

```bash
git add cmd/openai-bedrock-proxy/main.go
git add internal/config/config.go internal/config/config_test.go
git add internal/httpserver/server.go internal/proxy/service.go internal/bedrock/client.go
git add internal/logging/logging.go
git add internal/config/config_test.go internal/httpserver/server_test.go internal/proxy/service_test.go internal/bedrock/client_test.go
git commit -m "test: finalize debug logging wiring"
```
