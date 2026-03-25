# OpenAI Bedrock Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Go HTTP proxy that exposes OpenAI-compatible `POST /v1/responses` and `GET /v1/models` endpoints for Codex-first usage and forwards supported inference requests to Amazon Bedrock with the AWS default credential chain.

**Architecture:** The service starts in `cmd/openai-bedrock-proxy`, loads runtime config from flags and env, validates OpenAI Responses requests in `internal/openai`, translates them to Bedrock Converse / ConverseStream calls in `internal/bedrock`, and serves HTTP routes from `internal/httpserver`. Bedrock translation stays transport-agnostic so the later Realtime phase can reuse the same mapping logic.

**Tech Stack:** Go 1.24+, `net/http`, AWS SDK for Go v2 (`config`, `service/bedrockruntime`), `httptest`, standard-library SSE streaming

---

## File Structure

Planned files and responsibilities:

- Create: `.gitignore`
  - Ignore Go build artifacts and local env files.
- Create: `go.mod`
  - Initialize module path `github.com/moolen/openai-bedrock-proxy`.
- Create: `README.md`
  - Document local run instructions and AWS credential expectations.
- Create: `cmd/openai-bedrock-proxy/main.go`
  - Program entrypoint, config load, Bedrock client wiring, HTTP server startup, graceful shutdown.
- Create: `internal/config/config.go`
  - Runtime config model and env/flag loading helpers.
- Create: `internal/config/config_test.go`
  - Tests for config defaults and overrides.
- Create: `internal/openai/types.go`
  - Minimal Responses API request / response types required for v1.
- Create: `internal/openai/validate.go`
  - Request validation and unsupported-field checks.
- Create: `internal/openai/errors.go`
  - OpenAI-style error payload helpers.
- Create: `internal/openai/validate_test.go`
  - Validation tests for supported and unsupported request shapes.
- Create: `internal/openai/sse.go`
  - SSE event writer for streaming Responses output.
- Create: `internal/openai/sse_test.go`
  - Tests for SSE formatting and flush order.
- Create: `internal/bedrock/client.go`
  - Bedrock runtime interface and AWS SDK adapter construction.
- Create: `internal/bedrock/client_test.go`
  - Tests for adapter config behavior and method delegation.
- Create: `internal/bedrock/translate_request.go`
  - OpenAI Responses request to Bedrock Converse input mapping.
- Create: `internal/bedrock/translate_request_test.go`
  - Request translation tests.
- Create: `internal/bedrock/translate_response.go`
  - Bedrock Converse / stream event to OpenAI Responses payload mapping.
- Create: `internal/bedrock/translate_response_test.go`
  - Response translation tests.
- Create: `internal/httpserver/server.go`
  - Route registration and handlers for `/v1/responses` and `/v1/models`.
- Create: `internal/httpserver/server_test.go`
  - Handler tests with a fake Bedrock service.

### Task 1: Bootstrap Module And Runtime Config

**Files:**
- Create: `.gitignore`
- Create: `go.mod`
- Create: `README.md`
- Create: `internal/config/config.go`
- Test: `internal/config/config_test.go`

- [ ] **Step 1: Write the failing config tests**

```go
func TestLoadConfigDefaults(t *testing.T) {
	cfg := LoadFromEnv(func(string) string { return "" })
	if cfg.ListenAddr != "0.0.0.0" {
		t.Fatalf("expected default listen addr, got %q", cfg.ListenAddr)
	}
	if cfg.ListenPort != "8080" {
		t.Fatalf("expected default listen port, got %q", cfg.ListenPort)
	}
}

func TestLoadConfigOverrides(t *testing.T) {
	env := map[string]string{
		"LISTEN_ADDR": "127.0.0.1",
		"LISTEN_PORT": "9000",
		"AWS_REGION":  "us-east-1",
		"LOG_LEVEL":   "debug",
	}
	cfg := LoadFromEnv(func(key string) string { return env[key] })
	if cfg.ListenPort != "9000" {
		t.Fatalf("expected override port, got %q", cfg.ListenPort)
	}
}
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `go test ./internal/config -run TestLoadConfig -v`
Expected: FAIL because `go.mod`, package files, or `LoadFromEnv` do not exist yet

- [ ] **Step 3: Write the minimal module and config implementation**

```go
module github.com/moolen/openai-bedrock-proxy
```

```go
type Config struct {
	ListenAddr string
	ListenPort string
	AWSRegion  string
	LogLevel   string
}

func LoadFromEnv(getenv func(string) string) Config {
	return Config{
		ListenAddr: firstNonEmpty(getenv("LISTEN_ADDR"), "0.0.0.0"),
		ListenPort: firstNonEmpty(getenv("LISTEN_PORT"), "8080"),
		AWSRegion:  getenv("AWS_REGION"),
		LogLevel:   firstNonEmpty(getenv("LOG_LEVEL"), "info"),
	}
}
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `go test ./internal/config -run TestLoadConfig -v`
Expected: PASS

- [ ] **Step 5: Add repo bootstrap files and commit**

```bash
git init -b main
git add .gitignore go.mod README.md internal/config/config.go internal/config/config_test.go
git commit -m "chore: bootstrap module and runtime config"
```

### Task 2: Define Minimal OpenAI Responses Types And Validation

**Files:**
- Create: `internal/openai/types.go`
- Create: `internal/openai/validate.go`
- Create: `internal/openai/errors.go`
- Test: `internal/openai/validate_test.go`

- [ ] **Step 1: Write the failing validation tests**

```go
func TestValidateResponsesRequestAcceptsSimpleTextInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "anthropic.claude-3-7-sonnet-20250219-v1:0",
		Input: "write a haiku",
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateResponsesRequestRejectsMissingModel(t *testing.T) {
	req := ResponsesRequest{Input: "hi"}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected missing-model validation error")
	}
}

func TestValidateResponsesRequestRejectsUnsupportedFields(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		ParallelToolCalls: ptr(true),
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected unsupported-field validation error")
	}
}
```

- [ ] **Step 2: Run the validation tests to verify they fail**

Run: `go test ./internal/openai -run TestValidateResponsesRequest -v`
Expected: FAIL because request types and validator do not exist yet

- [ ] **Step 3: Implement minimal OpenAI request types and validator**

```go
type ResponsesRequest struct {
	Model             string `json:"model"`
	Input             any    `json:"input"`
	Instructions      string `json:"instructions,omitempty"`
	Stream            bool   `json:"stream,omitempty"`
	MaxOutputTokens   *int   `json:"max_output_tokens,omitempty"`
	Temperature       *float64 `json:"temperature,omitempty"`
	Tools             []Tool `json:"tools,omitempty"`
	ToolChoice        any    `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool  `json:"parallel_tool_calls,omitempty"`
}

func ValidateResponsesRequest(req ResponsesRequest) error {
	if req.Model == "" {
		return NewInvalidRequestError("model is required")
	}
	if req.Input == nil || req.Input == "" {
		return NewInvalidRequestError("input is required")
	}
	if req.ParallelToolCalls != nil {
		return NewInvalidRequestError("parallel_tool_calls is not supported")
	}
	return nil
}
```

- [ ] **Step 4: Run the validation tests to verify they pass**

Run: `go test ./internal/openai -run TestValidateResponsesRequest -v`
Expected: PASS

- [ ] **Step 5: Commit the OpenAI schema layer**

```bash
git add internal/openai/types.go internal/openai/validate.go internal/openai/errors.go internal/openai/validate_test.go
git commit -m "feat: add openai request validation"
```

### Task 3: Translate Responses Requests Into Bedrock Converse Inputs

**Files:**
- Create: `internal/bedrock/translate_request.go`
- Test: `internal/bedrock/translate_request_test.go`

- [ ] **Step 1: Write the failing request translation tests**

```go
func TestTranslateRequestPassesModelThrough(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "anthropic.claude-3-7-sonnet-20250219-v1:0",
		Input: "hello",
	}
	got, err := TranslateRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ModelID != req.Model {
		t.Fatalf("expected model passthrough, got %q", got.ModelID)
	}
}

func TestTranslateRequestBuildsSystemAndUserMessages(t *testing.T) {
	req := openai.ResponsesRequest{
		Model:        "model",
		Instructions: "be terse",
		Input:        "hello",
	}
	got, err := TranslateRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.System) != 1 || len(got.Messages) != 1 {
		t.Fatalf("expected one system and one user message")
	}
}
```

- [ ] **Step 2: Run the translation tests to verify they fail**

Run: `go test ./internal/bedrock -run TestTranslateRequest -v`
Expected: FAIL because `TranslateRequest` and Bedrock request types do not exist yet

- [ ] **Step 3: Implement the minimal request translator**

```go
type ConverseRequest struct {
	ModelID       string
	System        []string
	Messages      []Message
	MaxTokens     *int32
	Temperature   *float32
	ToolConfig    *ToolConfig
}

func TranslateRequest(req openai.ResponsesRequest) (ConverseRequest, error) {
	msgs, err := normalizeInput(req.Input)
	if err != nil {
		return ConverseRequest{}, err
	}
	out := ConverseRequest{ModelID: req.Model, Messages: msgs}
	if req.Instructions != "" {
		out.System = []string{req.Instructions}
	}
	return out, nil
}
```

- [ ] **Step 4: Run the translation tests to verify they pass**

Run: `go test ./internal/bedrock -run TestTranslateRequest -v`
Expected: PASS

- [ ] **Step 5: Commit the request translator**

```bash
git add internal/bedrock/translate_request.go internal/bedrock/translate_request_test.go
git commit -m "feat: translate responses requests to bedrock inputs"
```

### Task 4: Translate Bedrock Outputs And Streaming Events Back To OpenAI

**Files:**
- Create: `internal/bedrock/translate_response.go`
- Create: `internal/openai/sse.go`
- Test: `internal/bedrock/translate_response_test.go`
- Test: `internal/openai/sse_test.go`

- [ ] **Step 1: Write the failing response and SSE tests**

```go
func TestTranslateConverseResponseBuildsOutputText(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Text:       "hello back",
		StopReason: "end_turn",
	}
	got := TranslateResponse(resp, "model")
	if got.Object != "response" {
		t.Fatalf("expected response object, got %q", got.Object)
	}
}

func TestWriteEventFormatsSSEFrame(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteEvent(&buf, "response.output_text.delta", map[string]any{"delta": "hi"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "event: response.output_text.delta\n") {
		t.Fatalf("unexpected SSE payload: %q", buf.String())
	}
}
```

- [ ] **Step 2: Run the response translation tests to verify they fail**

Run: `go test ./internal/bedrock ./internal/openai -run 'TestTranslateConverseResponse|TestWriteEvent' -v`
Expected: FAIL because the response translator and SSE helpers do not exist yet

- [ ] **Step 3: Implement minimal response mapping and SSE helpers**

```go
type ResponsesAPIResponse struct {
	ID     string `json:"id"`
	Object string `json:"object"`
	Model  string `json:"model"`
	Output []OutputItem `json:"output"`
}

func TranslateResponse(resp ConverseResponse, model string) openai.Response {
	return openai.Response{
		ID:     "resp_" + resp.ResponseID,
		Object: "response",
		Model:  model,
		Output: []openai.OutputItem{{Type: "message", Role: "assistant", Content: []openai.ContentItem{{Type: "output_text", Text: resp.Text}}}},
	}
}

func WriteEvent(w io.Writer, name string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", name, data)
	return err
}
```

- [ ] **Step 4: Run the response translation tests to verify they pass**

Run: `go test ./internal/bedrock ./internal/openai -run 'TestTranslateConverseResponse|TestWriteEvent' -v`
Expected: PASS

- [ ] **Step 5: Commit the response translator and SSE layer**

```bash
git add internal/bedrock/translate_response.go internal/bedrock/translate_response_test.go internal/openai/sse.go internal/openai/sse_test.go
git commit -m "feat: translate bedrock outputs to openai responses"
```

### Task 5: Add The AWS Bedrock Adapter

**Files:**
- Create: `internal/bedrock/client.go`
- Test: `internal/bedrock/client_test.go`

- [ ] **Step 1: Write the failing Bedrock adapter tests**

```go
func TestNewClientUsesDefaultAWSConfig(t *testing.T) {
	called := false
	loader := func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error) {
		called = true
		return aws.Config{Region: "us-west-2"}, nil
	}
	_, err := NewClient(context.Background(), "", loader)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatal("expected AWS config loader to be called")
	}
}
```

- [ ] **Step 2: Run the Bedrock adapter tests to verify they fail**

Run: `go test ./internal/bedrock -run TestNewClientUsesDefaultAWSConfig -v`
Expected: FAIL because `NewClient` and the AWS loader seam do not exist yet

- [ ] **Step 3: Implement the minimal AWS SDK adapter**

```go
type RuntimeAPI interface {
	Converse(context.Context, *bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
	ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error)
}

func NewClient(ctx context.Context, region string, loadConfig LoadConfigFunc) (*Client, error) {
	opts := []func(*config.LoadOptions) error{}
	if region != "" {
		opts = append(opts, config.WithRegion(region))
	}
	awsCfg, err := loadConfig(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return &Client{runtime: bedrockruntime.NewFromConfig(awsCfg)}, nil
}
```

- [ ] **Step 4: Run the Bedrock adapter tests to verify they pass**

Run: `go test ./internal/bedrock -run TestNewClientUsesDefaultAWSConfig -v`
Expected: PASS

- [ ] **Step 5: Commit the AWS adapter**

```bash
git add internal/bedrock/client.go internal/bedrock/client_test.go
git commit -m "feat: add aws bedrock runtime adapter"
```

### Task 6: Implement HTTP Handlers For `/v1/responses` And `/v1/models`

**Files:**
- Create: `internal/httpserver/server.go`
- Test: `internal/httpserver/server_test.go`

- [ ] **Step 1: Write the failing handler tests**

```go
func TestResponsesHandlerIgnoresAuthorizationHeader(t *testing.T) {
	svc := fakeService{response: openai.Response{ID: "resp_1", Object: "response"}}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi"}`))
	req.Header.Set("Authorization", "Bearer ignored")
	rec := httptest.NewRecorder()

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}

func TestModelsHandlerReturnsOpenAIListShape(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)

	NewServer(fakeService{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}
```

- [ ] **Step 2: Run the handler tests to verify they fail**

Run: `go test ./internal/httpserver -run 'TestResponsesHandler|TestModelsHandler' -v`
Expected: FAIL because the HTTP server and fake service seams do not exist yet

- [ ] **Step 3: Implement the minimal HTTP server**

```go
func NewServer(svc Service) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", handleResponses(svc))
	mux.HandleFunc("GET /v1/models", handleModels())
	return mux
}

func handleResponses(svc Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req openai.ResponsesRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", err.Error())
			return
		}
		if req.Stream {
			streamResponse(w, r, svc, req)
			return
		}
		writeJSON(w, http.StatusOK, svc.Respond(r.Context(), req))
	}
}
```

- [ ] **Step 4: Run the handler tests to verify they pass**

Run: `go test ./internal/httpserver -run 'TestResponsesHandler|TestModelsHandler' -v`
Expected: PASS

- [ ] **Step 5: Commit the HTTP layer**

```bash
git add internal/httpserver/server.go internal/httpserver/server_test.go
git commit -m "feat: add openai-compatible http handlers"
```

### Task 7: Wire The Binary, Run Full Verification, And Document Usage

**Files:**
- Create: `cmd/openai-bedrock-proxy/main.go`
- Modify: `README.md`

- [ ] **Step 1: Write the failing binary wiring test or smoke check**

```go
func TestMainServerBuilds(t *testing.T) {
	cmd := exec.Command("go", "build", "./cmd/openai-bedrock-proxy")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}
}
```

- [ ] **Step 2: Run the smoke check to verify it fails**

Run: `go test ./... -run TestMainServerBuilds -v`
Expected: FAIL because `main.go` does not exist yet or the service is not wired together

- [ ] **Step 3: Implement the binary wiring and README usage**

```go
func main() {
	ctx := context.Background()
	cfg := config.LoadFromEnv(os.Getenv)
	client, err := bedrock.NewClient(ctx, cfg.AWSRegion, config.LoadDefaultConfig)
	if err != nil {
		log.Fatal(err)
	}
	srv := &http.Server{
		Addr:    net.JoinHostPort(cfg.ListenAddr, cfg.ListenPort),
		Handler: httpserver.NewServer(client),
	}
	log.Fatal(srv.ListenAndServe())
}
```

```bash
go mod tidy
go build ./cmd/openai-bedrock-proxy
go test ./...
```

- [ ] **Step 4: Run full verification**

Run: `go test ./... -v`
Expected: PASS

Run: `go build ./cmd/openai-bedrock-proxy`
Expected: PASS and produce the proxy binary

- [ ] **Step 5: Commit the integrated proxy**

```bash
git add cmd/openai-bedrock-proxy/main.go README.md go.mod go.sum
git commit -m "feat: add codex-first openai bedrock proxy"
```

## Review Notes

- This plan assumes the implementation will keep the supported OpenAI Responses subset intentionally small and explicit.
- `GET /v1/models` is planned as a minimal compatibility endpoint, not a Bedrock catalog browser.
- Realtime / WebSocket support is deferred until the HTTP Responses path is verified against Codex.
