# Responses Slice 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Responses API slice-1 compatibility by supporting structured text-only `input` items and `previous_response_id` continuation with bounded in-memory state.

**Architecture:** Extend the OpenAI request schema and validator to accept the approved item-based subset, introduce a normalized conversation model plus an in-memory response store, and route requests through a stateful proxy service that merges prior conversation state before calling Bedrock. Keep Bedrock translation transport-agnostic, and make streaming accumulate final assistant text so successful streamed turns can also be persisted.

**Tech Stack:** Go 1.24+, standard library, `net/http`, `httptest`, AWS SDK for Go v2 Bedrock Runtime

---

## File Structure

Planned files and responsibilities:

- Modify: `internal/openai/types.go`
  - Add `previous_response_id` and structured message/content request types for the accepted slice-1 subset.
- Modify: `internal/openai/validate.go`
  - Broaden validation from plain strings to accepted message forms and keep explicit rejections for tools and unsupported item/content types.
- Modify: `internal/openai/validate_test.go`
  - Cover accepted structured input shapes, `developer`/`system` roles, and rejected unsupported content.
- Create: `internal/conversation/types.go`
  - Define the normalized conversation model independent from raw OpenAI JSON and Bedrock SDK types.
- Create: `internal/conversation/normalize.go`
  - Convert `openai.ResponsesRequest` input into normalized system blocks and persisted conversation turns, and provide merge/append helpers for continuation state.
- Create: `internal/conversation/normalize_test.go`
  - Lock the normalization rules for easy-message input, explicit `type:"message"` input, `instructions` precedence, and continuation appends.
- Create: `internal/conversation/store.go`
  - Define the `Store` interface, response-record constructors, and provide a bounded, concurrency-safe FIFO in-memory response store keyed by OpenAI-style response ID.
- Create: `internal/conversation/store_test.go`
  - Verify lookup, save, unknown ID behavior, and eviction behavior.
- Modify: `internal/bedrock/translate_request.go`
  - Translate normalized conversation state into Bedrock `ConverseRequest` values.
- Modify: `internal/bedrock/translate_request_test.go`
  - Cover slice-1 translation from normalized conversation to Bedrock system/messages.
- Modify: `internal/bedrock/client.go`
  - Split “call Bedrock” from “parse OpenAI request” so a higher-level proxy service can manage `previous_response_id` state.
- Modify: `internal/bedrock/translate_response.go`
  - Add a helper to build/persist assistant text consistently for both non-streaming and streaming paths.
- Modify: `internal/bedrock/translate_response_test.go`
  - Lock the response-to-assistant-text behavior used for persistence.
- Create: `internal/proxy/service.go`
  - Implement the stateful `Respond` and `Stream` flow, including `loadPrevious` lookup semantics and incoming-model-wins continuation behavior.
- Create: `internal/proxy/service_test.go`
  - Verify `previous_response_id` lookup, no-persist-on-failure, model passthrough, and stream persistence.
- Modify: `internal/httpserver/server_test.go`
  - Add handler coverage for `previous_response_id` failures and accepted structured input payloads.
- Modify: `cmd/openai-bedrock-proxy/main.go`
  - Wire the Bedrock client through the new proxy service with an in-memory store.

### Task 1: Expand the OpenAI request schema and validator

**Files:**
- Modify: `internal/openai/types.go`
- Modify: `internal/openai/validate.go`
- Test: `internal/openai/validate_test.go`

- [ ] **Step 1: Write the failing validation tests**

```go
func TestValidateResponsesRequestAcceptsEasyMessageInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role":    "user",
			"content": "hello",
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsExplicitMessageInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: []any{
			map[string]any{
				"type": "message",
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "be terse"},
				},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsPlainStringRegression(t *testing.T) {
	req := ResponsesRequest{Model: "model", Input: "hello"}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsArrayOfMessages(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: []any{
			map[string]any{"role": "system", "content": "be terse"},
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "output_text", "text": "done"},
				},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsAssistantEasyMessage(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: []any{
			map[string]any{"role": "assistant", "content": "done"},
			map[string]any{"role": "user", "content": "continue"},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateResponsesRequestRejectsUnsupportedContentBlock(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"type": "message",
			"role": "user",
			"content": []any{
				map[string]any{"type": "input_image", "image_url": "ignored"},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected unsupported content block error")
	}
}

func TestValidateResponsesRequestRejectsNonMessageItem(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: []any{
			map[string]any{"type": "function_call", "name": "lookup"},
		},
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected unsupported item error")
	}
}

func TestValidateResponsesRequestRejectsAssistantInputTextBlocks(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"type": "message",
			"role": "assistant",
			"content": []any{
				map[string]any{"type": "input_text", "text": "wrong block"},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected role/block mismatch error")
	}
}

func TestValidateResponsesRequestRejectsUserOutputTextBlocks(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"type": "message",
			"role": "user",
			"content": []any{
				map[string]any{"type": "output_text", "text": "wrong block"},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected role/block mismatch error")
	}
}
```

- [ ] **Step 2: Run the validator tests to verify they fail**

Run: `go test ./internal/openai -run 'TestValidateResponsesRequest(AcceptsEasyMessageInput|AcceptsExplicitMessageInput|AcceptsPlainStringRegression|AcceptsArrayOfMessages|AcceptsAssistantEasyMessage|RejectsUnsupportedContentBlock|RejectsNonMessageItem|RejectsAssistantInputTextBlocks|RejectsUserOutputTextBlocks)' -v`
Expected: FAIL because the request types and validator only support plain string input today

- [ ] **Step 3: Implement the slice-1 request schema and validator**

```go
type ResponsesRequest struct {
	Model              string         `json:"model"`
	Input              any            `json:"input"`
	Instructions       string         `json:"instructions,omitempty"`
	PreviousResponseID string         `json:"previous_response_id,omitempty"`
	Stream             bool           `json:"stream,omitempty"`
	MaxOutputTokens    *int           `json:"max_output_tokens,omitempty"`
	Temperature        *float64       `json:"temperature,omitempty"`
	Tools              []Tool         `json:"tools,omitempty"`
	ToolChoice         any            `json:"tool_choice,omitempty"`
	ParallelToolCalls  *bool          `json:"parallel_tool_calls,omitempty"`
}

func ValidateResponsesRequest(req ResponsesRequest) error {
	if req.Model == "" {
		return NewInvalidRequestError("model is required")
	}
	if isEmptyInput(req.Input) {
		return NewInvalidRequestError("input is required")
	}
	if err := validateInput(req.Input); err != nil {
		return err
	}
	if len(req.Tools) > 0 {
		return NewInvalidRequestError("tools are not supported")
	}
	if req.ToolChoice != nil {
		return NewInvalidRequestError("tool_choice is not supported")
	}
	if req.ParallelToolCalls != nil {
		return NewInvalidRequestError("parallel_tool_calls is not supported")
	}
	return nil
}
```

- [ ] **Step 4: Run the validator tests to verify they pass**

Run: `go test ./internal/openai -run 'TestValidateResponsesRequest(AcceptsEasyMessageInput|AcceptsExplicitMessageInput|AcceptsPlainStringRegression|AcceptsArrayOfMessages|AcceptsAssistantEasyMessage|RejectsUnsupportedContentBlock|RejectsNonMessageItem|RejectsAssistantInputTextBlocks|RejectsUserOutputTextBlocks)' -v`
Expected: PASS

- [ ] **Step 5: Commit the request-schema changes**

```bash
git add internal/openai/types.go internal/openai/validate.go internal/openai/validate_test.go
git commit -m "feat: accept slice-1 responses request shapes"
```

### Task 2: Add normalized conversation parsing and bounded response storage

**Files:**
- Create: `internal/conversation/types.go`
- Create: `internal/conversation/normalize.go`
- Create: `internal/conversation/normalize_test.go`
- Create: `internal/conversation/store.go`
- Create: `internal/conversation/store_test.go`

- [ ] **Step 1: Write the failing normalization and store tests**

```go
func TestNormalizeRequestBuildsSystemAndMessages(t *testing.T) {
	req := openai.ResponsesRequest{
		Instructions: "final instruction",
		Input: []any{
			map[string]any{"role": "system", "content": "first system"},
			map[string]any{"role": "developer", "content": "first rule"},
			map[string]any{"role": "user", "content": "hello"},
		},
	}

	got, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := cmp.Diff([]string{"first system", "first rule", "final instruction"}, got.System); diff != "" {
		t.Fatalf("unexpected system blocks (-want +got):\n%s", diff)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" {
		t.Fatalf("expected one persisted user message, got %#v", got.Messages)
	}
}

func TestNormalizeRequestPersistsAssistantTurnsButNotSystemTurns(t *testing.T) {
	req := openai.ResponsesRequest{
		Input: []any{
			map[string]any{"role": "assistant", "content": "done"},
			map[string]any{"role": "user", "content": "continue"},
			map[string]any{"role": "system", "content": "do not persist"},
		},
	}

	got, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := cmp.Diff([]Message{
		{Role: "assistant", Text: "done"},
		{Role: "user", Text: "continue"},
	}, got.Messages); diff != "" {
		t.Fatalf("unexpected persisted messages (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"do not persist"}, got.System); diff != "" {
		t.Fatalf("unexpected system blocks (-want +got):\n%s", diff)
	}
}

func TestNormalizeRequestConcatenatesTextBlocksWithoutSeparators(t *testing.T) {
	req := openai.ResponsesRequest{
		Input: map[string]any{
			"type": "message",
			"role": "assistant",
			"content": []any{
				map[string]any{"type": "output_text", "text": "hel"},
				map[string]any{"type": "output_text", "text": "lo"},
			},
		},
	}

	got, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Messages[0].Text != "hello" {
		t.Fatalf("expected concatenated text, got %q", got.Messages[0].Text)
	}
}

func TestInMemoryStoreEvictsOldestRecord(t *testing.T) {
	store := NewInMemoryStore(2)
	store.Save(Record{ResponseID: "resp_1"})
	store.Save(Record{ResponseID: "resp_2"})
	store.Save(Record{ResponseID: "resp_3"})

	if _, ok := store.Get("resp_1"); ok {
		t.Fatal("expected oldest record to be evicted")
	}
}
```

- [ ] **Step 2: Run those tests to verify they fail**

Run: `go test ./internal/conversation -run 'Test(NormalizeRequestBuildsSystemAndMessages|NormalizeRequestPersistsAssistantTurnsButNotSystemTurns|NormalizeRequestConcatenatesTextBlocksWithoutSeparators|InMemoryStoreEvictsOldestRecord)' -v`
Expected: FAIL because the package and helpers do not exist yet

- [ ] **Step 3: Implement normalization and bounded store**

```go
type Message struct {
	Role string
	Text string
}

type Request struct {
	System   []string
	Messages []Message
}

type Record struct {
	ResponseID string
	ModelID    string
	Messages   []Message
	CreatedAt  time.Time
}

type InMemoryStore struct {
	limit int
	order []string
	data  map[string]Record
	mu    sync.Mutex
}
```

- [ ] **Step 4: Run the normalization and store tests to verify they pass**

Run: `go test ./internal/conversation -run 'Test(NormalizeRequestBuildsSystemAndMessages|NormalizeRequestPersistsAssistantTurnsButNotSystemTurns|NormalizeRequestConcatenatesTextBlocksWithoutSeparators|InMemoryStoreEvictsOldestRecord)' -v`
Expected: PASS

- [ ] **Step 5: Commit the normalization and store layer**

```bash
git add internal/conversation/types.go internal/conversation/normalize.go internal/conversation/normalize_test.go internal/conversation/store.go internal/conversation/store_test.go
git commit -m "feat: add normalized conversation state"
```

### Task 3: Refactor Bedrock translation to consume normalized conversations

**Files:**
- Modify: `internal/bedrock/translate_request.go`
- Modify: `internal/bedrock/translate_request_test.go`
- Modify: `internal/bedrock/client.go`
- Modify: `internal/bedrock/translate_response.go`
- Modify: `internal/bedrock/translate_response_test.go`

- [ ] **Step 1: Write the failing Bedrock translation tests**

```go
func TestTranslateConversationBuildsBedrockMessages(t *testing.T) {
	req := conversation.Request{
		System: []string{"be terse"},
		Messages: []conversation.Message{
			{Role: "user", Text: "hello"},
			{Role: "assistant", Text: "hi"},
		},
	}

	got, err := TranslateConversation("model", req, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ModelID != "model" || len(got.System) != 1 || len(got.Messages) != 2 {
		t.Fatalf("unexpected translated request: %#v", got)
	}
}

func TestAccumulateStreamTextJoinsAllDeltas(t *testing.T) {
	acc := NewTextAccumulator()
	acc.Add("hel")
	acc.Add("lo")
	if got := acc.Text(); got != "hello" {
		t.Fatalf("expected joined text, got %q", got)
	}
}
```

- [ ] **Step 2: Run those tests to verify they fail**

Run: `go test ./internal/bedrock -run 'Test(TranslateConversationBuildsBedrockMessages|AccumulateStreamTextJoinsAllDeltas)' -v`
Expected: FAIL because translation still accepts raw OpenAI requests and streaming text accumulation does not exist

- [ ] **Step 3: Implement normalized translation and response accumulation**

```go
func TranslateConversation(modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64) (ConverseRequest, error) {
	return ConverseRequest{
		ModelID:  modelID,
		System:   append([]string(nil), req.System...),
		Messages: toBedrockMessages(req.Messages),
	}, nil
}

type TextAccumulator struct {
	builder strings.Builder
}

func (a *TextAccumulator) Add(delta string) {
	a.builder.WriteString(delta)
}
```

- [ ] **Step 4: Run the Bedrock translation tests to verify they pass**

Run: `go test ./internal/bedrock -run 'Test(TranslateConversationBuildsBedrockMessages|AccumulateStreamTextJoinsAllDeltas)' -v`
Expected: PASS

- [ ] **Step 5: Commit the Bedrock translation refactor**

```bash
git add internal/bedrock/translate_request.go internal/bedrock/translate_request_test.go internal/bedrock/client.go internal/bedrock/translate_response.go internal/bedrock/translate_response_test.go
git commit -m "refactor: translate normalized conversations for bedrock"
```

### Task 4: Add the stateful proxy service and wire it into HTTP startup

**Files:**
- Create: `internal/proxy/service.go`
- Create: `internal/proxy/service_test.go`
- Modify: `internal/httpserver/server_test.go`
- Modify: `cmd/openai-bedrock-proxy/main.go`

- [ ] **Step 1: Write the failing proxy-service and handler tests**

```go
func TestServiceRespondUsesPreviousResponseSnapshot(t *testing.T) {
	store := conversation.NewInMemoryStore(4)
	store.Save(conversation.Record{
		ResponseID: "resp_prev",
		ModelID:    "model",
		Messages:   []conversation.Message{{Role: "user", Text: "hello"}},
	})

	runtime := &fakeRuntime{responseText: "patched"}
	svc := NewService(runtime, store)

	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model:              "model",
		PreviousResponseID: "resp_prev",
		Input:              "apply the patch",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := runtime.last.Messages; len(got) != 2 {
		t.Fatalf("expected prior+new messages, got %#v", got)
	}
}

func TestServiceRespondRebuildsTurnLocalSystemContext(t *testing.T) {
	store := conversation.NewInMemoryStore(4)
	store.Save(conversation.Record{
		ResponseID: "resp_prev",
		ModelID:    "model",
		Messages:   []conversation.Message{{Role: "assistant", Text: "done"}},
	})

	runtime := &fakeRuntime{responseText: "next"}
	svc := NewService(runtime, store)

	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model:              "model",
		PreviousResponseID: "resp_prev",
		Instructions:       "final instruction",
		Input: []any{
			map[string]any{"role": "developer", "content": "first rule"},
			map[string]any{"role": "user", "content": "continue"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := cmp.Diff([]string{"first rule", "final instruction"}, runtime.last.System); diff != "" {
		t.Fatalf("unexpected system blocks (-want +got):\n%s", diff)
	}
}

func TestServiceRespondPersistsAssistantReplyInSnapshot(t *testing.T) {
	store := conversation.NewInMemoryStore(4)
	runtime := &fakeRuntime{responseText: "assistant reply"}
	svc := NewService(runtime, store)

	resp, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model: "model",
		Input: "hello",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	record, ok := store.Get(resp.ID)
	if !ok {
		t.Fatal("expected response snapshot to be stored")
	}
	if diff := cmp.Diff([]conversation.Message{
		{Role: "user", Text: "hello"},
		{Role: "assistant", Text: "assistant reply"},
	}, record.Messages); diff != "" {
		t.Fatalf("unexpected stored messages (-want +got):\n%s", diff)
	}
}

func TestServiceRespondUsesIncomingModelForContinuation(t *testing.T) {
	store := conversation.NewInMemoryStore(4)
	store.Save(conversation.Record{
		ResponseID: "resp_prev",
		ModelID:    "old-model",
		Messages:   []conversation.Message{{Role: "user", Text: "hello"}},
	})

	runtime := &fakeRuntime{responseText: "assistant reply"}
	svc := NewService(runtime, store)

	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model:              "new-model",
		PreviousResponseID: "resp_prev",
		Input:              "continue",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if runtime.last.ModelID != "new-model" {
		t.Fatalf("expected incoming model to be authoritative, got %q", runtime.last.ModelID)
	}
}

func TestServiceStreamPersistsOnlyAfterCleanCompletion(t *testing.T) {
	store := conversation.NewInMemoryStore(4)
	runtime := &fakeRuntime{
		streamEvents: []string{"hel", "lo"},
		streamStatus: "completed",
	}
	svc := NewService(runtime, store)
	w := httptest.NewRecorder()

	err := svc.Stream(context.Background(), openai.ResponsesRequest{
		Model: "model",
		Input: "hello",
		Stream: true,
	}, w)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	record, ok := store.Get("resp_stream_1")
	if !ok {
		t.Fatal("expected successful stream to be persisted")
	}
	if diff := cmp.Diff([]conversation.Message{
		{Role: "user", Text: "hello"},
		{Role: "assistant", Text: "hello"},
	}, record.Messages); diff != "" {
		t.Fatalf("unexpected stored stream snapshot (-want +got):\n%s", diff)
	}
}

func TestServiceStreamDoesNotPersistFailedStream(t *testing.T) {
	store := conversation.NewInMemoryStore(4)
	runtime := &fakeRuntime{
		streamEvents: []string{"partial"},
		streamErr:    errors.New("stream failed"),
	}
	svc := NewService(runtime, store)

	err := svc.Stream(context.Background(), openai.ResponsesRequest{
		Model: "model",
		Input: "hello",
		Stream: true,
	}, httptest.NewRecorder())
	if err == nil {
		t.Fatal("expected stream error")
	}
	if runtime.savedResponseID != "" {
		t.Fatalf("expected no persisted response id, got %q", runtime.savedResponseID)
	}
}

func TestResponsesHandlerReturnsBadRequestForUnknownPreviousResponseID(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","previous_response_id":"resp_missing","input":"hello"}`))
	rec := httptest.NewRecorder()

	NewServer(NewService(&fakeRuntime{}, conversation.NewInMemoryStore(4))).ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}
```

- [ ] **Step 2: Run the proxy-service tests to verify they fail**

Run: `go test ./internal/proxy ./internal/httpserver -run 'Test(ServiceRespondUsesPreviousResponseSnapshot|ServiceRespondRebuildsTurnLocalSystemContext|ServiceRespondPersistsAssistantReplyInSnapshot|ServiceRespondUsesIncomingModelForContinuation|ServiceStreamPersistsOnlyAfterCleanCompletion|ServiceStreamDoesNotPersistFailedStream|ResponsesHandlerReturnsBadRequestForUnknownPreviousResponseID)' -v`
Expected: FAIL because the stateful proxy service and continuation behavior do not exist

- [ ] **Step 3: Implement the service, wire startup, and persist successful turns**

```go
type Service struct {
	bedrock Caller
	store   conversation.Store
}

func (s *Service) Respond(ctx context.Context, req openai.ResponsesRequest) (openai.Response, error) {
	base, err := s.loadPrevious(req.PreviousResponseID)
	if err != nil {
		return openai.Response{}, err
	}
	current, err := conversation.NormalizeRequest(req)
	if err != nil {
		return openai.Response{}, err
	}
	merged := conversation.Merge(base, current)
	resp, assistantText, err := s.bedrock.Respond(ctx, req.Model, merged, req.MaxOutputTokens, req.Temperature)
	if err != nil {
		return openai.Response{}, err
	}
	snapshot := conversation.AppendAssistantReply(merged, assistantText)
	s.store.Save(conversation.RecordFromResponse(resp.ID, req.Model, snapshot))
	return resp, nil
}

func (s *Service) Stream(ctx context.Context, req openai.ResponsesRequest, w http.ResponseWriter) error {
	base, err := s.loadPrevious(req.PreviousResponseID)
	if err != nil {
		return err
	}
	current, err := conversation.NormalizeRequest(req)
	if err != nil {
		return err
	}
	merged := conversation.Merge(base, current)
	resp, assistantText, err := s.bedrock.Stream(ctx, req.Model, merged, req.MaxOutputTokens, req.Temperature, w)
	if err != nil {
		return err
	}
	snapshot := conversation.AppendAssistantReply(merged, assistantText)
	s.store.Save(conversation.RecordFromResponse(resp.ID, req.Model, snapshot))
	return nil
}
```

`internal/proxy/service.go` should own:

- `loadPrevious(previousResponseID string) (conversation.Request, error)`
- the rule that stored `ModelID` is advisory only and the incoming request `Model` is authoritative
- the rule that only successful stream completion produces a persisted snapshot

`internal/conversation/normalize.go` should own:

- `Merge(base, current Request) Request`
- `AppendAssistantReply(req Request, assistantText string) Request`

`internal/conversation/store.go` should own:

- `type Store interface { Get(string) (Record, bool); Save(Record) }`
- `RecordFromResponse(responseID, modelID string, snapshot Request) Record`

Also wire the production default store size explicitly in `cmd/openai-bedrock-proxy/main.go`:

```go
const defaultResponseStoreCapacity = 4096

svc := proxy.NewService(client, conversation.NewInMemoryStore(defaultResponseStoreCapacity))
```

- [ ] **Step 4: Run the focused service and handler tests to verify they pass**

Run: `go test ./internal/proxy ./internal/httpserver -run 'Test(ServiceRespondUsesPreviousResponseSnapshot|ServiceRespondRebuildsTurnLocalSystemContext|ServiceRespondPersistsAssistantReplyInSnapshot|ServiceRespondUsesIncomingModelForContinuation|ServiceStreamPersistsOnlyAfterCleanCompletion|ServiceStreamDoesNotPersistFailedStream|ResponsesHandlerReturnsBadRequestForUnknownPreviousResponseID)' -v`
Expected: PASS

- [ ] **Step 5: Run full verification and commit**

Run: `go test ./... -v`
Expected: PASS

```bash
git add internal/proxy/service.go internal/proxy/service_test.go internal/httpserver/server_test.go cmd/openai-bedrock-proxy/main.go
git commit -m "feat: add responses continuation state"
```
