# AWS Sample Feature Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the local Go Bedrock proxy to practical feature parity with `aws-samples/bedrock-access-gateway` for local usage, while preserving the existing Responses API and adding Chat Completions, Embeddings, multimodal chat, richer model discovery, and advanced Bedrock feature mapping.

**Architecture:** Keep `internal/bedrock` as the single Bedrock integration layer. Add dedicated OpenAI Chat Completions and Embeddings request/response types under `internal/openai`, add API-specific orchestration in `internal/proxy`, and extend `internal/httpserver` to serve the broader OpenAI-compatible surface. Model discovery and provider-specific capability checks should sit behind a narrow Bedrock catalog layer so chat, multimodal, reasoning, prompt caching, and embeddings use one source of truth.

**Tech Stack:** Go 1.24+, `net/http`, AWS SDK for Go v2 (`config`, `service/bedrock`, `service/bedrockruntime`), `httptest`, standard library JSON/SSE, existing in-repo fakes and table-driven unit tests

---

## File Structure

Planned files and responsibilities:

- Modify: `README.md`
  - Document local parity endpoints, supported request fields, and `AWS_*` credential usage.
- Modify: `internal/config/config.go`
  - Add runtime settings needed for parity features, starting with global prompt-caching enablement.
- Modify: `internal/config/config_test.go`
  - Cover defaults and overrides for the new settings.
- Create: `internal/openai/chat_types.go`
  - OpenAI Chat Completions request and response types.
- Create: `internal/openai/chat_validate.go`
  - Validation rules for chat request shapes and supported field combinations.
- Create: `internal/openai/chat_validate_test.go`
  - Validation tests for chat requests.
- Create: `internal/openai/embeddings_types.go`
  - OpenAI embeddings request and response types.
- Create: `internal/openai/embeddings_validate.go`
  - Validation for embeddings requests.
- Create: `internal/openai/embeddings_validate_test.go`
  - Embeddings validation tests.
- Modify: `internal/openai/errors.go`
  - Add a typed not-found error for model lookup without changing existing invalid-request behavior.
- Create: `internal/bedrock/model_catalog.go`
  - Bedrock model discovery, inference-profile merging, deterministic ordering, and capability metadata.
- Create: `internal/bedrock/model_catalog_test.go`
  - Tests for catalog merging and profile resolution.
- Create: `internal/bedrock/chat_translate.go`
  - Chat Completions request translation to Bedrock `Converse`/`ConverseStream`.
- Create: `internal/bedrock/chat_translate_test.go`
  - Chat translation tests, including tools and field precedence.
- Create: `internal/bedrock/chat_stream.go`
  - Chat Completions streaming chunk shaping from Bedrock stream events.
- Create: `internal/bedrock/chat_stream_test.go`
  - Streaming delta and usage-chunk tests.
- Create: `internal/bedrock/image.go`
  - Remote image/data URL parsing for multimodal chat.
- Create: `internal/bedrock/image_test.go`
  - Data URL and invalid-image parsing tests.
- Create: `internal/bedrock/embeddings.go`
  - Model-family embedding adapters and request routing.
- Create: `internal/bedrock/embeddings_test.go`
  - Adapter selection and response parsing tests.
- Modify: `internal/bedrock/client.go`
  - Wire catalog APIs, chat invocation helpers, advanced inference config, and embedding invocation support.
- Modify: `internal/bedrock/client_test.go`
  - Integration-style unit tests for Bedrock adapter behavior.
- Create: `internal/proxy/chat_service.go`
  - Chat Completions orchestration over the Bedrock layer.
- Create: `internal/proxy/chat_service_test.go`
  - Chat service orchestration tests.
- Create: `internal/proxy/embeddings_service.go`
  - Embeddings orchestration over the Bedrock layer.
- Create: `internal/proxy/embeddings_service_test.go`
  - Embeddings service tests.
- Modify: `internal/httpserver/server.go`
  - Route registration and handlers for new parity endpoints.
- Modify: `internal/httpserver/server_test.go`
  - Endpoint coverage for chat, embeddings, model lookup, and health.

### Task 1: Add Model Catalog And Capability Plumbing

**Files:**
- Create: `internal/bedrock/model_catalog.go`
- Test: `internal/bedrock/model_catalog_test.go`
- Modify: `internal/bedrock/client.go`
- Modify: `internal/bedrock/client_test.go`

- [ ] **Step 1: Write failing catalog tests for merged discovery**

```go
func TestCatalogListModelsMergesFoundationAndInferenceProfiles(t *testing.T) {
	catalog := newFakeCatalogAPI()
	catalog.foundationModels = []ModelRecord{
		{ID: "anthropic.claude-3-7-sonnet-20250219-v1:0", InputModalities: []string{"TEXT", "IMAGE"}},
	}
	catalog.systemProfiles = []InferenceProfileRecord{
		{ID: "us.anthropic.claude-3-7-sonnet-20250219-v1:0", SourceModelID: "anthropic.claude-3-7-sonnet-20250219-v1:0"},
	}
	catalog.applicationProfiles = []InferenceProfileRecord{
		{ID: "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/app-profile", SourceModelID: "anthropic.claude-3-7-sonnet-20250219-v1:0"},
	}
	got, err := BuildModelCatalog(context.Background(), catalog)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Models) != 3 {
		t.Fatalf("expected merged catalog, got %#v", got.Models)
	}
}

func TestCatalogResolveModelReturnsUnderlyingFoundationModel(t *testing.T) {
	catalog := Catalog{
		ByID: map[string]ModelRecord{
			"us.anthropic.claude-3-7-sonnet-20250219-v1:0": {
				ID: "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
				ResolvedFoundationModelID: "anthropic.claude-3-7-sonnet-20250219-v1:0",
			},
		},
	}
	got, ok := catalog.Resolve("us.anthropic.claude-3-7-sonnet-20250219-v1:0")
	if !ok || got.ResolvedFoundationModelID != "anthropic.claude-3-7-sonnet-20250219-v1:0" {
		t.Fatalf("expected profile resolution, got %#v ok=%v", got, ok)
	}
}
```

- [ ] **Step 2: Run the catalog tests to verify they fail**

Run: `go test ./internal/bedrock -run 'TestCatalog(ListModelsMergesFoundationAndInferenceProfiles|ResolveModelReturnsUnderlyingFoundationModel)' -v`
Expected: FAIL because the catalog layer does not exist yet

- [ ] **Step 3: Implement catalog types and profile-aware discovery**

```go
type ModelRecord struct {
	ID                        string
	Name                      string
	Provider                  string
	InputModalities           []string
	ModelKind                 string
	ResolvedFoundationModelID string
}

type Catalog struct {
	Models []ModelRecord
	ByID   map[string]ModelRecord
}

func (c Catalog) Resolve(id string) (ModelRecord, bool) {
	record, ok := c.ByID[id]
	return record, ok
}
```

```go
func BuildModelCatalog(ctx context.Context, api CatalogAPI) (Catalog, error) {
	// Call ListFoundationModels plus both system-defined and application
	// inference-profile listing APIs, merge resolvable IDs into one catalog,
	// and sort the final model slice by ID for deterministic output.
}
```

- [ ] **Step 4: Extend the Bedrock client to use the catalog for model listing**

```go
func (c *Client) ListModels(ctx context.Context) ([]ModelSummary, error) {
	catalog, err := c.Catalog(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]ModelSummary, 0, len(catalog.Models))
	for _, model := range catalog.Models {
		out = append(out, ModelSummary{
			ID:       model.ID,
			Name:     model.Name,
			Provider: model.Provider,
		})
	}
	return out, nil
}
```

- [ ] **Step 5: Add a models wire-shape regression test**

```go
func TestListModelsPreservesOpenAIListShapeWhileExpandingCatalog(t *testing.T) {
	svc := proxy.NewService(fakeCatalogClientWithProfiles(), conversation.NewInMemoryStore(16))
	got, err := svc.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Object != "list" || len(got.Data) == 0 {
		t.Fatalf("unexpected list payload: %#v", got)
	}
	first := got.Data[0]
	if first.ID == "" || first.Object != "model" || first.OwnedBy == "" {
		t.Fatalf("unexpected model payload: %#v", first)
	}
}
```

- [ ] **Step 6: Run catalog and client tests to verify they pass**

Run: `go test ./internal/bedrock ./internal/proxy -run 'TestCatalog|TestClientListModels|TestListModelsPreservesOpenAIListShapeWhileExpandingCatalog' -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add internal/bedrock/model_catalog.go internal/bedrock/model_catalog_test.go internal/bedrock/client.go internal/bedrock/client_test.go internal/proxy/service_test.go
git commit -m "feat: add bedrock model catalog"
```

### Task 2: Add Chat Completions Types And Validation

**Files:**
- Create: `internal/openai/chat_types.go`
- Create: `internal/openai/chat_validate.go`
- Test: `internal/openai/chat_validate_test.go`

- [ ] **Step 1: Write failing validation tests for supported chat fields**

```go
func TestValidateChatRequestAcceptsBasicMessages(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "model",
		Messages: []ChatMessage{
			{Role: "system", Content: "be concise"},
			{Role: "user", Content: "hello"},
		},
	}
	if err := ValidateChatCompletionRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateChatRequestUsesMaxCompletionTokensPrecedence(t *testing.T) {
	req := ChatCompletionRequest{
		Model:               "model",
		Messages:            []ChatMessage{{Role: "user", Content: "hello"}},
		MaxTokens:           intPtr(256),
		MaxCompletionTokens: intPtr(1024),
	}
	if err := ValidateChatCompletionRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateChatRequestRejectsUnsupportedToolChoiceShape(t *testing.T) {
	req := ChatCompletionRequest{
		Model:    "model",
		Messages: []ChatMessage{{Role: "user", Content: "hello"}},
		ToolChoice: map[string]any{
			"type": "function",
			"name": "lookup",
		},
	}
	assertInvalidRequestMessage(t, ValidateChatCompletionRequest(req), "tool_choice is invalid")
}
```

- [ ] **Step 2: Run the chat validation tests to verify they fail**

Run: `go test ./internal/openai -run 'TestValidateChatRequest' -v`
Expected: FAIL because chat types and validators do not exist yet

- [ ] **Step 3: Add chat request/response types with explicit unions**

```go
type ChatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []ChatMessage   `json:"messages"`
	Stream              bool            `json:"stream,omitempty"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stop                any             `json:"stop,omitempty"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          any             `json:"tool_choice,omitempty"`
	ReasoningEffort     string          `json:"reasoning_effort,omitempty"`
	ExtraBody           map[string]any  `json:"extra_body,omitempty"`
}
```

- [ ] **Step 4: Implement strict validation for chat message roles and field precedence**

```go
func resolvedMaxTokens(req ChatCompletionRequest) *int {
	if req.MaxCompletionTokens != nil {
		return req.MaxCompletionTokens
	}
	return req.MaxTokens
}

func validateChatToolChoice(value any, tools []Tool) error {
	// Accept "auto", "required", or {"type":"function","function":{"name":"..."}}.
}
```

- [ ] **Step 5: Run the chat validation tests to verify they pass**

Run: `go test ./internal/openai -run 'TestValidateChatRequest' -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add internal/openai/chat_types.go internal/openai/chat_validate.go internal/openai/chat_validate_test.go
git commit -m "feat: add chat completion request validation"
```

### Task 3: Implement Non-Streaming Chat Completions Translation And Service

**Files:**
- Create: `internal/bedrock/chat_translate.go`
- Test: `internal/bedrock/chat_translate_test.go`
- Create: `internal/proxy/chat_service.go`
- Test: `internal/proxy/chat_service_test.go`
- Modify: `internal/bedrock/client.go`
- Modify: `internal/bedrock/client_test.go`

- [ ] **Step 1: Write failing translation tests for chat messages, tools, and finish reasons**

```go
func TestTranslateChatRequestBuildsSystemUserAndToolResultBlocks(t *testing.T) {
	req := openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{
			{Role: "system", Content: "be terse"},
			{Role: "user", Content: "hello"},
			{Role: "tool", ToolCallID: "call_1", Content: "sunny"},
		},
	}
	got, err := TranslateChatRequest(req, fakeCatalogRecord("model"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.System) != 1 || len(got.Messages) != 2 {
		t.Fatalf("unexpected translated request: %#v", got)
	}
}

func TestTranslateChatResponseBuildsToolCalls(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "abc123",
		Output: []OutputBlock{{
			Type: OutputBlockTypeToolCall,
			ToolCall: &ToolCall{ID: "call_1", Name: "lookup", Arguments: "{\"q\":\"weather\"}"},
		}},
		StopReason: "tool_use",
	}
	got := TranslateChatResponse(resp, "model")
	if got.Choices[0].Message.ToolCalls[0].Function.Name != "lookup" {
		t.Fatalf("expected tool call output, got %#v", got)
	}
}

func TestTranslateChatResponsePlacesReasoningInReasoningContent(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "abc123",
		Output: []OutputBlock{
			{Type: OutputBlockTypeReasoning, Text: "draft reasoning"},
			{Type: OutputBlockTypeText, Text: "final answer"},
		},
		StopReason: "end_turn",
	}
	got := TranslateChatResponse(resp, "model")
	if got.Choices[0].Message.ReasoningContent != "draft reasoning" {
		t.Fatalf("expected reasoning_content, got %#v", got.Choices[0].Message)
	}
	if got.Choices[0].Message.Content != "final answer" {
		t.Fatalf("expected assistant text in content, got %#v", got.Choices[0].Message)
	}
	if strings.Contains(got.Choices[0].Message.Content, "<think>") {
		t.Fatalf("expected no synthetic think tags, got %#v", got.Choices[0].Message)
	}
}
```

- [ ] **Step 2: Run the chat translation and service tests to verify they fail**

Run: `go test ./internal/bedrock ./internal/proxy -run 'TestTranslateChat|TestChatService' -v`
Expected: FAIL because the chat path does not exist yet

- [ ] **Step 3: Add Bedrock chat request/response translation**

```go
func TranslateChatRequest(req openai.ChatCompletionRequest, model ModelRecord) (ConverseRequest, error) {
	// Convert chat messages into Bedrock system blocks, conversation messages,
	// toolUse/toolResult blocks, inference config, and tool config.
}

func TranslateChatResponse(resp ConverseResponse, model string) openai.ChatCompletionResponse {
	// Map Bedrock output text/tool-use blocks into OpenAI chat completion choices.
}
```

- [ ] **Step 4: Add a dedicated chat service in `internal/proxy`**

```go
type ChatService struct {
	client BedrockChatAPI
}

func (s *ChatService) Complete(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	record, err := s.client.LookupModel(ctx, req.Model)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}
	translated, err := bedrock.TranslateChatRequest(req, record)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}
	resp, err := s.client.Chat(ctx, translated)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}
	return bedrock.TranslateChatResponse(resp, req.Model), nil
}
```

- [ ] **Step 5: Run the chat translation and service tests to verify they pass**

Run: `go test ./internal/bedrock ./internal/proxy -run 'TestTranslateChat|TestChatService' -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add internal/bedrock/chat_translate.go internal/bedrock/chat_translate_test.go internal/proxy/chat_service.go internal/proxy/chat_service_test.go internal/bedrock/client.go internal/bedrock/client_test.go
git commit -m "feat: add non-streaming chat completions"
```

### Task 4: Implement Chat Completions Streaming Chunk Fidelity

**Files:**
- Create: `internal/bedrock/chat_stream.go`
- Test: `internal/bedrock/chat_stream_test.go`
- Modify: `internal/bedrock/client.go`
- Modify: `internal/bedrock/client_test.go`
- Modify: `internal/proxy/chat_service.go`
- Modify: `internal/proxy/chat_service_test.go`

- [ ] **Step 1: Write failing streaming tests for text deltas, tool-call deltas, and usage chunks**

```go
func TestWriteChatStreamEmitsTextChunksAndDone(t *testing.T) {
	stream := newFakeChatStream(
		textDelta("hel"),
		textDelta("lo"),
		messageStop("end_turn"),
		metadataUsage(10, 2),
	)
	var buf bytes.Buffer
	err := WriteChatCompletionsStream(stream, "chatcmpl_123", "model", true, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "\"object\":\"chat.completion.chunk\"") {
		t.Fatalf("expected chunk output, got %s", buf.String())
	}
}

func TestWriteChatStreamEmitsReasoningDeltaSeparately(t *testing.T) {
	stream := newFakeChatStream(reasoningDelta("think"), messageStop("end_turn"))
	var buf bytes.Buffer
	err := WriteChatCompletionsStream(stream, "chatcmpl_123", "model", false, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "\"reasoning_content\":\"think\"") {
		t.Fatalf("expected reasoning delta, got %s", buf.String())
	}
}
```

- [ ] **Step 2: Run the streaming tests to verify they fail**

Run: `go test ./internal/bedrock ./internal/proxy -run 'TestWriteChatStream|TestChatServiceStream' -v`
Expected: FAIL because the chat streaming adapter does not exist yet

- [ ] **Step 3: Implement chat chunk shaping**

```go
func WriteChatCompletionsStream(stream streamEvents, responseID string, model string, includeUsage bool, w io.Writer) error {
	// Emit chat.completion.chunk payloads for content, reasoning_content,
	// tool_calls deltas, optional usage chunk, and final [DONE].
}
```

- [ ] **Step 4: Extend the chat service to invoke Bedrock streaming**

```go
func (s *ChatService) Stream(ctx context.Context, req openai.ChatCompletionRequest, w io.Writer) error {
	record, err := s.client.LookupModel(ctx, req.Model)
	if err != nil {
		return err
	}
	translated, err := bedrock.TranslateChatRequest(req, record)
	if err != nil {
		return err
	}
	resp, err := s.client.ChatStream(ctx, translated)
	if err != nil {
		return err
	}
	return bedrock.WriteChatCompletionsStream(resp.Stream, resp.ResponseID, req.Model, includeUsage(req), w)
}
```

- [ ] **Step 5: Run the streaming tests to verify they pass**

Run: `go test ./internal/bedrock ./internal/proxy -run 'TestWriteChatStream|TestChatServiceStream' -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add internal/bedrock/chat_stream.go internal/bedrock/chat_stream_test.go internal/bedrock/client.go internal/bedrock/client_test.go internal/proxy/chat_service.go internal/proxy/chat_service_test.go
git commit -m "feat: add chat completion streaming"
```

### Task 5: Add Multimodal Chat Input Support

**Files:**
- Create: `internal/bedrock/image.go`
- Test: `internal/bedrock/image_test.go`
- Modify: `internal/openai/chat_validate.go`
- Modify: `internal/openai/chat_validate_test.go`
- Modify: `internal/bedrock/chat_translate.go`
- Modify: `internal/bedrock/chat_translate_test.go`

- [ ] **Step 1: Write failing tests for text-plus-image user content**

```go
func TestValidateChatRequestAcceptsTextAndImageParts(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "model",
		Messages: []ChatMessage{{
			Role: "user",
			Content: []ChatContentPart{
				{Type: "text", Text: "describe this"},
				{Type: "image_url", ImageURL: &ImageURL{URL: "data:image/png;base64,aGVsbG8="}},
			},
		}},
	}
	if err := ValidateChatCompletionRequest(req); err != nil {
		t.Fatalf("expected valid multimodal request, got %v", err)
	}
}

func TestTranslateChatRequestRejectsImageForTextOnlyModel(t *testing.T) {
	req := openai.ChatCompletionRequest{
		Model: "text-only-model",
		Messages: []openai.ChatMessage{{
			Role: "user",
			Content: []openai.ChatContentPart{{Type: "image_url", ImageURL: &openai.ImageURL{URL: "data:image/png;base64,aGVsbG8="}}},
		}},
	}
	_, err := TranslateChatRequest(req, fakeCatalogRecord("text-only-model", "TEXT"))
	assertInvalidRequestMessage(t, err, "multimodal message is not supported by this model")
}
```

- [ ] **Step 2: Run the multimodal tests to verify they fail**

Run: `go test ./internal/openai ./internal/bedrock -run 'Test(ValidateChatRequestAcceptsTextAndImageParts|TranslateChatRequestRejectsImageForTextOnlyModel|ParseImage)' -v`
Expected: FAIL because multimodal content is not supported yet

- [ ] **Step 3: Implement image parsing helpers for data URLs and remote URLs**

```go
func ParseImageURL(raw string, fetch func(string) ([]byte, string, error)) ([]byte, string, error) {
	// Support data:image/...;base64,... and fetched remote image URLs.
}
```

- [ ] **Step 4: Translate structured user content into Bedrock text/image blocks**

```go
func toChatContentBlocks(parts []openai.ChatContentPart, model ModelRecord) ([]ContentBlock, error) {
	// Validate IMAGE modality and map text/image parts into Bedrock blocks.
}
```

- [ ] **Step 5: Run the multimodal tests to verify they pass**

Run: `go test ./internal/openai ./internal/bedrock -run 'Test(ValidateChatRequestAcceptsTextAndImageParts|TranslateChatRequestRejectsImageForTextOnlyModel|ParseImage)' -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add internal/bedrock/image.go internal/bedrock/image_test.go internal/openai/chat_validate.go internal/openai/chat_validate_test.go internal/bedrock/chat_translate.go internal/bedrock/chat_translate_test.go
git commit -m "feat: add multimodal chat support"
```

### Task 6: Add Advanced Bedrock Request Features

**Files:**
- Modify: `internal/config/config.go`
- Modify: `internal/config/config_test.go`
- Modify: `internal/bedrock/chat_translate.go`
- Modify: `internal/bedrock/chat_translate_test.go`
- Modify: `internal/bedrock/client.go`
- Modify: `internal/bedrock/client_test.go`

- [ ] **Step 1: Write failing tests for `top_p`, `stop`, `reasoning_effort`, and prompt caching**

```go
func TestTranslateChatRequestUsesMaxCompletionTokensForReasoningBudget(t *testing.T) {
	req := openai.ChatCompletionRequest{
		Model:               "anthropic.claude-sonnet",
		Messages:            []openai.ChatMessage{{Role: "user", Content: "hello"}},
		MaxTokens:           intPtr(256),
		MaxCompletionTokens: intPtr(2048),
		ReasoningEffort:     "low",
	}
	got, err := TranslateChatRequest(req, fakeReasoningModel("anthropic.claude-sonnet"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.MaxTokens == nil || *got.MaxTokens != 2048 {
		t.Fatalf("expected max_completion_tokens precedence, got %#v", got.MaxTokens)
	}
}

func TestTranslateChatRequestConsumesPromptCachingControlsFromExtraBody(t *testing.T) {
	req := openai.ChatCompletionRequest{
		Model:    "amazon.nova-pro",
		Messages: []openai.ChatMessage{{Role: "system", Content: "cached prompt"}, {Role: "user", Content: "hello"}},
		ExtraBody: map[string]any{
			"prompt_caching": map[string]any{"system": true, "messages": true},
			"thinking":       map[string]any{"type": "enabled", "budget_tokens": 4096},
		},
	}
	got, err := TranslateChatRequest(req, fakeCachingModel("amazon.nova-pro"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.AdditionalModelRequestFields["prompt_caching"] != nil {
		t.Fatalf("expected prompt_caching to be consumed locally, got %#v", got.AdditionalModelRequestFields)
	}
}

func TestTranslateChatRequestInjectsCachePointsForSupportedModel(t *testing.T) {
	req := openai.ChatCompletionRequest{
		Model: "amazon.nova-pro",
		Messages: []openai.ChatMessage{
			{Role: "system", Content: "cached system"},
			{Role: "user", Content: "cached user"},
		},
		ExtraBody: map[string]any{
			"prompt_caching": map[string]any{"system": true, "messages": true},
		},
	}
	got, err := TranslateChatRequest(req, fakeCachingModel("amazon.nova-pro"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !hasSystemCachePoint(got.System) || !lastUserMessageHasCachePoint(got.Messages) {
		t.Fatalf("expected cache points in translated request, got %#v", got)
	}
}
```

- [ ] **Step 2: Run the advanced-feature tests to verify they fail**

Run: `go test ./internal/config ./internal/bedrock -run 'Test(TranslateChatRequestUsesMaxCompletionTokensForReasoningBudget|TranslateChatRequestConsumesPromptCachingControlsFromExtraBody|TranslateChatRequestInjectsCachePointsForSupportedModel|LoadConfigPromptCaching)' -v`
Expected: FAIL because the advanced request fields are not implemented yet

- [ ] **Step 3: Add config for global prompt-caching enablement**

```go
type Config struct {
	ListenAddr          string
	ListenPort          string
	AWSRegion           string
	LogLevel            string
	EnablePromptCaching bool
}
```

- [ ] **Step 4: Expand Bedrock chat translation for advanced fields**

```go
type ConverseRequest struct {
	ModelID                      string
	System                       []string
	Messages                     []Message
	MaxTokens                    *int32
	Temperature                  *float32
	TopP                         *float32
	StopSequences                []string
	ToolConfig                   *ToolConfig
	AdditionalModelRequestFields map[string]any
}
```

```go
func consumePromptCaching(extra map[string]any) (PromptCachingConfig, map[string]any, error) {
	// Read extra_body.prompt_caching.system/messages booleans,
	// remove them from pass-through fields, and return the remainder.
}

func applyPromptCaching(req *ConverseRequest, cfg PromptCachingConfig, model ModelRecord, enabledByDefault bool) {
	// Inject cache points into system content and the last user message
	// only when the resolved model supports caching and the request opts in.
}
```

- [ ] **Step 5: Run the advanced-feature tests to verify they pass**

Run: `go test ./internal/config ./internal/bedrock -run 'Test(TranslateChatRequestUsesMaxCompletionTokensForReasoningBudget|TranslateChatRequestConsumesPromptCachingControlsFromExtraBody|TranslateChatRequestInjectsCachePointsForSupportedModel|LoadConfigPromptCaching)' -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add internal/config/config.go internal/config/config_test.go internal/bedrock/chat_translate.go internal/bedrock/chat_translate_test.go internal/bedrock/client.go internal/bedrock/client_test.go
git commit -m "feat: add advanced bedrock chat controls"
```

### Task 7: Add Embeddings Types, Adapters, And Service

**Files:**
- Create: `internal/openai/embeddings_types.go`
- Create: `internal/openai/embeddings_validate.go`
- Test: `internal/openai/embeddings_validate_test.go`
- Create: `internal/bedrock/embeddings.go`
- Test: `internal/bedrock/embeddings_test.go`
- Create: `internal/proxy/embeddings_service.go`
- Test: `internal/proxy/embeddings_service_test.go`
- Modify: `internal/bedrock/client.go`
- Modify: `internal/bedrock/client_test.go`

- [ ] **Step 1: Write failing embeddings validation and adapter tests**

```go
func TestValidateEmbeddingsRequestAcceptsFloatAndBase64(t *testing.T) {
	for _, format := range []string{"float", "base64"} {
		req := EmbeddingsRequest{Model: "cohere.embed-english-v3", Input: "hello", EncodingFormat: format}
		if err := ValidateEmbeddingsRequest(req); err != nil {
			t.Fatalf("expected %s to validate, got %v", format, err)
		}
	}
}

func TestValidateEmbeddingsRequestRejectsDimensionsForNonNovaModel(t *testing.T) {
	req := EmbeddingsRequest{
		Model:      "cohere.embed-english-v3",
		Input:      "hello",
		Dimensions: intPtr(256),
	}
	assertInvalidRequestMessage(t, ValidateEmbeddingsRequest(req), "dimensions is only supported for Nova embedding models")
}

func TestSelectEmbeddingAdapterByModelFamily(t *testing.T) {
	for _, tc := range []struct {
		model string
		want  string
	}{
		{"cohere.embed-english-v3", "cohere"},
		{"amazon.titan-embed-text-v2:0", "titan"},
		{"amazon.nova-2-multimodal-embeddings-v1:0", "nova"},
	} {
		got, err := selectEmbeddingAdapter(tc.model)
		if err != nil || got.Name() != tc.want {
			t.Fatalf("model %q -> %#v %v", tc.model, got, err)
		}
	}
}
```

- [ ] **Step 2: Run the embeddings tests to verify they fail**

Run: `go test ./internal/openai ./internal/bedrock ./internal/proxy -run 'Test(ValidateEmbeddingsRequest|SelectEmbeddingAdapter|EmbeddingsService)' -v`
Expected: FAIL because the embeddings path does not exist yet

- [ ] **Step 3: Add OpenAI embeddings request/response types and validation**

```go
type EmbeddingsRequest struct {
	Model          string `json:"model"`
	Input          any    `json:"input"`
	EncodingFormat string `json:"encoding_format,omitempty"`
	Dimensions     *int   `json:"dimensions,omitempty"`
}
```

```go
func ValidateEmbeddingsRequest(req EmbeddingsRequest) error {
	// Require model, support string or []string input,
	// accept encoding_format float/base64, and reject dimensions unless the
	// model is a supported Nova embeddings model.
}
```

- [ ] **Step 4: Implement model-family embedding adapters and service**

```go
type EmbeddingAdapter interface {
	Name() string
	Embed(ctx context.Context, req openai.EmbeddingsRequest) (openai.EmbeddingsResponse, error)
}

func selectEmbeddingAdapter(modelID string) (EmbeddingAdapter, error) {
	// Dispatch Cohere, Titan, and Nova by model prefix / exact supported IDs.
}
```

- [ ] **Step 5: Run the embeddings tests to verify they pass**

Run: `go test ./internal/openai ./internal/bedrock ./internal/proxy -run 'Test(ValidateEmbeddingsRequest|SelectEmbeddingAdapter|EmbeddingsService)' -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add internal/openai/embeddings_types.go internal/openai/embeddings_validate.go internal/openai/embeddings_validate_test.go internal/bedrock/embeddings.go internal/bedrock/embeddings_test.go internal/proxy/embeddings_service.go internal/proxy/embeddings_service_test.go internal/bedrock/client.go internal/bedrock/client_test.go
git commit -m "feat: add embeddings support"
```

### Task 8: Extend HTTP Routes For Parity Endpoints

**Files:**
- Modify: `internal/httpserver/server.go`
- Modify: `internal/httpserver/server_test.go`
- Modify: `internal/proxy/service.go`
- Modify: `internal/proxy/service_test.go`
- Modify: `cmd/openai-bedrock-proxy/main.go`
- Modify: `internal/openai/errors.go`

- [ ] **Step 1: Write failing handler tests for chat, embeddings, model lookup, and health**

```go
func TestChatCompletionsHandlerReturnsJSON(t *testing.T) {
	svc := &fakeCompositeService{
		chatResponse: openai.ChatCompletionResponse{
			ID:    "chatcmpl_123",
			Model: "model",
			Choices: []openai.ChatChoice{{
				Message: openai.ChatResponseMessage{Role: "assistant", Content: "hello"},
			}},
		},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"model","messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	NewServer(svc).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}

func TestChatCompletionsHandlerStreamsSSEWhenRequested(t *testing.T) {
	svc := &fakeCompositeService{chatStreamBody: "data: {\"object\":\"chat.completion.chunk\"}\n\ndata: [DONE]\n\n"}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"model","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`))
	rec := httptest.NewRecorder()
	NewServer(svc).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Fatalf("expected SSE content type, got %q", got)
	}
	if !strings.Contains(rec.Body.String(), "[DONE]") {
		t.Fatalf("expected stream terminator, got %s", rec.Body.String())
	}
}

func TestModelByIDHandlerReturns404ForUnknownModel(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/v1/models/missing", nil)
	rec := httptest.NewRecorder()
	NewServer(&fakeCompositeService{lookupModelErr: openai.NewNotFoundError("model not found")}).ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", rec.Code)
	}
}

func TestModelByIDHandlerReturnsModelPayloadOnSuccess(t *testing.T) {
	svc := &fakeCompositeService{
		modelLookup: openai.Model{
			ID:      "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
			Object:  "model",
			OwnedBy: "Anthropic",
			Name:    "Claude 3.7 Sonnet",
		},
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/models/us.anthropic.claude-3-7-sonnet-20250219-v1:0", nil)
	rec := httptest.NewRecorder()
	NewServer(svc).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"object":"model"`) || !strings.Contains(rec.Body.String(), `"id":"us.anthropic.claude-3-7-sonnet-20250219-v1:0"`) {
		t.Fatalf("unexpected model payload: %s", rec.Body.String())
	}
}

func TestHealthHandlerReturnsOK(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	NewServer(&fakeCompositeService{}).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "\"status\":\"ok\"") {
		t.Fatalf("unexpected health response: %d %s", rec.Code, rec.Body.String())
	}
}
```

- [ ] **Step 2: Run the handler tests to verify they fail**

Run: `go test ./internal/httpserver -run 'Test(ChatCompletionsHandler|EmbeddingsHandler|ModelByIDHandler|HealthHandler)' -v`
Expected: FAIL because the new routes and service interfaces do not exist yet

- [ ] **Step 3: Extend the composite HTTP service interfaces and route table**

```go
type Service interface {
	Respond(context.Context, openai.ResponsesRequest) (openai.Response, error)
	ListModels(context.Context) (openai.ModelsList, error)
	GetModel(context.Context, string) (openai.Model, error)
	CompleteChat(context.Context, openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
	Embed(context.Context, openai.EmbeddingsRequest) (openai.EmbeddingsResponse, error)
}
```

```go
func NewServer(svc Service) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", handleResponses(svc))
	mux.HandleFunc("POST /v1/chat/completions", handleChatCompletions(svc))
	mux.HandleFunc("POST /v1/embeddings", handleEmbeddings(svc))
	mux.HandleFunc("GET /v1/models", handleModels(svc))
	mux.HandleFunc("GET /v1/models/{id}", handleModelByID(svc))
	mux.HandleFunc("GET /health", handleHealth())
	return mux
}
```

Streaming handler rule:

```go
if req.Stream {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	return streamChatCompletions(svc, req, w)
}
```

- [ ] **Step 4: Map unknown-model errors to `404` and preserve existing Responses behavior**

```go
type NotFoundError struct{ Message string }

func (e NotFoundError) Error() string { return e.Message }
func (e NotFoundError) NotFound() bool { return true }

func statusCodeFor(err error) int {
	var notFound interface{ NotFound() bool }
	if errors.As(err, &notFound) && notFound.NotFound() {
		return http.StatusNotFound
	}
	// Keep existing invalid-request and upstream error mappings after this.
}
```

- [ ] **Step 5: Run the handler tests to verify they pass**

Run: `go test ./internal/httpserver -run 'Test(ChatCompletionsHandler|EmbeddingsHandler|ModelByIDHandler|HealthHandler)' -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add internal/httpserver/server.go internal/httpserver/server_test.go internal/proxy/service.go internal/proxy/service_test.go cmd/openai-bedrock-proxy/main.go internal/openai/errors.go
git commit -m "feat: add parity http endpoints"
```

### Task 9: Update README And Run Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add local parity documentation**

Update `README.md` to document:

- local run command
- reliance on AWS default credential chain
- supported endpoints:
  - `/v1/responses`
  - `/v1/chat/completions`
  - `/v1/embeddings`
  - `/v1/models`
  - `/v1/models/{id}`
  - `/health`
- parity caveats:
  - no inbound auth
  - no CORS layer
  - local use focus

Suggested section:

```md
## Supported Endpoints

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `GET /v1/models`
- `GET /v1/models/{id}`
- `GET /health`
```

- [ ] **Step 2: Run the full test suite**

Run: `go test ./...`
Expected: PASS

- [ ] **Step 3: Smoke-check the binary startup**

Run: `go test ./cmd/openai-bedrock-proxy ./internal/httpserver -v`
Expected: PASS

- [ ] **Step 4: Review git diff for scope correctness**

Run: `git log --oneline --stat --max-count 12`
Expected: only parity-related files changed

- [ ] **Step 5: Commit docs and verification-safe cleanup**

```bash
git add README.md
git commit -m "docs: describe parity endpoints"
```
