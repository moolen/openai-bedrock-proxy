# Tool Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add non-streaming tool support to `POST /v1/responses` so Codex can send tools, receive model tool calls translated from Bedrock, execute them locally, and send tool results back through the proxy.

**Architecture:** Replace the current text-only normalized conversation model with a block-based internal representation that can persist text, assistant tool calls, and user tool results across `previous_response_id`. Translate OpenAI/Codex-facing tools into Bedrock `ToolConfiguration`, including synthetic adapters for built-in OpenAI tools, then translate Bedrock tool-use responses back into Codex-usable output items.

**Tech Stack:** Go, standard library `encoding/json`, AWS SDK v2 Bedrock Runtime, existing `internal/openai`, `internal/conversation`, `internal/bedrock`, and `internal/proxy` packages.

---

### File Structure

**Modify:**

- `internal/openai/types.go`
- `internal/openai/validate.go`
- `internal/openai/validate_test.go`
- `internal/conversation/types.go`
- `internal/conversation/normalize.go`
- `internal/conversation/normalize_test.go`
- `internal/conversation/store.go`
- `internal/conversation/store_test.go`
- `internal/bedrock/translate_request.go`
- `internal/bedrock/translate_request_test.go`
- `internal/bedrock/translate_response.go`
- `internal/bedrock/translate_response_test.go`
- `internal/bedrock/client.go`
- `internal/bedrock/client_test.go`
- `internal/proxy/service.go`
- `internal/proxy/service_test.go`
- `internal/httpserver/server_test.go`

**Maybe modify if compilation requires type propagation changes:**

- `internal/httpserver/server.go`

**Do not modify in this slice:**

- streaming tool event handling beyond keeping current text-only stream path intact

### Task 1: Expand OpenAI Request Types And Validation

**Files:**

- Modify: `internal/openai/types.go`
- Modify: `internal/openai/validate.go`
- Test: `internal/openai/validate_test.go`

- [ ] **Step 1: Write failing validation tests for supported tool payloads**

Add tests that describe the new accepted subset:

```go
func TestValidateResponsesRequestAcceptsFunctionTools(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{
				Type: "function",
				Function: ToolFunction{
					Name:        "lookup",
					Description: "Find records",
					Parameters: map[string]any{
						"type": "object",
					},
				},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
}
```

Add similar tests for:

- supported built-in tool types in `Tools`
- `tool_choice: "auto"`
- explicit named tool choice object
- continued rejection of `parallel_tool_calls`
- rejection of malformed function tools

- [ ] **Step 2: Run validation tests to verify they fail for the new support cases**

Run: `go test ./internal/openai -run 'TestValidateResponsesRequest(Accepts|Rejects)'`

Expected: FAIL because `ValidateResponsesRequest` still rejects `tools` and `tool_choice`.

- [ ] **Step 3: Expand request type definitions to model the accepted subset**

Update `internal/openai/types.go` to introduce structured tool and tool-choice types instead of using only `any` where clarity is needed.

Target shape:

```go
type Tool struct {
	Type     string         `json:"type"`
	Name     string         `json:"name,omitempty"`
	Function ToolFunction   `json:"function"`
	Metadata map[string]any `json:"-"`
}

type ToolChoice struct {
	Type     string `json:"type,omitempty"`
	Name     string `json:"name,omitempty"`
	Function struct {
		Name string `json:"name,omitempty"`
	} `json:"function,omitempty"`
}
```

Keep top-level decoding permissive enough to accept Codex/OpenAI wire shapes.

- [ ] **Step 4: Implement strict supported-subset validation**

In `internal/openai/validate.go`, replace the blanket rejections with helper validation:

```go
if err := validateTools(req.Tools); err != nil {
	return err
}
if err := validateToolChoice(req.ToolChoice, req.Tools); err != nil {
	return err
}
if req.ParallelToolCalls != nil {
	return NewInvalidRequestError("parallel_tool_calls is not supported")
}
```

Validation rules:

- `function` tools need a non-empty `function.name`
- built-in tool types are accepted only if listed in a table-driven allowlist
- malformed tool configs return `invalid_request_error`
- unmappable `tool_choice` returns `invalid_request_error`

- [ ] **Step 5: Run tests to verify validation passes**

Run: `go test ./internal/openai -run 'TestValidateResponsesRequest(Accepts|Rejects)'`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/openai/types.go internal/openai/validate.go internal/openai/validate_test.go
git commit -m "feat: validate supported responses tools"
```

### Task 2: Replace The Text-Only Conversation Model With Blocks

**Files:**

- Modify: `internal/conversation/types.go`
- Modify: `internal/conversation/normalize.go`
- Test: `internal/conversation/normalize_test.go`

- [ ] **Step 1: Write failing normalization tests for mixed text and tool-result input**

Add tests for:

- function tools normalize into internal tool definitions
- built-in tools normalize into synthetic internal tool definitions
- tool-result input items normalize into `user` message blocks
- mixed text and tool-result content preserves order

Representative test:

```go
func TestNormalizeRequestBuildsToolResultBlocks(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use the result"},
					map[string]any{
						"type": "function_call_output",
						"call_id": "call_123",
						"output": map[string]any{"ok": true},
					},
				},
			},
		},
	}
	got, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Messages) != 1 || len(got.Messages[0].Blocks) != 2 {
		t.Fatalf("expected mixed blocks, got %#v", got.Messages)
	}
}
```

- [ ] **Step 2: Run normalization tests to verify they fail**

Run: `go test ./internal/conversation -run 'TestNormalizeRequest|TestMerge|TestAppendAssistantReply'`

Expected: FAIL because the conversation model is still `Role + Text`.

- [ ] **Step 3: Expand the internal conversation types**

Replace `Message{Role, Text}` with a block-based model in `internal/conversation/types.go`.

Target shape:

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
```

Include types for:

- text blocks
- tool call blocks
- tool result blocks
- normalized tool definitions
- normalized tool choice

- [ ] **Step 4: Rewrite normalization to emit blocks instead of plain text**

Refactor `internal/conversation/normalize.go`:

- parse message content into ordered blocks
- keep `system`/`developer` as turn-local system strings
- normalize `tools` and `tool_choice` onto the request
- preserve existing text-only behavior as a degenerate case

Implementation style:

```go
func normalizeContentItems(role string, content any) ([]Block, error) {
	switch value := content.(type) {
	case string:
		return []Block{{Type: BlockTypeText, Text: value}}, nil
	case []any:
		return normalizeBlockItems(role, value)
	default:
		return nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
}
```

- [ ] **Step 5: Rework `Merge` and assistant append helpers**

Update `Merge` to append block-based messages and preserve current request tool definitions.

Replace `AppendAssistantReply` with a helper that appends an assistant `Message` built from response blocks instead of raw text.

- [ ] **Step 6: Run tests to verify the block-based model passes**

Run: `go test ./internal/conversation`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add internal/conversation/types.go internal/conversation/normalize.go internal/conversation/normalize_test.go
git commit -m "feat: normalize responses conversations with tool blocks"
```

### Task 3: Preserve Tool Blocks In Stored Continuation State

**Files:**

- Modify: `internal/conversation/store.go`
- Test: `internal/conversation/store_test.go`

- [ ] **Step 1: Write failing store tests for block-based snapshots**

Add tests that prove `RecordFromResponse`, `Save`, and `Get` deep-clone nested blocks and tool metadata.

Representative test:

```go
func TestRecordFromResponseClonesToolBlocks(t *testing.T) {
	req := Request{
		Messages: []Message{
			{
				Role: "assistant",
				Blocks: []Block{
					{Type: BlockTypeToolCall, ToolCall: &ToolCall{ID: "call_1", Name: "lookup"}},
				},
			},
		},
	}
	record := RecordFromResponse("resp_1", "model", req)
	req.Messages[0].Blocks[0].ToolCall.Name = "mutated"
	if record.Messages[0].Blocks[0].ToolCall.Name != "lookup" {
		t.Fatalf("expected deep clone, got %#v", record.Messages)
	}
}
```

- [ ] **Step 2: Run store tests to verify they fail**

Run: `go test ./internal/conversation -run 'Test(InMemoryStore|RecordFromResponse)'`

Expected: FAIL because clone logic only handles flat text messages.

- [ ] **Step 3: Update store cloning helpers for nested blocks**

Refactor `cloneMessages` and related helpers to deep-copy:

- blocks
- tool call structs
- tool result structs
- normalized tool definitions when persisted

- [ ] **Step 4: Run store tests to verify they pass**

Run: `go test ./internal/conversation -run 'Test(InMemoryStore|RecordFromResponse)'`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/conversation/store.go internal/conversation/store_test.go
git commit -m "feat: persist tool-aware conversation snapshots"
```

### Task 4: Translate Tool-Capable Conversations Into Bedrock Requests

**Files:**

- Modify: `internal/bedrock/translate_request.go`
- Test: `internal/bedrock/translate_request_test.go`
- Modify if needed: `internal/bedrock/client.go`
- Test if needed: `internal/bedrock/client_test.go`

- [ ] **Step 1: Write failing request-translation tests for tools**

Add tests that prove:

- function tools map to Bedrock `ToolConfiguration`
- built-in tools become synthetic Bedrock tool specs
- assistant tool-call history becomes Bedrock `toolUse`
- user tool-result history becomes Bedrock `toolResult`
- `tool_choice:auto` maps to Bedrock auto

Representative test:

```go
func TestTranslateConversationMapsFunctionTools(t *testing.T) {
	req := conversation.Request{
		Messages: []conversation.Message{
			{Role: "user", Blocks: []conversation.Block{{Type: conversation.BlockTypeText, Text: "hi"}}},
		},
		Tools: []conversation.ToolDefinition{
			{
				Kind:        conversation.ToolKindFunction,
				Name:        "lookup",
				Description: "Find data",
				Schema:      map[string]any{"type": "object"},
			},
		},
	}
	got, err := TranslateConversation("model", req, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ToolConfig == nil || len(got.ToolConfig.Tools) != 1 {
		t.Fatalf("expected one bedrock tool, got %#v", got.ToolConfig)
	}
}
```

- [ ] **Step 2: Run Bedrock translation tests to verify they fail**

Run: `go test ./internal/bedrock -run 'TestTranslate(Request|Conversation)'`

Expected: FAIL because the Bedrock translator only emits text content today.

- [ ] **Step 3: Expand the internal Bedrock request structs**

Update `internal/bedrock/translate_request.go` to represent Bedrock tool-capable content:

```go
type ContentBlock struct {
	Text       string
	ToolUse    *ToolUse
	ToolResult *ToolResult
}

type ToolConfig struct {
	Tools      []ToolSpec
	ToolChoice ToolChoice
}
```

Keep the local structs easy to test before converting into AWS SDK types.

- [ ] **Step 4: Implement request translation helpers**

Add helpers for:

- `toBedrockContentBlocks`
- `toToolConfig`
- `toBedrockToolChoice`
- synthetic built-in tool naming and schema generation

Minimal mapping pattern:

```go
func toToolSpec(tool conversation.ToolDefinition) (ToolSpec, error) {
	switch tool.Kind {
	case conversation.ToolKindFunction:
		return ToolSpec{Name: tool.Name, Description: tool.Description, Schema: tool.Schema}, nil
	case conversation.ToolKindSyntheticBuiltIn:
		return ToolSpec{Name: tool.Name, Description: tool.Description, Schema: tool.Schema}, nil
	default:
		return ToolSpec{}, openai.NewInvalidRequestError("unsupported tool type")
	}
}
```

- [ ] **Step 5: Wire translated tool config into AWS SDK conversion**

In `internal/bedrock/client.go`, extend:

- `toConverseInput`
- `toConverseStreamInput`
- `toBedrockMessages`

so they emit Bedrock SDK `ToolConfiguration`, `ContentBlockMemberToolUse`, and `ContentBlockMemberToolResult` where present.

- [ ] **Step 6: Run Bedrock tests to verify request translation passes**

Run: `go test ./internal/bedrock -run 'TestTranslate(Request|Conversation)|TestClientRespond|TestClientStream'`

Expected: PASS for non-streaming tool tests and no regressions in existing text tests.

- [ ] **Step 7: Commit**

```bash
git add internal/bedrock/translate_request.go internal/bedrock/translate_request_test.go internal/bedrock/client.go internal/bedrock/client_test.go
git commit -m "feat: translate tool-capable conversations to bedrock"
```

### Task 5: Translate Bedrock Tool-Use Responses Back To OpenAI/Codex Output

**Files:**

- Modify: `internal/bedrock/translate_response.go`
- Test: `internal/bedrock/translate_response_test.go`
- Modify if needed: `internal/bedrock/client.go`
- Test if needed: `internal/bedrock/client_test.go`

- [ ] **Step 1: Write failing response-translation tests for tool use**

Add tests that prove:

- text output still maps to `output_text`
- Bedrock tool-use blocks map to client-facing tool-call output items
- mixed text plus tool-use preserves order
- synthetic built-in tool metadata maps back to the original OpenAI tool type

Representative test:

```go
func TestTranslateConverseResponseBuildsFunctionCallOutput(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Output: []OutputBlock{
			{Type: OutputBlockTypeToolCall, ToolCall: &ToolCall{ID: "toolu_1", Name: "lookup", Arguments: map[string]any{"q": "hi"}}},
		},
	}
	got := TranslateResponse(resp, "model")
	if len(got.Output) != 1 {
		t.Fatalf("expected one output item, got %d", len(got.Output))
	}
	if got.Output[0].Type != "function_call" {
		t.Fatalf("expected function_call output, got %#v", got.Output[0])
	}
}
```

- [ ] **Step 2: Run response translation tests to verify they fail**

Run: `go test ./internal/bedrock -run 'TestTranslateConverseResponse|TestClientRespond'`

Expected: FAIL because `ConverseResponse` currently stores only `Text`.

- [ ] **Step 3: Expand Bedrock response structs to carry ordered output blocks**

Refactor `internal/bedrock/translate_response.go`:

```go
type ConverseResponse struct {
	ResponseID string
	Output     []OutputBlock
	StopReason string
}
```

Where `OutputBlock` can represent:

- assistant text
- tool call

- [ ] **Step 4: Parse Bedrock output blocks in order**

In `internal/bedrock/client.go`, replace `extractText` with a helper that walks Bedrock output content and returns ordered blocks instead of concatenated text only.

Pattern:

```go
func extractOutput(resp *bedrockruntime.ConverseOutput) []OutputBlock {
	// append text and tool_use blocks in encounter order
}
```

- [ ] **Step 5: Translate ordered output blocks into OpenAI response items**

Update `TranslateResponse` so:

- text blocks stay under assistant `message` content items
- tool-call blocks produce Codex-usable output items
- a mixed response returns output in encounter order rather than flattening to text

- [ ] **Step 6: Run response translation tests to verify they pass**

Run: `go test ./internal/bedrock -run 'TestTranslateConverseResponse|TestClientRespond'`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add internal/bedrock/translate_response.go internal/bedrock/translate_response_test.go internal/bedrock/client.go internal/bedrock/client_test.go
git commit -m "feat: translate bedrock tool use to responses output"
```

### Task 6: Wire Proxy Continuation And Persistence For Tool Loops

**Files:**

- Modify: `internal/proxy/service.go`
- Test: `internal/proxy/service_test.go`

- [ ] **Step 1: Write failing proxy tests for tool-loop continuation**

Add tests that prove:

- tool definitions from the incoming request reach the Bedrock client
- assistant tool-call responses are persisted in the stored snapshot
- follow-up tool-result requests merge with the prior snapshot under `previous_response_id`

Representative test:

```go
func TestServiceRespondPersistsAssistantToolCallInSnapshot(t *testing.T) {
	store := newRecordingStore()
	client := &fakeBedrock{
		respondResp: bedrock.ConverseResponse{
			ResponseID: "abc",
			Output: []bedrock.OutputBlock{
				{Type: bedrock.OutputBlockTypeToolCall, ToolCall: &bedrock.ToolCall{ID: "call_1", Name: "lookup"}},
			},
		},
	}
	svc := NewService(client, store)
	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []openai.Tool{{Type: "function", Function: openai.ToolFunction{Name: "lookup"}}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := store.last.Messages[1].Blocks[0].ToolCall.Name; got != "lookup" {
		t.Fatalf("expected tool call to persist, got %#v", store.last.Messages)
	}
}
```

- [ ] **Step 2: Run proxy tests to verify they fail**

Run: `go test ./internal/proxy -run 'TestServiceRespond|TestServiceStream|TestServiceListModels'`

Expected: FAIL because proxy persistence helpers still assume assistant text only.

- [ ] **Step 3: Update proxy response handling to persist ordered response blocks**

Refactor `internal/proxy/service.go` so:

- the Bedrock response is translated to an OpenAI response as before
- the stored snapshot appends an assistant message built from Bedrock output blocks
- follow-up requests merge prior tool calls and new tool results without flattening to text

- [ ] **Step 4: Keep streaming behavior unchanged**

Do not add tool persistence for streaming in this slice. Keep the existing stream path text-only and ensure tests still pass.

- [ ] **Step 5: Run proxy tests to verify they pass**

Run: `go test ./internal/proxy`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/proxy/service.go internal/proxy/service_test.go
git commit -m "feat: persist tool loops across responses continuations"
```

### Task 7: Cover The HTTP Layer And End-To-End Request Acceptance

**Files:**

- Modify: `internal/httpserver/server_test.go`
- Modify if needed: `internal/httpserver/server.go`

- [ ] **Step 1: Write failing HTTP tests for non-streaming tool requests**

Add handler tests for:

- `POST /v1/responses` with function tools returns `200`
- malformed tool definitions return `400`
- non-streaming tool-use responses are serialized in the HTTP response body

Representative test:

```go
func TestServerAcceptsResponsesRequestWithTools(t *testing.T) {
	body := []byte(`{
		"model":"model",
		"input":"hi",
		"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]
	}`)
	// assert 200 and that fake service was called
}
```

- [ ] **Step 2: Run handler tests to verify they fail**

Run: `go test ./internal/httpserver -run 'TestServer'`

Expected: FAIL because request validation currently blocks tools.

- [ ] **Step 3: Adjust server assertions only if response serialization changed**

If the response body format changes because tool calls are now output items instead of only assistant text content, update only the tests and any necessary JSON assertions.

- [ ] **Step 4: Run handler tests to verify they pass**

Run: `go test ./internal/httpserver -run 'TestServer'`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/httpserver/server_test.go internal/httpserver/server.go
git commit -m "test: cover tool-capable responses requests"
```

### Task 8: Full Verification And Regression Sweep

**Files:**

- No planned source edits

- [ ] **Step 1: Run focused package tests**

Run: `go test ./internal/openai ./internal/conversation ./internal/bedrock ./internal/proxy ./internal/httpserver`

Expected: all selected packages PASS.

- [ ] **Step 2: Run the full test suite**

Run: `go test ./...`

Expected: PASS.

- [ ] **Step 3: Review the diff for accidental streaming or API-surface regressions**

Run: `git diff --stat`

Expected: only the planned files changed.

- [ ] **Step 4: Commit the final integrated change**

```bash
git add internal/openai internal/conversation internal/bedrock internal/proxy internal/httpserver
git commit -m "feat: add non-streaming responses tool compatibility"
```
