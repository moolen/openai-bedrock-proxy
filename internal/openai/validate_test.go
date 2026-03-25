package openai

import "testing"

const invalidResponsesInputMessage = "input must be a non-empty string or supported message object/array"

func TestValidateResponsesRequestAcceptsSimpleTextInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "anthropic.claude-3-7-sonnet-20250219-v1:0",
		Input: "write a haiku",
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsEasyMessageInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role":    "user",
			"content": "hello",
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected easy message input to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsExplicitMessageInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"type": "message",
			"role": "user",
			"content": []map[string]any{
				{"type": "input_text", "text": "hello"},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected explicit message input to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsPlainStringRegression(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hello",
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected plain string input to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsArrayOfMessages(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{
				"role":    "user",
				"content": "hello",
			},
			{
				"type": "message",
				"role": "developer",
				"content": []map[string]any{
					{"type": "input_text", "text": "be brief"},
				},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected array of messages input to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsAssistantEasyMessage(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": "hi"},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected assistant easy message input to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestRejectsUnsupportedContentBlock(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "user",
			"content": []map[string]any{
				{"type": "input_image", "image_url": "https://example.com/cat.png"},
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsNonMessageItem(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: []any{
			map[string]any{
				"role":    "user",
				"content": "hello",
			},
			"not-a-message",
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsAssistantInputTextBlocks(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "assistant",
			"content": []map[string]any{
				{"type": "input_text", "text": "hi"},
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsUserOutputTextBlocks(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "user",
			"content": []map[string]any{
				{"type": "output_text", "text": "hi"},
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsEmptyEasyMessageContent(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role":    "user",
			"content": "",
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsEmptyTextBlockContent(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": ""},
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsMissingModel(t *testing.T) {
	req := ResponsesRequest{Input: "hi"}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected missing-model validation error")
	}
}

func TestValidateResponsesRequestRejectsMissingInput(t *testing.T) {
	req := ResponsesRequest{Model: "model"}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected missing-input validation error")
	}
}

func TestValidateResponsesRequestRejectsEmptyStringInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "",
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected empty-input validation error")
	}
}

func TestValidateResponsesRequestRejectsUnsupportedFields(t *testing.T) {
	req := ResponsesRequest{
		Model:             "model",
		Input:             "hi",
		ParallelToolCalls: ptr(true),
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected unsupported-field validation error")
	}
}

func TestValidateResponsesRequestAcceptsFunctionTools(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{{Type: "function", Function: &ToolFunction{Name: "lookup"}}},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected tools to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsBuiltInTools(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{Type: "web_search_preview"},
			{Type: "file_search"},
			{Type: "computer_use_preview"},
			{Type: "code_interpreter"},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected built-in tools to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsToolChoiceAuto(t *testing.T) {
	req := ResponsesRequest{
		Model:      "model",
		Input:      "hi",
		ToolChoice: "auto",
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected tool_choice auto to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsNamedFunctionToolChoice(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{Type: "function", Function: &ToolFunction{Name: "lookup"}},
		},
		ToolChoice: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "lookup",
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected named function tool_choice to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestRejectsMalformedFunctionTool(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{{Type: "function"}},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tools[0].function.name is required")
}

func TestValidateResponsesRequestRejectsMalformedToolChoice(t *testing.T) {
	req := ResponsesRequest{
		Model:      "model",
		Input:      "hi",
		ToolChoice: map[string]any{"type": "function", "function": map[string]any{}},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice.function.name is required")
}

func TestValidateResponsesRequestRejectsUnmappableToolChoice(t *testing.T) {
	req := ResponsesRequest{
		Model:      "model",
		Input:      "hi",
		ToolChoice: 42,
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice is invalid")
}

func TestValidateResponsesRequestRejectsUnsupportedBuiltInTool(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{Type: "image_generation"},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tools[0].type is not supported")
}

func TestErrorResponseFromClassifiesWrappedInvalidRequestErrors(t *testing.T) {
	err := NewInvalidRequestError("bad request")
	resp := ErrorResponseFrom(wrapError{err: err})
	if resp.Error.Type != "invalid_request_error" {
		t.Fatalf("expected invalid request error type, got %q", resp.Error.Type)
	}
	if resp.Error.Message != "bad request" {
		t.Fatalf("expected wrapped error message, got %q", resp.Error.Message)
	}
}

func assertInvalidRequestMessage(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatal("expected validation error")
	}
	if err.Error() != want {
		t.Fatalf("expected error %q, got %q", want, err.Error())
	}
}

func ptr[T any](value T) *T {
	return &value
}

type wrapError struct {
	err error
}

func (e wrapError) Error() string {
	return "wrapped: " + e.err.Error()
}

func (e wrapError) Unwrap() error {
	return e.err
}
