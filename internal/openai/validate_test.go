package openai

import "testing"

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
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected unsupported content block validation error")
	}
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
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected non-message item validation error")
	}
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
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected assistant input_text block validation error")
	}
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
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected user output_text block validation error")
	}
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

func TestValidateResponsesRequestRejectsTools(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{{Type: "function", Function: ToolFunction{Name: "lookup"}}},
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected tools validation error")
	}
}

func TestValidateResponsesRequestRejectsToolChoice(t *testing.T) {
	req := ResponsesRequest{
		Model:      "model",
		Input:      "hi",
		ToolChoice: "auto",
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected tool_choice validation error")
	}
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
