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

func TestValidateResponsesRequestRejectsStructuredInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: []map[string]any{{"type": "input_text", "text": "hi"}},
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected structured-input validation error")
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
