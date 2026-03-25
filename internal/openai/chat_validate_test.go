package openai

import "testing"

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

func TestValidateChatRequestRejectsUnsupportedRole(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "model",
		Messages: []ChatMessage{
			{Role: "critic", Content: "hello"},
		},
	}
	assertInvalidRequestMessage(t, ValidateChatCompletionRequest(req), "messages[0].role is invalid")
}

func TestValidateChatRequestAcceptsToolChoiceRequired(t *testing.T) {
	req := ChatCompletionRequest{
		Model:      "model",
		Messages:   []ChatMessage{{Role: "user", Content: "hello"}},
		ToolChoice: "required",
	}
	if err := ValidateChatCompletionRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func intPtr(value int) *int {
	return &value
}
