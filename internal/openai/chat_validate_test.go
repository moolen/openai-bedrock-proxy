package openai

import (
	"encoding/json"
	"testing"
)

func TestValidateChatRequestAcceptsBasicMessages(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "model",
		Messages: []ChatMessage{
			{Role: "system", Content: ChatMessageText("be concise")},
			{Role: "user", Content: ChatMessageText("hello")},
		},
	}
	if err := ValidateChatCompletionRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateChatRequestUsesMaxCompletionTokensPrecedence(t *testing.T) {
	req := ChatCompletionRequest{
		Model:               "model",
		Messages:            []ChatMessage{{Role: "user", Content: ChatMessageText("hello")}},
		MaxTokens:           intPtr(256),
		MaxCompletionTokens: intPtr(1024),
	}
	if err := ValidateChatCompletionRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateChatRequestUsesMaxCompletionTokensPrecedenceWhenMaxTokensInvalid(t *testing.T) {
	req := ChatCompletionRequest{
		Model:               "model",
		Messages:            []ChatMessage{{Role: "user", Content: ChatMessageText("hello")}},
		MaxTokens:           intPtr(0),
		MaxCompletionTokens: intPtr(128),
	}
	if err := ValidateChatCompletionRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateChatRequestUsesMaxCompletionTokensPrecedenceWhenMaxCompletionTokensInvalid(t *testing.T) {
	req := ChatCompletionRequest{
		Model:               "model",
		Messages:            []ChatMessage{{Role: "user", Content: ChatMessageText("hello")}},
		MaxTokens:           intPtr(128),
		MaxCompletionTokens: intPtr(0),
	}
	assertInvalidRequestMessage(t, ValidateChatCompletionRequest(req), "max_completion_tokens must be greater than 0")
}

func TestValidateChatRequestRejectsUnsupportedToolChoiceShape(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"messages":[{"role":"user","content":"hello"}],
		"tool_choice":{"type":"function","name":"lookup"}
	}`)
	var req ChatCompletionRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	assertInvalidRequestMessage(t, ValidateChatCompletionRequest(req), "tool_choice is invalid")
}

func TestValidateChatRequestRejectsUnsupportedRole(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "model",
		Messages: []ChatMessage{
			{Role: "critic", Content: ChatMessageText("hello")},
		},
	}
	assertInvalidRequestMessage(t, ValidateChatCompletionRequest(req), "messages[0].role is invalid")
}

func TestValidateChatRequestAcceptsToolChoiceRequired(t *testing.T) {
	req := ChatCompletionRequest{
		Model:      "model",
		Messages:   []ChatMessage{{Role: "user", Content: ChatMessageText("hello")}},
		ToolChoice: ChatToolChoiceString("required"),
	}
	if err := ValidateChatCompletionRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateChatRequestAcceptsFunctionToolChoiceShapeWithoutTools(t *testing.T) {
	req := ChatCompletionRequest{
		Model:      "model",
		Messages:   []ChatMessage{{Role: "user", Content: ChatMessageText("hello")}},
		ToolChoice: ChatToolChoiceFunctionName("lookup"),
	}
	if err := ValidateChatCompletionRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func intPtr(value int) *int {
	return &value
}
