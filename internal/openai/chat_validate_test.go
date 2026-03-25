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

func TestValidateChatRequestRejectsToolChoiceFunctionWithExtraTopLevelKey(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"messages":[{"role":"user","content":"hello"}],
		"tool_choice":{"type":"function","function":{"name":"lookup"},"mode":"x"}
	}`)
	var req ChatCompletionRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	assertInvalidRequestMessage(t, ValidateChatCompletionRequest(req), "tool_choice is invalid")
}

func TestValidateChatRequestRejectsToolChoiceFunctionWithExtraFunctionKey(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"messages":[{"role":"user","content":"hello"}],
		"tool_choice":{"type":"function","function":{"name":"lookup","x":"y"}}
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

func TestValidateChatRequestAcceptsStructuredContentArray(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"messages":[
			{
				"role":"user",
				"content":[
					{"type":"text","text":"describe this"},
					{"type":"image_url","image_url":{"url":"https://example.com/cat.png","detail":"high"}}
				]
			}
		]
	}`)
	var req ChatCompletionRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	if err := ValidateChatCompletionRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestChatMessageContentMarshalUsesOpenAIWireShape(t *testing.T) {
	resp := ChatCompletionResponse{
		ID:      "chatcmpl_123",
		Object:  "chat.completion",
		Created: 1,
		Model:   "model",
		Choices: []ChatCompletionChoice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: ChatMessageText("hello"),
				},
				FinishReason: "stop",
			},
		},
	}
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("expected json marshal to succeed, got %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	choices, ok := got["choices"].([]any)
	if !ok || len(choices) != 1 {
		t.Fatalf("expected one choice, got %#v", got["choices"])
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		t.Fatalf("expected object choice, got %#v", choices[0])
	}
	message, ok := choice["message"].(map[string]any)
	if !ok {
		t.Fatalf("expected object message, got %#v", choice["message"])
	}
	content, ok := message["content"].(string)
	if !ok || content != "hello" {
		t.Fatalf("expected content to be wire string \"hello\", got %#v", message["content"])
	}
}

func TestChatMessageContentImagePartRoundTripPreservesWireData(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"messages":[
			{
				"role":"user",
				"content":[
					{"type":"image_url","image_url":{"url":"https://example.com/cat.png","detail":"high"}}
				]
			}
		]
	}`)
	var req ChatCompletionRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}

	encoded, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("expected json marshal to succeed, got %v", err)
	}

	var roundTripped map[string]any
	if err := json.Unmarshal(encoded, &roundTripped); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	messages, ok := roundTripped["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("expected one message, got %#v", roundTripped["messages"])
	}
	message, ok := messages[0].(map[string]any)
	if !ok {
		t.Fatalf("expected object message, got %#v", messages[0])
	}
	content, ok := message["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("expected one content part, got %#v", message["content"])
	}
	part, ok := content[0].(map[string]any)
	if !ok {
		t.Fatalf("expected object part, got %#v", content[0])
	}
	image, ok := part["image_url"].(map[string]any)
	if !ok {
		t.Fatalf("expected image_url object, got %#v", part["image_url"])
	}
	if image["url"] != "https://example.com/cat.png" {
		t.Fatalf("expected image_url.url to be preserved, got %#v", image["url"])
	}
	if image["detail"] != "high" {
		t.Fatalf("expected image_url.detail to be preserved, got %#v", image["detail"])
	}
}

func intPtr(value int) *int {
	return &value
}
