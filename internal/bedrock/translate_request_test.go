package bedrock

import (
	"math"
	"testing"

	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

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
	if len(got.Messages) != 1 {
		t.Fatalf("expected one message, got %d", len(got.Messages))
	}
	if got.Messages[0].Role != "user" {
		t.Fatalf("expected user role, got %q", got.Messages[0].Role)
	}
	if len(got.Messages[0].Content) != 1 {
		t.Fatalf("expected one content block, got %d", len(got.Messages[0].Content))
	}
	if got.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("expected message text to pass through, got %q", got.Messages[0].Content[0].Text)
	}
}

func TestTranslateConversationBuildsBedrockMessages(t *testing.T) {
	request := conversation.Request{
		System: []string{"be precise"},
		Messages: []conversation.Message{
			{Role: "user", Text: "hello"},
			{Role: "assistant", Text: "hi"},
		},
	}

	got, err := TranslateConversation("model", request, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ModelID != "model" {
		t.Fatalf("expected model passthrough, got %q", got.ModelID)
	}
	if len(got.System) != 1 || got.System[0] != "be precise" {
		t.Fatalf("expected system to map, got %#v", got.System)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("expected two messages, got %d", len(got.Messages))
	}
	if got.Messages[0].Role != "user" || got.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("expected user message to map, got %#v", got.Messages[0])
	}
	if got.Messages[1].Role != "assistant" || got.Messages[1].Content[0].Text != "hi" {
		t.Fatalf("expected assistant message to map, got %#v", got.Messages[1])
	}
}

func TestTranslateConversationMapsInferenceControls(t *testing.T) {
	maxTokens := 128
	temperature := 0.4
	request := conversation.Request{Messages: []conversation.Message{{Role: "user", Text: "hello"}}}

	got, err := TranslateConversation("model", request, &maxTokens, &temperature)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.MaxTokens == nil || *got.MaxTokens != int32(maxTokens) {
		t.Fatalf("expected max tokens to map, got %v", got.MaxTokens)
	}
	if got.Temperature == nil || *got.Temperature != float32(temperature) {
		t.Fatalf("expected temperature to map, got %v", got.Temperature)
	}
}

func TestTranslateConversationRejectsOutOfRangeMaxTokens(t *testing.T) {
	maxTokens := math.MaxInt32 + 1
	request := conversation.Request{Messages: []conversation.Message{{Role: "user", Text: "hello"}}}

	if _, err := TranslateConversation("model", request, &maxTokens, nil); err == nil {
		t.Fatal("expected out-of-range max tokens error")
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
	if got.System[0] != "be terse" {
		t.Fatalf("expected system instructions to pass through, got %q", got.System[0])
	}
}

func TestTranslateRequestMapsInferenceControls(t *testing.T) {
	maxTokens := 512
	temperature := 0.75
	req := openai.ResponsesRequest{
		Model:           "model",
		Input:           "hello",
		MaxOutputTokens: &maxTokens,
		Temperature:     &temperature,
	}
	got, err := TranslateRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.MaxTokens == nil || *got.MaxTokens != 512 {
		t.Fatalf("expected max tokens to map, got %v", got.MaxTokens)
	}
	if got.Temperature == nil || *got.Temperature != float32(temperature) {
		t.Fatalf("expected temperature to map, got %v", got.Temperature)
	}
}

func TestTranslateRequestRejectsOutOfRangeMaxTokens(t *testing.T) {
	maxTokens := math.MaxInt32 + 1
	req := openai.ResponsesRequest{
		Model:           "model",
		Input:           "hello",
		MaxOutputTokens: &maxTokens,
	}
	if _, err := TranslateRequest(req); err == nil {
		t.Fatal("expected out-of-range max tokens error")
	}
}

func TestTranslateRequestRejectsUnsupportedInputTypes(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []string{"hello"},
	}
	if _, err := TranslateRequest(req); err == nil {
		t.Fatal("expected unsupported input type error")
	}
}
