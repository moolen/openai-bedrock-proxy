package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

func TestTranslateChatRequestBuildsSystemUserAndToolResultBlocks(t *testing.T) {
	req := openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{
			{Role: "system", Content: openai.ChatMessageText("be terse")},
			{Role: "user", Content: openai.ChatMessageText("hello")},
			{Role: "tool", ToolCallID: "call_1", Content: openai.ChatMessageText("sunny")},
		},
	}

	got, err := TranslateChatRequest(context.Background(), req, fakeCatalogRecord("model"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.System) != 1 || len(got.Messages) != 2 {
		t.Fatalf("unexpected translated request: %#v", got)
	}
	if got.System[0] != "be terse" {
		t.Fatalf("expected system message text, got %#v", got.System)
	}
	if got.Messages[0].Role != "user" || len(got.Messages[0].Content) != 1 || got.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("expected user text message, got %#v", got.Messages[0])
	}
	if got.Messages[1].Role != "user" || len(got.Messages[1].Content) != 1 || got.Messages[1].Content[0].ToolResult == nil {
		t.Fatalf("expected synthetic user tool result, got %#v", got.Messages[1])
	}
	if got.Messages[1].Content[0].ToolResult.ToolUseID != "call_1" {
		t.Fatalf("expected tool call id to map, got %#v", got.Messages[1].Content[0].ToolResult)
	}
	if len(got.Messages[1].Content[0].ToolResult.Content) != 1 || got.Messages[1].Content[0].ToolResult.Content[0].Text != "sunny" {
		t.Fatalf("expected tool result text payload, got %#v", got.Messages[1].Content[0].ToolResult.Content)
	}
}

func TestTranslateChatRequestBuildsAssistantToolUseAndToolConfig(t *testing.T) {
	req := openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{
			{
				Role:    "assistant",
				Content: openai.ChatMessageText("checking"),
				ToolCalls: []openai.ChatToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: openai.ChatToolCallFunction{
							Name:      "lookup",
							Arguments: `{"q":"weather"}`,
						},
					},
				},
			},
		},
		Tools: []openai.Tool{
			{
				Type: "function",
				Function: &openai.ToolFunction{
					Name:        "lookup",
					Description: "lookup weather",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"q": map[string]any{"type": "string"},
						},
					},
				},
			},
			{
				Type: "web_search_preview",
				Config: map[string]json.RawMessage{
					"user_location": json.RawMessage(`{"type":"approximate","country":"DE"}`),
				},
			},
		},
		ToolChoice: openai.ChatToolChoiceString("auto"),
	}

	got, err := TranslateChatRequest(context.Background(), req, fakeCatalogRecord("model"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "assistant" {
		t.Fatalf("expected translated assistant message, got %#v", got.Messages)
	}
	if len(got.Messages[0].Content) != 2 {
		t.Fatalf("expected text+tool_use content blocks, got %#v", got.Messages[0].Content)
	}
	if got.Messages[0].Content[0].Text != "checking" {
		t.Fatalf("expected assistant text block first, got %#v", got.Messages[0].Content[0])
	}
	if got.Messages[0].Content[1].ToolUse == nil {
		t.Fatalf("expected assistant tool_use block, got %#v", got.Messages[0].Content[1])
	}
	if got.Messages[0].Content[1].ToolUse.ToolUseID != "call_1" || got.Messages[0].Content[1].ToolUse.Name != "lookup" {
		t.Fatalf("unexpected translated assistant tool_use: %#v", got.Messages[0].Content[1].ToolUse)
	}
	input, ok := got.Messages[0].Content[1].ToolUse.Input.(map[string]any)
	if !ok || input["q"] != "weather" {
		t.Fatalf("expected parsed assistant tool args object, got %#v", got.Messages[0].Content[1].ToolUse.Input)
	}
	if got.ToolConfig == nil || len(got.ToolConfig.Tools) != 2 {
		t.Fatalf("expected translated tool config, got %#v", got.ToolConfig)
	}
	if got.ToolConfig.ToolChoice == nil || !got.ToolConfig.ToolChoice.Auto {
		t.Fatalf("expected translated auto tool choice, got %#v", got.ToolConfig.ToolChoice)
	}
}

func TestTranslateChatRequestRejectsRequiredToolChoice(t *testing.T) {
	req := openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: openai.ChatMessageText("hello")},
		},
		Tools: []openai.Tool{
			{
				Type: "function",
				Function: &openai.ToolFunction{
					Name: "lookup",
				},
			},
		},
		ToolChoice: openai.ChatToolChoiceString("required"),
	}

	_, err := TranslateChatRequest(context.Background(), req, fakeCatalogRecord("model"))
	assertInvalidRequestError(t, err)
}

func TestTranslateChatRequestBuildsMixedTextAndImageBlocksForMultimodalModel(t *testing.T) {
	req := openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{{
			Role: "user",
			Content: openai.ChatMessageContent{
				Kind: openai.ChatMessageContentKindParts,
				Parts: []openai.ChatMessageContentPart{
					{Type: "text", Text: "describe this"},
					{Type: "image_url", ImageURL: map[string]any{"url": testPNGDataURL(t)}},
				},
			},
		}},
	}

	got, err := TranslateChatRequest(context.Background(), req, fakeMultimodalCatalogRecord("model"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" {
		t.Fatalf("expected translated user message, got %#v", got.Messages)
	}
	if len(got.Messages[0].Content) != 2 {
		t.Fatalf("expected mixed text and image blocks, got %#v", got.Messages[0].Content)
	}
	if got.Messages[0].Content[0].Text != "describe this" {
		t.Fatalf("expected leading text block, got %#v", got.Messages[0].Content[0])
	}
	if got.Messages[0].Content[1].Image == nil {
		t.Fatalf("expected image block, got %#v", got.Messages[0].Content[1])
	}
	if got.Messages[0].Content[1].Image.Format != "png" {
		t.Fatalf("expected png image format, got %#v", got.Messages[0].Content[1].Image)
	}
	if !bytes.Equal(got.Messages[0].Content[1].Image.Bytes, testPNGBytes(t)) {
		t.Fatalf("expected decoded image bytes, got %#v", got.Messages[0].Content[1].Image.Bytes)
	}
}

func TestTranslateChatRequestRejectsImageForTextOnlyModel(t *testing.T) {
	req := openai.ChatCompletionRequest{
		Model: "text-only-model",
		Messages: []openai.ChatMessage{{
			Role: "user",
			Content: openai.ChatMessageContent{
				Kind: openai.ChatMessageContentKindParts,
				Parts: []openai.ChatMessageContentPart{
					{Type: "image_url", ImageURL: map[string]any{"url": testPNGDataURL(t)}},
				},
			},
		}},
	}

	_, err := TranslateChatRequest(context.Background(), req, fakeCatalogRecord("text-only-model"))
	assertInvalidRequestError(t, err)
	if !strings.Contains(err.Error(), "multimodal message is not supported by this model") {
		t.Fatalf("expected multimodal rejection message, got %v", err)
	}
}

func TestTranslateChatRequestPropagatesContextCancellationDuringImageFetch(t *testing.T) {
	previousFetcher := defaultImageFetcher
	defaultImageFetcher = func(ctx context.Context, _ string) ([]byte, string, error) {
		return nil, "", ctx.Err()
	}
	t.Cleanup(func() {
		defaultImageFetcher = previousFetcher
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{{
			Role: "user",
			Content: openai.ChatMessageContent{
				Kind: openai.ChatMessageContentKindParts,
				Parts: []openai.ChatMessageContentPart{
					{Type: "image_url", ImageURL: map[string]any{"url": "https://example.com/cat.png"}},
				},
			},
		}},
	}

	_, err := TranslateChatRequest(ctx, req, fakeMultimodalCatalogRecord("model"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context cancellation to propagate, got %v", err)
	}
}

func TestTranslateChatResponseBuildsToolCalls(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "abc123",
		Output: []OutputBlock{{
			Type: OutputBlockTypeToolCall,
			ToolCall: &ToolCall{
				ID:        "call_1",
				Name:      "lookup",
				Arguments: "{\"q\":\"weather\"}",
			},
		}},
		StopReason: "tool_use",
	}

	got := TranslateChatResponse(resp, "model")
	if got.Choices[0].Message.ToolCalls[0].Function.Name != "lookup" {
		t.Fatalf("expected tool call output, got %#v", got)
	}
	if got.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("expected finish_reason to map to tool_calls, got %#v", got.Choices[0])
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
	if got.Choices[0].Message.Content.Text != "final answer" {
		t.Fatalf("expected assistant text in content, got %#v", got.Choices[0].Message)
	}
	if strings.Contains(got.Choices[0].Message.Content.Text, "<think>") {
		t.Fatalf("expected no synthetic think tags, got %#v", got.Choices[0].Message)
	}
}

func TestTranslateChatResponseMapsFinishReasons(t *testing.T) {
	cases := []struct {
		name       string
		stopReason string
		want       string
	}{
		{name: "end turn", stopReason: "end_turn", want: "stop"},
		{name: "stop sequence", stopReason: "stop_sequence", want: "stop"},
		{name: "max tokens", stopReason: "max_tokens", want: "length"},
		{name: "tool use", stopReason: "tool_use", want: "tool_calls"},
		{name: "guardrail", stopReason: "guardrail_intervened", want: "content_filter"},
		{name: "unknown", stopReason: "unknown", want: "stop"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := TranslateChatResponse(ConverseResponse{
				ResponseID: "resp",
				Output: []OutputBlock{
					{Type: OutputBlockTypeText, Text: "hello"},
				},
				StopReason: tc.stopReason,
			}, "model")
			if got.Choices[0].FinishReason != tc.want {
				t.Fatalf("expected finish reason %q, got %#v", tc.want, got.Choices[0])
			}
		})
	}
}

func fakeCatalogRecord(id string) ModelRecord {
	return ModelRecord{
		ID:              id,
		ModelKind:       modelKindFoundationModel,
		InputModalities: []string{"TEXT"},
	}
}

func fakeMultimodalCatalogRecord(id string) ModelRecord {
	return ModelRecord{
		ID:              id,
		ModelKind:       modelKindFoundationModel,
		InputModalities: []string{"TEXT", "IMAGE"},
	}
}
