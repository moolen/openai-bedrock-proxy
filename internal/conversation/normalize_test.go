package conversation

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

func TestNormalizeRequestBuildsSystemAndMessages(t *testing.T) {
	req := openai.ResponsesRequest{
		Model:        "model",
		Instructions: "follow",
		Input: []map[string]any{
			{"role": "system", "content": "sys"},
			{"role": "developer", "content": []map[string]any{{"type": "input_text", "text": "dev"}}},
			{"role": "user", "content": "hi"},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.System) != 3 {
		t.Fatalf("expected 3 system entries, got %d", len(normalized.System))
	}
	if normalized.System[0] != "sys" || normalized.System[1] != "dev" || normalized.System[2] != "follow" {
		t.Fatalf("unexpected system ordering: %#v", normalized.System)
	}

	if len(normalized.Messages) != 1 {
		t.Fatalf("expected 1 persisted message, got %d", len(normalized.Messages))
	}
	if normalized.Messages[0].Role != "user" {
		t.Fatalf("unexpected persisted message: %#v", normalized.Messages[0])
	}
	if len(normalized.Messages[0].Blocks) != 1 {
		t.Fatalf("expected 1 user block, got %#v", normalized.Messages[0].Blocks)
	}
	if normalized.Messages[0].Blocks[0].Type != BlockTypeText || normalized.Messages[0].Blocks[0].Text != "hi" {
		t.Fatalf("unexpected user block: %#v", normalized.Messages[0].Blocks[0])
	}
}

func TestNormalizeRequestPersistsAssistantTurnsButNotSystemTurns(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{"role": "system", "content": "sys"},
			{"role": "assistant", "content": "assistant reply"},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.System) != 1 || normalized.System[0] != "sys" {
		t.Fatalf("unexpected system content: %#v", normalized.System)
	}
	if len(normalized.Messages) != 1 {
		t.Fatalf("expected 1 persisted message, got %d", len(normalized.Messages))
	}
	if normalized.Messages[0].Role != "assistant" {
		t.Fatalf("unexpected persisted message: %#v", normalized.Messages[0])
	}
	if len(normalized.Messages[0].Blocks) != 1 {
		t.Fatalf("expected 1 assistant block, got %#v", normalized.Messages[0].Blocks)
	}
	if normalized.Messages[0].Blocks[0].Type != BlockTypeText || normalized.Messages[0].Blocks[0].Text != "assistant reply" {
		t.Fatalf("unexpected assistant block: %#v", normalized.Messages[0].Blocks[0])
	}
}

func TestNormalizeRequestPreservesTextBlocksWithoutSeparators(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{"role": "user", "content": []map[string]any{
				{"type": "input_text", "text": "hello"},
				{"type": "input_text", "text": "world"},
			}},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.Messages) != 1 {
		t.Fatalf("expected 1 persisted message, got %d", len(normalized.Messages))
	}
	if len(normalized.Messages[0].Blocks) != 2 {
		t.Fatalf("expected 2 text blocks, got %#v", normalized.Messages[0].Blocks)
	}
	if normalized.Messages[0].Blocks[0].Text != "hello" || normalized.Messages[0].Blocks[1].Text != "world" {
		t.Fatalf("unexpected text block ordering: %#v", normalized.Messages[0].Blocks)
	}
}

func TestNormalizeRequestNormalizesFunctionToolsAndChoice(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []openai.Tool{
			{
				Type: "function",
				Function: &openai.ToolFunction{
					Name:        "lookup",
					Description: "Look up a value",
					Parameters:  map[string]any{"type": "object"},
				},
			},
		},
		ToolChoice: &openai.ToolChoice{
			Mode: "object",
			Type: "function",
			Function: &openai.ToolChoiceFunction{
				Name: "lookup",
			},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.Tools) != 1 {
		t.Fatalf("expected 1 normalized tool, got %#v", normalized.Tools)
	}
	if normalized.Tools[0].Type != "function" || normalized.Tools[0].Name != "lookup" {
		t.Fatalf("unexpected function tool normalization: %#v", normalized.Tools[0])
	}
	if normalized.Tools[0].Description != "Look up a value" {
		t.Fatalf("unexpected function tool description: %#v", normalized.Tools[0])
	}
	if normalized.ToolChoice.Type != "function" || normalized.ToolChoice.Name != "lookup" {
		t.Fatalf("unexpected tool choice: %#v", normalized.ToolChoice)
	}
}

func TestNormalizeRequestNormalizesBuiltInToolsIntoSyntheticDefinitions(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []openai.Tool{
			{
				Type: "web_search_preview",
				Config: map[string]json.RawMessage{
					"user_location": json.RawMessage(`{"type":"approximate","country":"DE"}`),
				},
			},
		},
		ToolChoice: &openai.ToolChoice{
			Mode: "object",
			Type: "web_search_preview",
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.Tools) != 1 {
		t.Fatalf("expected 1 normalized tool, got %#v", normalized.Tools)
	}
	if normalized.Tools[0].Type != "web_search_preview" || normalized.Tools[0].Name != syntheticBuiltInToolName("web_search_preview") {
		t.Fatalf("unexpected built-in tool normalization: %#v", normalized.Tools[0])
	}
	if !normalized.Tools[0].BuiltIn {
		t.Fatalf("expected built-in tool marker, got %#v", normalized.Tools[0])
	}
	if string(normalized.Tools[0].Config["user_location"]) != `{"type":"approximate","country":"DE"}` {
		t.Fatalf("unexpected built-in config: %#v", normalized.Tools[0].Config)
	}
	if normalized.ToolChoice.Type != "web_search_preview" || normalized.ToolChoice.Name != syntheticBuiltInToolName("web_search_preview") {
		t.Fatalf("unexpected built-in tool choice: %#v", normalized.ToolChoice)
	}
}

func TestNormalizeRequestNormalizesToolResultItemsIntoUserBlocks(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{
				"role": "user",
				"content": []any{
					map[string]any{
						"type":    "function_call_output",
						"call_id": "call_123",
						"output":  map[string]any{"ok": true},
					},
				},
			},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.Messages) != 1 {
		t.Fatalf("expected 1 persisted message, got %d", len(normalized.Messages))
	}
	if len(normalized.Messages[0].Blocks) != 1 {
		t.Fatalf("expected 1 tool-result block, got %#v", normalized.Messages[0].Blocks)
	}
	block := normalized.Messages[0].Blocks[0]
	if block.Type != BlockTypeToolResult || block.ToolResult == nil {
		t.Fatalf("unexpected tool-result block: %#v", block)
	}
	if block.ToolResult.CallID != "call_123" {
		t.Fatalf("unexpected tool-result call id: %#v", block.ToolResult)
	}
	output, ok := block.ToolResult.Output.(map[string]any)
	if !ok || output["ok"] != true {
		t.Fatalf("unexpected tool-result output: %#v", block.ToolResult.Output)
	}
}

func TestNormalizeRequestPreservesMixedTextAndToolResultOrdering(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Use the result"},
					map[string]any{
						"type":    "function_call_output",
						"call_id": "call_123",
						"output":  map[string]any{"ok": true},
					},
					map[string]any{"type": "input_text", "text": " now"},
				},
			},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.Messages) != 1 || len(normalized.Messages[0].Blocks) != 3 {
		t.Fatalf("expected 1 message with 3 blocks, got %#v", normalized.Messages)
	}
	if normalized.Messages[0].Blocks[0].Type != BlockTypeText || normalized.Messages[0].Blocks[0].Text != "Use the result" {
		t.Fatalf("unexpected first block: %#v", normalized.Messages[0].Blocks[0])
	}
	if normalized.Messages[0].Blocks[1].Type != BlockTypeToolResult || normalized.Messages[0].Blocks[1].ToolResult == nil {
		t.Fatalf("unexpected second block: %#v", normalized.Messages[0].Blocks[1])
	}
	if normalized.Messages[0].Blocks[2].Type != BlockTypeText || normalized.Messages[0].Blocks[2].Text != " now" {
		t.Fatalf("unexpected third block: %#v", normalized.Messages[0].Blocks[2])
	}
}

func TestMergeUsesCurrentSystemAndAppendsMessages(t *testing.T) {
	base := Request{
		System: []string{"old"},
		Messages: []Message{
			{
				Role: "user",
				Blocks: []Block{
					{Type: BlockTypeText, Text: "previous"},
				},
			},
		},
		Tools: []ToolDefinition{{Type: "function", Name: "old_tool"}},
	}
	current := Request{
		System: []string{"current"},
		Messages: []Message{
			{
				Role: "assistant",
				Blocks: []Block{
					{Type: BlockTypeText, Text: "now"},
				},
			},
		},
		Tools:      []ToolDefinition{{Type: "function", Name: "new_tool"}},
		ToolChoice: ToolChoice{Type: "function", Name: "new_tool"},
	}

	merged := Merge(base, current)

	if len(merged.System) != 1 || merged.System[0] != "current" {
		t.Fatalf("unexpected merged system: %#v", merged.System)
	}
	if len(merged.Messages) != 2 {
		t.Fatalf("expected 2 merged messages, got %d", len(merged.Messages))
	}
	if merged.Messages[0].Blocks[0].Text != "previous" || merged.Messages[1].Blocks[0].Text != "now" {
		t.Fatalf("unexpected merged messages: %#v", merged.Messages)
	}
	if len(merged.Tools) != 1 || merged.Tools[0].Name != "new_tool" {
		t.Fatalf("expected current tools to win, got %#v", merged.Tools)
	}
	if merged.ToolChoice.Type != "function" || merged.ToolChoice.Name != "new_tool" {
		t.Fatalf("expected current tool choice to win, got %#v", merged.ToolChoice)
	}
}

func TestAppendAssistantReplyAppendsToMessages(t *testing.T) {
	req := Request{
		System: []string{"sys"},
		Messages: []Message{
			{
				Role: "user",
				Blocks: []Block{
					{Type: BlockTypeText, Text: "hi"},
				},
			},
		},
	}

	updated := AppendAssistantReply(req, []Block{
		{Type: BlockTypeText, Text: "ok"},
		{Type: BlockTypeToolCall, ToolCall: &ToolCall{ID: "call_123", Name: "lookup", Arguments: `{"q":"x"}`}},
	})

	if len(updated.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(updated.Messages))
	}
	if updated.Messages[1].Role != "assistant" {
		t.Fatalf("unexpected assistant append: %#v", updated.Messages[1])
	}
	if len(updated.Messages[1].Blocks) != 2 {
		t.Fatalf("unexpected assistant blocks: %#v", updated.Messages[1].Blocks)
	}
	if updated.Messages[1].Blocks[0].Type != BlockTypeText || updated.Messages[1].Blocks[0].Text != "ok" {
		t.Fatalf("unexpected first assistant block: %#v", updated.Messages[1].Blocks[0])
	}
	if updated.Messages[1].Blocks[1].Type != BlockTypeToolCall || updated.Messages[1].Blocks[1].ToolCall == nil {
		t.Fatalf("unexpected second assistant block: %#v", updated.Messages[1].Blocks[1])
	}
	if len(updated.System) != 1 || updated.System[0] != "sys" {
		t.Fatalf("unexpected system after append: %#v", updated.System)
	}
}

func TestNormalizeRequestRejectsMalformedInputs(t *testing.T) {
	cases := []struct {
		name  string
		input any
	}{
		{
			name:  "missing role",
			input: map[string]any{"content": "hi"},
		},
		{
			name:  "wrong role type",
			input: map[string]any{"role": 7, "content": "hi"},
		},
		{
			name:  "wrong content type",
			input: map[string]any{"role": "user", "content": 7},
		},
		{
			name:  "non-object item",
			input: []any{"oops"},
		},
		{
			name: "wrong block type",
			input: map[string]any{"role": "user", "content": []map[string]any{
				{"type": "output_text", "text": "hi"},
			}},
		},
		{
			name: "missing text",
			input: map[string]any{"role": "user", "content": []map[string]any{
				{"type": "input_text"},
			}},
		},
		{
			name: "missing tool result call id",
			input: map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "function_call_output", "output": "ok"},
			}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := openai.ResponsesRequest{
				Model: "model",
				Input: tc.input,
			}

			_, err := NormalizeRequest(req)
			if err == nil {
				t.Fatalf("expected error for malformed input")
			}
			var invalid openai.InvalidRequestError
			if !errors.As(err, &invalid) {
				t.Fatalf("expected invalid request error, got %T", err)
			}
		})
	}
}

func TestNormalizeRequestRejectsStructuredInputWithoutPersistedMessages(t *testing.T) {
	cases := []struct {
		name  string
		input any
	}{
		{
			name:  "system only",
			input: []map[string]any{{"role": "system", "content": "sys"}},
		},
		{
			name:  "developer only",
			input: []map[string]any{{"role": "developer", "content": "dev"}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := openai.ResponsesRequest{
				Model: "model",
				Input: tc.input,
			}

			_, err := NormalizeRequest(req)
			if err == nil {
				t.Fatalf("expected error for structured input without persisted messages")
			}

			var invalid openai.InvalidRequestError
			if !errors.As(err, &invalid) {
				t.Fatalf("expected invalid request error, got %T", err)
			}
		})
	}
}
