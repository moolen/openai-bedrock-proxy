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

func TestNormalizeRequestPreservesCodexBuiltInMetadata(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []openai.Tool{
			{
				Type:        "tool_search",
				Description: "Search app tools",
				Config: map[string]json.RawMessage{
					"execution":  json.RawMessage(`"client"`),
					"parameters": json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
				},
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
	if normalized.Tools[0].Description != "Search app tools" {
		t.Fatalf("expected built-in description to be preserved, got %#v", normalized.Tools[0])
	}
	if string(normalized.Tools[0].Config["execution"]) != `"client"` {
		t.Fatalf("expected built-in execution metadata to be preserved, got %#v", normalized.Tools[0].Config)
	}
	if _, ok := normalized.Tools[0].Config["parameters"]; !ok {
		t.Fatalf("expected built-in parameters metadata to be preserved, got %#v", normalized.Tools[0].Config)
	}
}

func TestNormalizeRequestNormalizesCodexCustomToolsIntoSyntheticDefinitions(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []openai.Tool{
			{
				Type:        "custom",
				Name:        "apply_patch",
				Description: "Apply a patch",
				Config: map[string]json.RawMessage{
					"format": json.RawMessage(`{"type":"grammar","syntax":"lark","definition":"start: /[\\s\\S]+/"}`),
				},
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
	if normalized.Tools[0].Type != "custom" || normalized.Tools[0].Name != syntheticCustomToolName("apply_patch") {
		t.Fatalf("unexpected custom tool normalization: %#v", normalized.Tools[0])
	}
	if normalized.Tools[0].Description != "Apply a patch" {
		t.Fatalf("expected custom description to be preserved, got %#v", normalized.Tools[0])
	}
	if !normalized.Tools[0].BuiltIn {
		t.Fatalf("expected custom tool to use synthetic built-in translation path, got %#v", normalized.Tools[0])
	}
	if string(normalized.Tools[0].Config["format"]) != `{"type":"grammar","syntax":"lark","definition":"start: /[\\s\\S]+/"}` {
		t.Fatalf("unexpected custom tool config: %#v", normalized.Tools[0].Config)
	}
}

func TestNormalizeRequestLeavesToolChoiceEmptyForNone(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: "hi",
		ToolChoice: &openai.ToolChoice{
			Mode: "string",
			Type: "none",
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if normalized.ToolChoice != (ToolChoice{}) {
		t.Fatalf("expected empty tool choice for none, got %#v", normalized.ToolChoice)
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

func TestNormalizeRequestNormalizesCustomToolCallOutputsIntoUserBlocks(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{
				"role": "user",
				"content": []any{
					map[string]any{
						"type":    "custom_tool_call_output",
						"call_id": "call_custom",
						"output":  "ok",
					},
				},
			},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}
	if len(normalized.Messages) != 1 || len(normalized.Messages[0].Blocks) != 1 {
		t.Fatalf("expected one normalized tool result block, got %#v", normalized.Messages)
	}
	if normalized.Messages[0].Blocks[0].Type != BlockTypeToolResult {
		t.Fatalf("expected tool result block, got %#v", normalized.Messages[0].Blocks[0])
	}
	if normalized.Messages[0].Blocks[0].ToolResult.CallID != "call_custom" {
		t.Fatalf("expected custom call id to map, got %#v", normalized.Messages[0].Blocks[0].ToolResult)
	}
	if normalized.Messages[0].Blocks[0].ToolResult.Output != "ok" {
		t.Fatalf("expected custom output to map, got %#v", normalized.Messages[0].Blocks[0].ToolResult)
	}
}

func TestNormalizeRequestNormalizesToolSearchOutputsIntoUserBlocks(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{
				"role": "user",
				"content": []any{
					map[string]any{
						"type":      "tool_search_output",
						"call_id":   "search_1",
						"status":    "completed",
						"execution": "client",
						"tools": []any{
							map[string]any{"name": "calendar_create"},
						},
					},
				},
			},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}
	result := normalized.Messages[0].Blocks[0].ToolResult
	if result == nil || result.CallID != "search_1" {
		t.Fatalf("expected tool_search output to normalize into tool result, got %#v", normalized.Messages)
	}
	output, ok := result.Output.(map[string]any)
	if !ok || output["type"] != "tool_search_output" || output["status"] != "completed" || output["execution"] != "client" {
		t.Fatalf("unexpected normalized tool_search output: %#v", result.Output)
	}
}

func TestNormalizeRequestNormalizesCodexAssistantToolCalls(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type":      "tool_search_call",
						"call_id":   "search_1",
						"execution": "client",
						"arguments": map[string]any{"query": "calendar"},
					},
					map[string]any{
						"type":    "local_shell_call",
						"call_id": "shell_1",
						"action": map[string]any{
							"type":    "exec",
							"command": []any{"pwd"},
						},
					},
					map[string]any{
						"type":           "image_generation_call",
						"id":             "img_1",
						"status":         "completed",
						"revised_prompt": "A blue square",
						"result":         "Zm9v",
					},
				},
			},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}
	if len(normalized.Messages) != 1 || len(normalized.Messages[0].Blocks) != 3 {
		t.Fatalf("expected normalized assistant blocks, got %#v", normalized.Messages)
	}
	if normalized.Messages[0].Blocks[0].ToolCall == nil || normalized.Messages[0].Blocks[0].ToolCall.Name != syntheticBuiltInToolName("tool_search") {
		t.Fatalf("expected tool_search call to normalize, got %#v", normalized.Messages[0].Blocks[0])
	}
	if normalized.Messages[0].Blocks[1].ToolCall == nil || normalized.Messages[0].Blocks[1].ToolCall.Name != syntheticBuiltInToolName("local_shell") {
		t.Fatalf("expected local_shell call to normalize, got %#v", normalized.Messages[0].Blocks[1])
	}
	if normalized.Messages[0].Blocks[2].Type != BlockTypeText || normalized.Messages[0].Blocks[2].Text == "" {
		t.Fatalf("expected image generation call to normalize into text context, got %#v", normalized.Messages[0].Blocks[2])
	}
}

func TestNormalizeRequestNormalizesMixedTopLevelResponseInputItems(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []any{
			map[string]any{"role": "developer", "content": "be brief"},
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "hello"},
				},
			},
			map[string]any{
				"type":      "function_call",
				"call_id":   "call_1",
				"name":      "lookup",
				"arguments": `{"q":"weather"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_1",
				"output":  "sunny",
			},
			map[string]any{
				"type":    "mcp_tool_call_output",
				"call_id": "call_mcp_1",
				"output": map[string]any{
					"content": []any{
						map[string]any{"type": "text", "text": "done"},
					},
					"isError": false,
				},
			},
			map[string]any{
				"type":   "image_generation_call",
				"id":     "img_1",
				"status": "completed",
				"result": "Zm9v",
			},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.System) != 1 || normalized.System[0] != "be brief" {
		t.Fatalf("expected developer input to map into system instructions, got %#v", normalized.System)
	}
	if len(normalized.Messages) != 4 {
		t.Fatalf("expected 4 persisted messages, got %#v", normalized.Messages)
	}
	if normalized.Messages[0].Role != "user" || normalized.Messages[0].Blocks[0].Text != "hello" {
		t.Fatalf("unexpected first normalized message: %#v", normalized.Messages[0])
	}
	if normalized.Messages[1].Role != "assistant" || normalized.Messages[1].Blocks[0].ToolCall == nil {
		t.Fatalf("expected assistant tool call in second message, got %#v", normalized.Messages[1])
	}
	if normalized.Messages[1].Blocks[0].ToolCall.Name != "lookup" {
		t.Fatalf("expected function call name to be preserved, got %#v", normalized.Messages[1].Blocks[0].ToolCall)
	}
	if normalized.Messages[2].Role != "user" || len(normalized.Messages[2].Blocks) != 2 {
		t.Fatalf("expected grouped user outputs in third message, got %#v", normalized.Messages[2])
	}
	if normalized.Messages[2].Blocks[0].ToolResult == nil {
		t.Fatalf("expected function output as first grouped block, got %#v", normalized.Messages[2].Blocks[0])
	}
	if normalized.Messages[2].Blocks[0].ToolResult.CallID != "call_1" || normalized.Messages[2].Blocks[0].ToolResult.Output != "sunny" {
		t.Fatalf("unexpected function output normalization: %#v", normalized.Messages[2].Blocks[0].ToolResult)
	}
	if normalized.Messages[2].Blocks[1].ToolResult == nil {
		t.Fatalf("expected MCP output as second grouped block, got %#v", normalized.Messages[2].Blocks[1])
	}
	mcpOutput, ok := normalized.Messages[2].Blocks[1].ToolResult.Output.(map[string]any)
	if !ok || mcpOutput["isError"] != false {
		t.Fatalf("unexpected MCP output normalization: %#v", normalized.Messages[2].Blocks[1].ToolResult.Output)
	}
	if normalized.Messages[3].Role != "assistant" || normalized.Messages[3].Blocks[0].Type != BlockTypeText || normalized.Messages[3].Blocks[0].Text == "" {
		t.Fatalf("expected image generation replay in fourth message, got %#v", normalized.Messages[3])
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

func TestMergeClonesMessagesAndTools(t *testing.T) {
	base := Request{
		System: []string{"old"},
		Messages: []Message{
			{
				Role: "user",
				Blocks: []Block{
					{
						Type: BlockTypeToolResult,
						ToolResult: &ToolResult{
							CallID: "call_base",
							Output: []byte("base"),
						},
					},
				},
			},
		},
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
		Tools: []ToolDefinition{
			{
				Type:       "function",
				Name:       "lookup",
				Parameters: map[string]any{"type": "object", "properties": map[string]any{"q": "string"}},
			},
			{
				Type:    "web_search_preview",
				Name:    syntheticBuiltInToolName("web_search_preview"),
				BuiltIn: true,
				Config: map[string]json.RawMessage{
					"user_location": json.RawMessage(`{"country":"DE"}`),
				},
			},
		},
		ToolChoice: ToolChoice{Type: "function", Name: "lookup"},
	}

	merged := Merge(base, current)

	base.Messages[0].Blocks[0].ToolResult.Output.([]byte)[0] = 'X'
	current.Messages[0].Blocks[0].Text = "changed"
	current.Tools[0].Parameters["properties"].(map[string]any)["q"] = "number"
	current.Tools[1].Config["user_location"][2] = 'X'

	if string(merged.Messages[0].Blocks[0].ToolResult.Output.([]byte)) != "base" {
		t.Fatalf("expected base tool result payload to be cloned, got %#v", merged.Messages[0].Blocks[0].ToolResult.Output)
	}
	if merged.Messages[1].Blocks[0].Text != "now" {
		t.Fatalf("expected current message blocks to be cloned, got %#v", merged.Messages[1].Blocks)
	}
	if merged.Tools[0].Parameters["properties"].(map[string]any)["q"] != "string" {
		t.Fatalf("expected tool parameters to be cloned, got %#v", merged.Tools[0].Parameters)
	}
	if string(merged.Tools[1].Config["user_location"]) != `{"country":"DE"}` {
		t.Fatalf("expected built-in config to be cloned, got %#v", merged.Tools[1].Config)
	}

	merged.Messages[1].Blocks[0].Text = "mutated"
	merged.Tools[0].Parameters["type"] = "array"
	merged.Tools[1].Config["user_location"][2] = 'Y'

	if current.Messages[0].Blocks[0].Text != "changed" {
		t.Fatalf("expected current messages not to alias merged messages")
	}
	if current.Tools[0].Parameters["type"] != "object" {
		t.Fatalf("expected current tool parameters not to alias merged tools, got %#v", current.Tools[0].Parameters)
	}
	if string(current.Tools[1].Config["user_location"]) != `{"Xountry":"DE"}` {
		t.Fatalf("expected current config not to alias merged config, got %#v", current.Tools[1].Config)
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

func TestAppendAssistantReplyClonesMessagesToolsAndBlocks(t *testing.T) {
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
		Tools: []ToolDefinition{
			{
				Type:       "function",
				Name:       "lookup",
				Parameters: map[string]any{"type": "object"},
			},
			{
				Type:    "web_search_preview",
				Name:    syntheticBuiltInToolName("web_search_preview"),
				BuiltIn: true,
				Config: map[string]json.RawMessage{
					"user_location": json.RawMessage(`{"country":"DE"}`),
				},
			},
		},
	}
	assistantBlocks := []Block{
		{Type: BlockTypeText, Text: "ok"},
		{
			Type: BlockTypeToolResult,
			ToolResult: &ToolResult{
				CallID: "call_1",
				Output: []byte("payload"),
			},
		},
	}

	updated := AppendAssistantReply(req, assistantBlocks)

	req.Messages[0].Blocks[0].Text = "changed"
	req.Tools[0].Parameters["type"] = "array"
	req.Tools[1].Config["user_location"][2] = 'X'
	assistantBlocks[0].Text = "mutated"
	assistantBlocks[1].ToolResult.Output.([]byte)[0] = 'X'

	if updated.Messages[0].Blocks[0].Text != "hi" {
		t.Fatalf("expected existing request messages to be cloned, got %#v", updated.Messages[0].Blocks)
	}
	if updated.Messages[1].Blocks[0].Text != "ok" {
		t.Fatalf("expected appended blocks to be cloned, got %#v", updated.Messages[1].Blocks)
	}
	if string(updated.Messages[1].Blocks[1].ToolResult.Output.([]byte)) != "payload" {
		t.Fatalf("expected appended tool result payload to be cloned, got %#v", updated.Messages[1].Blocks[1].ToolResult.Output)
	}
	if updated.Tools[0].Parameters["type"] != "object" {
		t.Fatalf("expected tool parameters to be cloned, got %#v", updated.Tools[0].Parameters)
	}
	if string(updated.Tools[1].Config["user_location"]) != `{"country":"DE"}` {
		t.Fatalf("expected built-in config to be cloned, got %#v", updated.Tools[1].Config)
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
