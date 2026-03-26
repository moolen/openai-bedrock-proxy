package bedrock

import (
	"encoding/json"
	"errors"
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
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "hello"},
				},
			},
			{
				Role: "assistant",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "hi"},
				},
			},
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

func TestTranslateConversationMapsToolConfigAndAutoChoice(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "Find it"},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type:        "function",
				Name:        "lookup",
				Description: "Look up a value",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"q": map[string]any{"type": "string"},
					},
					"required": []any{"q"},
				},
			},
			{
				Type:    "web_search_preview",
				Name:    "__builtin_web_search_preview",
				BuiltIn: true,
				Config: map[string]json.RawMessage{
					"user_location": json.RawMessage(`{"type":"approximate","country":"DE"}`),
				},
			},
		},
		ToolChoice: conversation.ToolChoice{Type: "auto"},
	}

	got, err := TranslateConversation("model", request, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ToolConfig == nil {
		t.Fatal("expected tool config to be present")
	}
	if len(got.ToolConfig.Tools) != 2 {
		t.Fatalf("expected 2 translated tools, got %#v", got.ToolConfig.Tools)
	}
	if got.ToolConfig.Tools[0].Name != "lookup" {
		t.Fatalf("expected function tool name to map, got %#v", got.ToolConfig.Tools[0])
	}
	if got.ToolConfig.Tools[0].Description != "Look up a value" {
		t.Fatalf("expected function tool description to map, got %#v", got.ToolConfig.Tools[0])
	}
	if got.ToolConfig.Tools[0].InputSchema["type"] != "object" {
		t.Fatalf("expected function tool schema to map, got %#v", got.ToolConfig.Tools[0].InputSchema)
	}

	builtIn := got.ToolConfig.Tools[1]
	if builtIn.Name != "__builtin_web_search_preview" {
		t.Fatalf("expected synthetic built-in name, got %#v", builtIn)
	}
	if builtIn.InputSchema["x-openai-tool-type"] != "web_search_preview" {
		t.Fatalf("expected built-in schema metadata, got %#v", builtIn.InputSchema)
	}
	properties, ok := builtIn.InputSchema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected built-in schema properties, got %#v", builtIn.InputSchema)
	}
	if _, ok := properties["query"]; !ok {
		t.Fatalf("expected built-in schema to expose query input, got %#v", properties)
	}
	rawConfig, ok := builtIn.InputSchema["x-openai-config"].(map[string]any)
	if !ok {
		t.Fatalf("expected built-in schema config metadata, got %#v", builtIn.InputSchema)
	}
	location, ok := rawConfig["user_location"].(map[string]any)
	if !ok || location["country"] != "DE" {
		t.Fatalf("expected built-in config to survive translation, got %#v", rawConfig)
	}
	if got.ToolConfig.ToolChoice == nil || !got.ToolConfig.ToolChoice.Auto {
		t.Fatalf("expected auto tool choice to map, got %#v", got.ToolConfig.ToolChoice)
	}
}

func TestTranslateConversationMapsCodexCustomTools(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "Edit it"},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type:        "custom",
				Name:        "__custom_apply_patch",
				Description: "Apply a patch",
				BuiltIn:     true,
				Config: map[string]json.RawMessage{
					"format": json.RawMessage(`{"type":"grammar","syntax":"lark","definition":"start: /[\\s\\S]+/"}`),
				},
			},
		},
	}

	got, err := TranslateConversation("model", request, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ToolConfig == nil || len(got.ToolConfig.Tools) != 1 {
		t.Fatalf("expected translated custom tool config, got %#v", got.ToolConfig)
	}
	customTool := got.ToolConfig.Tools[0]
	if customTool.Name != "__custom_apply_patch" {
		t.Fatalf("expected custom tool name to pass through, got %#v", customTool)
	}
	if customTool.InputSchema["x-openai-tool-type"] != "custom" {
		t.Fatalf("expected custom schema metadata, got %#v", customTool.InputSchema)
	}
	properties, ok := customTool.InputSchema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected custom schema properties, got %#v", customTool.InputSchema)
	}
	input, ok := properties["input"].(map[string]any)
	if !ok || input["type"] != "string" {
		t.Fatalf("expected custom input schema to expose raw string input, got %#v", properties)
	}
	rawConfig, ok := customTool.InputSchema["x-openai-config"].(map[string]any)
	if !ok {
		t.Fatalf("expected custom schema config metadata, got %#v", customTool.InputSchema)
	}
	format, ok := rawConfig["format"].(map[string]any)
	if !ok || format["syntax"] != "lark" {
		t.Fatalf("expected custom format metadata to survive translation, got %#v", rawConfig)
	}
}

func TestTranslateConversationMapsWebSearchTools(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "Search"},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type:    "web_search",
				Name:    "__builtin_web_search",
				BuiltIn: true,
				Config: map[string]json.RawMessage{
					"external_web_access": json.RawMessage(`true`),
				},
			},
		},
	}

	got, err := TranslateConversation("model", request, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ToolConfig == nil || len(got.ToolConfig.Tools) != 1 {
		t.Fatalf("expected translated web_search tool config, got %#v", got.ToolConfig)
	}
	webSearch := got.ToolConfig.Tools[0]
	if webSearch.InputSchema["x-openai-tool-type"] != "web_search" {
		t.Fatalf("expected web_search schema metadata, got %#v", webSearch.InputSchema)
	}
	properties, ok := webSearch.InputSchema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected web_search schema properties, got %#v", webSearch.InputSchema)
	}
	if _, ok := properties["query"]; !ok {
		t.Fatalf("expected web_search schema to expose query input, got %#v", properties)
	}
}

func TestTranslateConversationMapsToolSearchTools(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "Search tools"},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type:        "tool_search",
				Name:        "__builtin_tool_search",
				Description: "Search tools",
				BuiltIn:     true,
				Config: map[string]json.RawMessage{
					"execution":  json.RawMessage(`"client"`),
					"parameters": json.RawMessage(`{"type":"object"}`),
				},
			},
		},
	}

	got, err := TranslateConversation("model", request, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ToolConfig == nil || len(got.ToolConfig.Tools) != 1 {
		t.Fatalf("expected translated tool_search config, got %#v", got.ToolConfig)
	}
	toolSearch := got.ToolConfig.Tools[0]
	if toolSearch.InputSchema["x-openai-tool-type"] != "tool_search" {
		t.Fatalf("expected tool_search schema metadata, got %#v", toolSearch.InputSchema)
	}
	properties, ok := toolSearch.InputSchema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected tool_search schema properties, got %#v", toolSearch.InputSchema)
	}
	if _, ok := properties["query"]; !ok {
		t.Fatalf("expected tool_search schema to expose query input, got %#v", properties)
	}
	if _, ok := properties["limit"]; !ok {
		t.Fatalf("expected tool_search schema to expose limit input, got %#v", properties)
	}
}

func TestTranslateConversationMapsLocalShellTools(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "Run pwd"},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type:    "local_shell",
				Name:    "__builtin_local_shell",
				BuiltIn: true,
			},
		},
	}

	got, err := TranslateConversation("model", request, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ToolConfig == nil || len(got.ToolConfig.Tools) != 1 {
		t.Fatalf("expected translated local_shell config, got %#v", got.ToolConfig)
	}
	localShell := got.ToolConfig.Tools[0]
	if localShell.InputSchema["x-openai-tool-type"] != "local_shell" {
		t.Fatalf("expected local_shell schema metadata, got %#v", localShell.InputSchema)
	}
	properties, ok := localShell.InputSchema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected local_shell schema properties, got %#v", localShell.InputSchema)
	}
	if _, ok := properties["type"]; !ok {
		t.Fatalf("expected local_shell schema to expose action type, got %#v", properties)
	}
	if _, ok := properties["command"]; !ok {
		t.Fatalf("expected local_shell schema to expose command input, got %#v", properties)
	}
}

func TestTranslateConversationMapsToolUseAndToolResultBlocks(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "assistant",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "Checking"},
					{
						Type: conversation.BlockTypeToolCall,
						ToolCall: &conversation.ToolCall{
							ID:        "call_123",
							Name:      "lookup",
							Arguments: `{"q":"x"}`,
						},
					},
				},
			},
			{
				Role: "user",
				Blocks: []conversation.Block{
					{
						Type: conversation.BlockTypeToolResult,
						ToolResult: &conversation.ToolResult{
							CallID: "call_123",
							Output: map[string]any{"answer": "ok"},
						},
					},
					{Type: conversation.BlockTypeText, Text: "continue"},
				},
			},
		},
	}

	got, err := TranslateConversation("model", request, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %#v", got.Messages)
	}

	assistant := got.Messages[0]
	if len(assistant.Content) != 2 {
		t.Fatalf("expected assistant content ordering to be preserved, got %#v", assistant.Content)
	}
	if assistant.Content[0].Text != "Checking" {
		t.Fatalf("expected assistant text block first, got %#v", assistant.Content[0])
	}
	if assistant.Content[1].ToolUse == nil {
		t.Fatalf("expected assistant tool use block, got %#v", assistant.Content[1])
	}
	if assistant.Content[1].ToolUse.ToolUseID != "call_123" || assistant.Content[1].ToolUse.Name != "lookup" {
		t.Fatalf("unexpected translated tool use block: %#v", assistant.Content[1].ToolUse)
	}
	input, ok := assistant.Content[1].ToolUse.Input.(map[string]any)
	if !ok || input["q"] != "x" {
		t.Fatalf("expected tool use arguments to decode to JSON, got %#v", assistant.Content[1].ToolUse.Input)
	}

	user := got.Messages[1]
	if len(user.Content) != 2 {
		t.Fatalf("expected user content ordering to be preserved, got %#v", user.Content)
	}
	if user.Content[0].ToolResult == nil {
		t.Fatalf("expected user tool result block, got %#v", user.Content[0])
	}
	if user.Content[0].ToolResult.ToolUseID != "call_123" {
		t.Fatalf("unexpected translated tool result id: %#v", user.Content[0].ToolResult)
	}
	if len(user.Content[0].ToolResult.Content) != 1 {
		t.Fatalf("expected single translated tool result content block, got %#v", user.Content[0].ToolResult.Content)
	}
	jsonContent := user.Content[0].ToolResult.Content[0].JSON
	output, ok := jsonContent.(map[string]any)
	if !ok || output["answer"] != "ok" {
		t.Fatalf("expected structured tool result output, got %#v", jsonContent)
	}
	if user.Content[1].Text != "continue" {
		t.Fatalf("expected trailing user text block, got %#v", user.Content[1])
	}
}

func TestTranslateConversationRejectsUnsupportedNonBuiltInToolDefinitions(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "hello"},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type: "custom_tool",
				Name: "custom_tool",
			},
		},
	}

	_, err := TranslateConversation("model", request, nil, nil)
	assertInvalidRequestError(t, err)
}

func TestTranslateConversationRejectsEmptyToolNames(t *testing.T) {
	cases := []struct {
		name string
		tool conversation.ToolDefinition
	}{
		{
			name: "empty function tool name",
			tool: conversation.ToolDefinition{
				Type: "function",
				Name: "",
			},
		},
		{
			name: "empty synthetic built-in name",
			tool: conversation.ToolDefinition{
				Type:    "web_search_preview",
				Name:    "",
				BuiltIn: true,
			},
		},
		{
			name: "empty synthetic built-in type",
			tool: conversation.ToolDefinition{
				Type:    "",
				Name:    "__builtin_unknown",
				BuiltIn: true,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request := conversation.Request{
				Messages: []conversation.Message{
					{
						Role: "user",
						Blocks: []conversation.Block{
							{Type: conversation.BlockTypeText, Text: "hello"},
						},
					},
				},
				Tools: []conversation.ToolDefinition{tc.tool},
			}

			_, err := TranslateConversation("model", request, nil, nil)
			assertInvalidRequestError(t, err)
		})
	}
}

func TestTranslateConversationRejectsToolChoiceWithoutTools(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "hello"},
				},
			},
		},
		ToolChoice: conversation.ToolChoice{Type: "auto"},
	}

	_, err := TranslateConversation("model", request, nil, nil)
	assertInvalidRequestError(t, err)
}

func TestTranslateConversationAcceptsNameOnlySpecificToolChoice(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "hello"},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type: "function",
				Name: "lookup",
			},
		},
		ToolChoice: conversation.ToolChoice{
			Name: "lookup",
		},
	}

	got, err := TranslateConversation("model", request, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ToolConfig == nil || got.ToolConfig.ToolChoice == nil || got.ToolConfig.ToolChoice.Tool != "lookup" {
		t.Fatalf("expected name-only specific tool choice to map, got %#v", got.ToolConfig)
	}
}

func TestTranslateConversationRejectsMalformedSpecificToolChoiceShape(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "hello"},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type: "function",
				Name: "lookup",
			},
		},
		ToolChoice: conversation.ToolChoice{
			Type: "unexpected_type",
			Name: "lookup",
		},
	}

	_, err := TranslateConversation("model", request, nil, nil)
	assertInvalidRequestError(t, err)
}

func TestTranslateConversationRejectsToolChoiceTargetNotInToolSet(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "hello"},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type: "function",
				Name: "lookup",
			},
		},
		ToolChoice: conversation.ToolChoice{
			Type: "function",
			Name: "other_tool",
		},
	}

	_, err := TranslateConversation("model", request, nil, nil)
	assertInvalidRequestError(t, err)
}

func TestTranslateConversationRejectsInvalidToolCallPayloads(t *testing.T) {
	cases := []struct {
		name     string
		toolCall conversation.ToolCall
	}{
		{
			name: "missing id",
			toolCall: conversation.ToolCall{
				Name:      "lookup",
				Arguments: `{"q":"x"}`,
			},
		},
		{
			name: "missing name",
			toolCall: conversation.ToolCall{
				ID:        "call_123",
				Arguments: `{"q":"x"}`,
			},
		},
		{
			name: "invalid arguments json",
			toolCall: conversation.ToolCall{
				ID:        "call_123",
				Name:      "lookup",
				Arguments: `{"q":`,
			},
		},
		{
			name: "arguments json array",
			toolCall: conversation.ToolCall{
				ID:        "call_123",
				Name:      "lookup",
				Arguments: `["x"]`,
			},
		},
		{
			name: "arguments json string",
			toolCall: conversation.ToolCall{
				ID:        "call_123",
				Name:      "lookup",
				Arguments: `"x"`,
			},
		},
		{
			name: "arguments json null",
			toolCall: conversation.ToolCall{
				ID:        "call_123",
				Name:      "lookup",
				Arguments: `null`,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request := conversation.Request{
				Messages: []conversation.Message{
					{
						Role: "assistant",
						Blocks: []conversation.Block{
							{
								Type:     conversation.BlockTypeToolCall,
								ToolCall: &tc.toolCall,
							},
						},
					},
				},
			}

			_, err := TranslateConversation("model", request, nil, nil)
			assertInvalidRequestError(t, err)
		})
	}
}

func TestTranslateConversationRejectsMissingToolResultCallID(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{
						Type: conversation.BlockTypeToolResult,
						ToolResult: &conversation.ToolResult{
							Output: "ok",
						},
					},
				},
			},
		},
	}

	_, err := TranslateConversation("model", request, nil, nil)
	assertInvalidRequestError(t, err)
}

func TestTranslateConversationRejectsToolCallBlocksOutsideAssistantMessages(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{
						Type: conversation.BlockTypeToolCall,
						ToolCall: &conversation.ToolCall{
							ID:        "call_123",
							Name:      "lookup",
							Arguments: `{"q":"x"}`,
						},
					},
				},
			},
		},
	}

	_, err := TranslateConversation("model", request, nil, nil)
	assertInvalidRequestError(t, err)
}

func TestTranslateConversationRejectsToolResultBlocksOutsideUserMessages(t *testing.T) {
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "assistant",
				Blocks: []conversation.Block{
					{
						Type: conversation.BlockTypeToolResult,
						ToolResult: &conversation.ToolResult{
							CallID: "call_123",
							Output: "ok",
						},
					},
				},
			},
		},
	}

	_, err := TranslateConversation("model", request, nil, nil)
	assertInvalidRequestError(t, err)
}

func TestTranslateConversationMapsInferenceControls(t *testing.T) {
	maxTokens := 128
	temperature := 0.4
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "hello"},
				},
			},
		},
	}

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
	request := conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "hello"},
				},
			},
		},
	}

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

func assertInvalidRequestError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected invalid request error")
	}
	var invalid openai.InvalidRequestError
	if !errors.As(err, &invalid) {
		t.Fatalf("expected invalid request error, got %T", err)
	}
}
