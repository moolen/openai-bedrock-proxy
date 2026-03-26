package bedrock

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type ContentBlock struct {
	Text       string
	Image      *ImageBlock
	CachePoint *CachePointBlock
	ToolUse    *ToolUseBlock
	ToolResult *ToolResultBlock
}

type ImageBlock struct {
	Format string
	Bytes  []byte
}

type CachePointBlock struct {
	Type string
}

type ToolUseBlock struct {
	ToolUseID string
	Name      string
	Input     any
}

type ToolResultBlock struct {
	ToolUseID string
	Content   []ToolResultContentBlock
}

const (
	toolResultContentTypeText = "text"
	toolResultContentTypeJSON = "json"
)

type ToolResultContentBlock struct {
	Type string
	Text string
	JSON any
}

type Message struct {
	Role    string
	Content []ContentBlock
}

type ToolSpec struct {
	Name        string
	Description string
	InputSchema map[string]any
}

type ToolChoice struct {
	Auto bool
	Tool string
}

type ToolConfig struct {
	Tools      []ToolSpec
	ToolChoice *ToolChoice
}

type ConverseRequest struct {
	ModelID                      string
	System                       []string
	SystemCachePoint             bool
	Messages                     []Message
	MaxTokens                    *int32
	Temperature                  *float32
	TopP                         *float32
	StopSequences                []string
	ToolConfig                   *ToolConfig
	AdditionalModelRequestFields map[string]any
}

func TranslateConversation(modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64) (ConverseRequest, error) {
	messages, err := toBedrockConversationMessages(req.Messages)
	if err != nil {
		return ConverseRequest{}, err
	}
	toolConfig, err := toToolConfig(req.Tools, req.ToolChoice)
	if err != nil {
		return ConverseRequest{}, err
	}

	out := ConverseRequest{
		ModelID:    modelID,
		System:     append([]string(nil), req.System...),
		Messages:   messages,
		ToolConfig: toolConfig,
	}
	if maxOutputTokens != nil {
		if *maxOutputTokens < 0 || *maxOutputTokens > math.MaxInt32 {
			return ConverseRequest{}, openai.NewInvalidRequestError("max_output_tokens is out of range")
		}
		maxTokens := int32(*maxOutputTokens)
		out.MaxTokens = &maxTokens
	}
	if temperature != nil {
		temperature32 := float32(*temperature)
		out.Temperature = &temperature32
	}
	return out, nil
}

func TranslateRequest(req openai.ResponsesRequest) (ConverseRequest, error) {
	normalized, err := conversation.NormalizeRequest(req)
	if err != nil {
		return ConverseRequest{}, err
	}
	return TranslateConversation(req.Model, normalized, req.MaxOutputTokens, req.Temperature)
}

func toBedrockConversationMessages(messages []conversation.Message) ([]Message, error) {
	out := make([]Message, 0, len(messages))
	for _, message := range messages {
		content, err := toBedrockContentBlocks(message.Role, message.Blocks)
		if err != nil {
			return nil, err
		}
		out = append(out, Message{Role: message.Role, Content: content})
	}
	return out, nil
}

func toBedrockContentBlocks(role string, blocks []conversation.Block) ([]ContentBlock, error) {
	out := make([]ContentBlock, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case conversation.BlockTypeText:
			out = append(out, ContentBlock{Text: block.Text})
		case conversation.BlockTypeToolCall:
			if role != "assistant" {
				return nil, openai.NewInvalidRequestError("tool_call blocks are only allowed in assistant messages")
			}
			if block.ToolCall == nil {
				return nil, openai.NewInvalidRequestError("assistant tool_call block is missing data")
			}
			if strings.TrimSpace(block.ToolCall.ID) == "" {
				return nil, openai.NewInvalidRequestError("assistant tool_call id is required")
			}
			if strings.TrimSpace(block.ToolCall.Name) == "" {
				return nil, openai.NewInvalidRequestError("assistant tool_call name is required")
			}
			input, err := parseToolUseInput(block.ToolCall.Arguments)
			if err != nil {
				return nil, err
			}
			out = append(out, ContentBlock{
				ToolUse: &ToolUseBlock{
					ToolUseID: block.ToolCall.ID,
					Name:      block.ToolCall.Name,
					Input:     input,
				},
			})
		case conversation.BlockTypeToolResult:
			if role != "user" {
				return nil, openai.NewInvalidRequestError("tool_result blocks are only allowed in user messages")
			}
			if block.ToolResult == nil {
				return nil, openai.NewInvalidRequestError("user tool_result block is missing data")
			}
			if strings.TrimSpace(block.ToolResult.CallID) == "" {
				return nil, openai.NewInvalidRequestError("user tool_result call_id is required")
			}
			out = append(out, ContentBlock{
				ToolResult: &ToolResultBlock{
					ToolUseID: block.ToolResult.CallID,
					Content:   toToolResultContentBlocks(block.ToolResult.Output),
				},
			})
		default:
			return nil, openai.NewInvalidRequestError("unsupported conversation block type")
		}
	}
	return out, nil
}

func toToolConfig(tools []conversation.ToolDefinition, choice conversation.ToolChoice) (*ToolConfig, error) {
	if len(tools) == 0 {
		if choice != (conversation.ToolChoice{}) {
			return nil, openai.NewInvalidRequestError("tool_choice requires tools")
		}
		return nil, nil
	}

	specs := make([]ToolSpec, 0, len(tools))
	for _, tool := range tools {
		spec, err := toToolSpec(tool)
		if err != nil {
			return nil, err
		}
		specs = append(specs, spec)
	}

	toolChoice, err := toBedrockToolChoice(choice, tools)
	if err != nil {
		return nil, err
	}
	if toolChoice != nil && toolChoice.Tool != "" && !toolSetContains(specs, toolChoice.Tool) {
		return nil, openai.NewInvalidRequestError("tool_choice target is not present in tools")
	}
	return &ToolConfig{
		Tools:      specs,
		ToolChoice: toolChoice,
	}, nil
}

func toBedrockToolChoice(choice conversation.ToolChoice, tools []conversation.ToolDefinition) (*ToolChoice, error) {
	switch {
	case choice == (conversation.ToolChoice{}):
		return nil, nil
	case choice.Type == "auto":
		return &ToolChoice{Auto: true}, nil
	case choice.Name != "" && namedToolChoiceAllowed(choice, tools):
		return &ToolChoice{Tool: choice.Name}, nil
	default:
		return nil, openai.NewInvalidRequestError("unsupported tool_choice")
	}
}

func toToolSpec(tool conversation.ToolDefinition) (ToolSpec, error) {
	if tool.BuiltIn {
		if strings.TrimSpace(tool.Type) == "" {
			return ToolSpec{}, openai.NewInvalidRequestError("built-in tool type is required")
		}
		if strings.TrimSpace(tool.Name) == "" {
			return ToolSpec{}, openai.NewInvalidRequestError("built-in tool name is required")
		}
		schema, err := syntheticBuiltInInputSchema(tool)
		if err != nil {
			return ToolSpec{}, err
		}
		return ToolSpec{
			Name:        tool.Name,
			Description: syntheticBuiltInDescription(tool),
			InputSchema: schema,
		}, nil
	}
	if tool.Type != "function" {
		return ToolSpec{}, openai.NewInvalidRequestError("unsupported tool definition type")
	}
	if strings.TrimSpace(tool.Name) == "" {
		return ToolSpec{}, openai.NewInvalidRequestError("function tool name is required")
	}

	return ToolSpec{
		Name:        tool.Name,
		Description: tool.Description,
		InputSchema: functionInputSchema(tool.Parameters),
	}, nil
}

func syntheticBuiltInInputSchema(tool conversation.ToolDefinition) (map[string]any, error) {
	schema := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{},
		"additionalProperties": true,
		"x-openai-tool-type":   tool.Type,
	}

	config, err := decodeBuiltInConfig(tool.Config)
	if err != nil {
		return nil, err
	}
	if len(config) > 0 {
		schema["x-openai-config"] = config
	}

	switch tool.Type {
	case "web_search_preview", "web_search":
		schema["properties"] = map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "Search query",
			},
		}
		schema["required"] = []any{"query"}
		schema["additionalProperties"] = false
	case "tool_search":
		schema["properties"] = map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "Search query for tools",
			},
			"limit": map[string]any{
				"type":        "number",
				"description": "Maximum number of tools to return",
			},
		}
		schema["required"] = []any{"query"}
		schema["additionalProperties"] = false
	case "local_shell":
		schema["properties"] = map[string]any{
			"type": map[string]any{
				"type":        "string",
				"description": "Local shell action type",
			},
			"command": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "string",
				},
				"description": "Command to execute",
			},
			"timeout_ms": map[string]any{
				"type":        "number",
				"description": "Optional timeout in milliseconds",
			},
			"working_directory": map[string]any{
				"type":        "string",
				"description": "Optional working directory",
			},
			"user": map[string]any{
				"type":        "string",
				"description": "Optional user to run as",
			},
			"env": map[string]any{
				"type":        "object",
				"description": "Optional environment overrides",
			},
		}
		schema["required"] = []any{"type", "command"}
		schema["additionalProperties"] = false
	case "custom":
		schema["properties"] = map[string]any{
			"input": map[string]any{
				"type":        "string",
				"description": fmt.Sprintf("Raw input for %s", tool.Name),
			},
		}
		schema["required"] = []any{"input"}
		schema["additionalProperties"] = false
	default:
		schema["properties"] = map[string]any{
			"input": map[string]any{
				"type":        "string",
				"description": fmt.Sprintf("Synthetic input for %s", tool.Type),
			},
		}
	}

	return schema, nil
}

func decodeBuiltInConfig(config map[string]json.RawMessage) (map[string]any, error) {
	if len(config) == 0 {
		return nil, nil
	}

	decoded := make(map[string]any, len(config))
	for key, value := range config {
		var parsed any
		if err := json.Unmarshal(value, &parsed); err != nil {
			return nil, openai.NewInvalidRequestError("built-in tool config must be valid JSON")
		}
		decoded[key] = parsed
	}
	return decoded, nil
}

func functionInputSchema(parameters map[string]any) map[string]any {
	if len(parameters) == 0 {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	}
	return parameters
}

func syntheticBuiltInDescription(tool conversation.ToolDefinition) string {
	if tool.Description != "" {
		return tool.Description
	}
	return fmt.Sprintf("Synthetic %s tool", tool.Type)
}

func parseToolUseInput(arguments string) (any, error) {
	if strings.TrimSpace(arguments) == "" {
		return map[string]any{}, nil
	}

	var input any
	if err := json.Unmarshal([]byte(arguments), &input); err != nil {
		return nil, openai.NewInvalidRequestError("assistant tool_call arguments must be valid JSON")
	}
	inputObject, ok := input.(map[string]any)
	if !ok {
		return nil, openai.NewInvalidRequestError("assistant tool_call arguments must be a JSON object")
	}
	return inputObject, nil
}

func toToolResultContentBlocks(output any) []ToolResultContentBlock {
	if text, ok := output.(string); ok {
		return []ToolResultContentBlock{{
			Type: toolResultContentTypeText,
			Text: text,
		}}
	}
	return []ToolResultContentBlock{{
		Type: toolResultContentTypeJSON,
		JSON: output,
	}}
}

func toolSetContains(tools []ToolSpec, name string) bool {
	for _, tool := range tools {
		if tool.Name == name {
			return true
		}
	}
	return false
}

func namedToolChoiceAllowed(choice conversation.ToolChoice, tools []conversation.ToolDefinition) bool {
	for _, tool := range tools {
		if tool.Name != choice.Name {
			continue
		}
		if choice.Type == "" || tool.Type == choice.Type {
			return true
		}
	}
	return false
}
