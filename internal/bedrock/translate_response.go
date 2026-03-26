package bedrock

import (
	"encoding/json"
	"strings"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

const (
	OutputBlockTypeText      = "text"
	OutputBlockTypeToolCall  = "tool_call"
	OutputBlockTypeReasoning = "reasoning"
)

type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

type OutputBlock struct {
	Type     string
	Text     string
	ToolCall *ToolCall
}

type ConverseResponse struct {
	ResponseID string
	Output     []OutputBlock
	StopReason string
	Usage      *Usage
}

type TextAccumulator struct {
	builder strings.Builder
}

func (t *TextAccumulator) Add(delta string) {
	t.builder.WriteString(delta)
}

func (t *TextAccumulator) Text() string {
	return t.builder.String()
}

func TranslateResponse(resp ConverseResponse, model string) openai.Response {
	return openai.Response{
		ID:     "resp_" + resp.ResponseID,
		Object: "response",
		Model:  model,
		Output: translateOutputItems(resp.Output),
	}
}

func translateOutputItems(blocks []OutputBlock) []openai.OutputItem {
	if len(blocks) == 0 {
		return nil
	}

	items := make([]openai.OutputItem, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case OutputBlockTypeToolCall:
			if block.ToolCall == nil {
				continue
			}
			items = append(items, translateToolCallItem(*block.ToolCall))
		case OutputBlockTypeReasoning:
			continue
		default:
			items = append(items, openai.OutputItem{
				Type: "message",
				Role: "assistant",
				Content: []openai.ContentItem{
					{
						Type: "output_text",
						Text: block.Text,
					},
				},
			})
		}
	}

	return items
}

func translateToolCallItem(call ToolCall) openai.OutputItem {
	if toolName, ok := syntheticCustomToolName(call.Name); ok {
		return openai.OutputItem{
			Type:   "custom_tool_call",
			CallID: call.ID,
			Name:   toolName,
			Input:  parseCustomToolInput(call.Arguments),
		}
	}
	if toolType, ok := syntheticBuiltInType(call.Name); ok {
		switch toolType {
		case "tool_search":
			return openai.OutputItem{
				Type: "tool_search_call",
				Raw: map[string]any{
					"type":      "tool_search_call",
					"call_id":   call.ID,
					"execution": "client",
					"arguments": parseToolCallPayload(call.Arguments),
				},
			}
		case "local_shell":
			return openai.OutputItem{
				Type: "local_shell_call",
				Raw: map[string]any{
					"type":    "local_shell_call",
					"call_id": call.ID,
					"status":  "completed",
					"action":  parseLocalShellAction(call.Arguments),
				},
			}
		case "image_generation":
			raw := parseToolCallObject(call.Arguments)
			payload := map[string]any{
				"type":   "image_generation_call",
				"id":     firstNonEmptyString(raw["id"], call.ID),
				"status": firstNonEmptyString(raw["status"], "completed"),
				"result": firstNonEmptyString(raw["result"], ""),
			}
			if revisedPrompt := firstNonEmptyString(raw["revised_prompt"]); revisedPrompt != "" {
				payload["revised_prompt"] = revisedPrompt
			}
			return openai.OutputItem{
				Type: "image_generation_call",
				Raw:  payload,
			}
		}
		return openai.OutputItem{
			Type:   builtInToolCallOutputType(toolType),
			CallID: call.ID,
			Action: parseToolCallAction(call.Arguments),
		}
	}

	return openai.OutputItem{
		Type:      "function_call",
		CallID:    call.ID,
		Name:      call.Name,
		Arguments: normalizeArgumentsString(call.Arguments),
	}
}

func syntheticCustomToolName(name string) (string, bool) {
	const prefix = "__custom_"
	if !strings.HasPrefix(name, prefix) {
		return "", false
	}
	toolName := strings.TrimPrefix(name, prefix)
	if toolName == "" {
		return "", false
	}
	return toolName, true
}

func syntheticBuiltInType(name string) (string, bool) {
	const prefix = "__builtin_"
	if !strings.HasPrefix(name, prefix) {
		return "", false
	}
	toolType := strings.TrimPrefix(name, prefix)
	if toolType == "" {
		return "", false
	}
	return toolType, true
}

func builtInToolCallOutputType(toolType string) string {
	switch toolType {
	case "web_search_preview":
		return "web_search_call"
	case "file_search":
		return "file_search_call"
	case "computer_use_preview":
		return "computer_call"
	case "code_interpreter":
		return "code_interpreter_call"
	default:
		return toolType + "_call"
	}
}

func parseToolCallAction(arguments string) map[string]any {
	if object, ok := parseToolCallPayload(arguments).(map[string]any); ok {
		return object
	}
	return map[string]any{"input": normalizeArgumentsString(arguments)}
}

func parseCustomToolInput(arguments string) string {
	arguments = normalizeArgumentsString(arguments)

	var parsed any
	if err := json.Unmarshal([]byte(arguments), &parsed); err != nil {
		return arguments
	}
	object, ok := parsed.(map[string]any)
	if !ok {
		return arguments
	}
	input, ok := object["input"].(string)
	if !ok {
		return arguments
	}
	return input
}

func normalizeArgumentsString(arguments string) string {
	if strings.TrimSpace(arguments) == "" {
		return "{}"
	}
	return arguments
}

func parseToolCallPayload(arguments string) any {
	arguments = normalizeArgumentsString(arguments)

	var parsed any
	if err := json.Unmarshal([]byte(arguments), &parsed); err != nil {
		return map[string]any{"input": arguments}
	}
	return parsed
}

func parseToolCallObject(arguments string) map[string]any {
	parsed, ok := parseToolCallPayload(arguments).(map[string]any)
	if !ok {
		return map[string]any{}
	}
	return parsed
}

func parseLocalShellAction(arguments string) map[string]any {
	object := parseToolCallObject(arguments)
	if _, ok := object["type"].(string); ok {
		return object
	}

	action := map[string]any{
		"type": "exec",
	}
	for key, value := range object {
		action[key] = value
	}
	return action
}

func firstNonEmptyString(values ...any) string {
	for _, value := range values {
		text, ok := value.(string)
		if ok && text != "" {
			return text
		}
	}
	return ""
}
