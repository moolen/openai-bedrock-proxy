package bedrock

import (
	"encoding/json"
	"strings"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

const (
	OutputBlockTypeText     = "text"
	OutputBlockTypeToolCall = "tool_call"
)

type ToolCall struct {
	ID        string
	Name      string
	Arguments string
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
	if toolType, ok := syntheticBuiltInType(call.Name); ok {
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
	arguments = normalizeArgumentsString(arguments)

	var parsed any
	if err := json.Unmarshal([]byte(arguments), &parsed); err != nil {
		return map[string]any{"input": arguments}
	}
	if object, ok := parsed.(map[string]any); ok {
		return object
	}
	return map[string]any{"input": parsed}
}

func normalizeArgumentsString(arguments string) string {
	if strings.TrimSpace(arguments) == "" {
		return "{}"
	}
	return arguments
}
