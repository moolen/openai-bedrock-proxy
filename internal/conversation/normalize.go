package conversation

import (
	"encoding/json"
	"strings"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

const invalidResponsesInputErrorMessage = "input must be a non-empty string or supported message object/array"

func NormalizeRequest(req openai.ResponsesRequest) (Request, error) {
	system, messages, err := normalizeInput(req.Input)
	if err != nil {
		return Request{}, err
	}
	if req.Instructions != "" {
		system = append(system, req.Instructions)
	}
	if len(messages) == 0 {
		return Request{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	return Request{
		System:     system,
		Messages:   messages,
		Tools:      normalizeTools(req.Tools),
		ToolChoice: normalizeToolChoice(req.ToolChoice),
	}, nil
}

func Merge(base, current Request) Request {
	baseMessages := cloneMessages(base.Messages)
	currentMessages := cloneMessages(current.Messages)
	mergedMessages := make([]Message, 0, len(baseMessages)+len(currentMessages))
	mergedMessages = append(mergedMessages, baseMessages...)
	mergedMessages = append(mergedMessages, currentMessages...)
	return Request{
		System:     append([]string(nil), current.System...),
		Messages:   mergedMessages,
		Tools:      cloneToolDefinitions(current.Tools),
		ToolChoice: current.ToolChoice,
	}
}

func AppendAssistantReply(req Request, assistantBlocks []Block) Request {
	updated := Request{
		System:     append([]string(nil), req.System...),
		Messages:   cloneMessages(req.Messages),
		Tools:      cloneToolDefinitions(req.Tools),
		ToolChoice: req.ToolChoice,
	}
	updated.Messages = append(updated.Messages, Message{
		Role:   "assistant",
		Blocks: cloneBlocks(assistantBlocks),
	})
	return updated
}

func normalizeInput(input any) ([]string, []Message, error) {
	if input == nil {
		return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}

	switch value := input.(type) {
	case string:
		if value == "" {
			return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		blocks := []Block{{Type: BlockTypeText, Text: value}}
		return nil, []Message{{Role: "user", Blocks: blocks}}, nil
	case map[string]any:
		return normalizeMessage(value)
	case []map[string]any:
		return normalizeMessageSlice(value)
	case []any:
		return normalizeMessageItems(value)
	default:
		return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
}

func normalizeMessageSlice(messages []map[string]any) ([]string, []Message, error) {
	system := make([]string, 0, len(messages))
	persisted := make([]Message, 0, len(messages))
	for _, message := range messages {
		messageSystem, messagePersisted, err := normalizeMessage(message)
		if err != nil {
			return nil, nil, err
		}
		system = append(system, messageSystem...)
		persisted = append(persisted, messagePersisted...)
	}
	return system, persisted, nil
}

func normalizeMessageItems(messages []any) ([]string, []Message, error) {
	system := make([]string, 0, len(messages))
	persisted := make([]Message, 0, len(messages))
	for _, item := range messages {
		message, ok := item.(map[string]any)
		if !ok {
			return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		messageSystem, messagePersisted, err := normalizeMessage(message)
		if err != nil {
			return nil, nil, err
		}
		system = append(system, messageSystem...)
		persisted = append(persisted, messagePersisted...)
	}
	return system, persisted, nil
}

func normalizeMessage(message map[string]any) ([]string, []Message, error) {
	roleValue, ok := message["role"]
	if !ok {
		return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	role, ok := roleValue.(string)
	if !ok {
		return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	switch role {
	case "system", "developer", "user", "assistant":
	default:
		return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}

	if typeValue, ok := message["type"]; ok {
		messageType, ok := typeValue.(string)
		if !ok || messageType != "message" {
			return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
	}

	contentValue, ok := message["content"]
	if !ok || contentValue == nil {
		return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}

	blocks, err := normalizeContent(role, contentValue)
	if err != nil {
		return nil, nil, err
	}

	switch role {
	case "system", "developer":
		text := systemTextFromBlocks(blocks)
		if text == "" {
			return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		return []string{text}, nil, nil
	case "user", "assistant":
		return nil, []Message{{Role: role, Blocks: blocks}}, nil
	default:
		return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
}

func normalizeContent(role string, content any) ([]Block, error) {
	switch value := content.(type) {
	case string:
		if value == "" {
			return nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		return []Block{{Type: BlockTypeText, Text: value}}, nil
	case []map[string]any:
		return normalizeBlockSlice(role, value)
	case []any:
		return normalizeBlockItems(role, value)
	default:
		return nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
}

func normalizeBlockSlice(role string, blocks []map[string]any) ([]Block, error) {
	if len(blocks) == 0 {
		return nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	normalized := make([]Block, 0, len(blocks))
	for _, block := range blocks {
		normalizedBlock, err := normalizeBlock(role, block)
		if err != nil {
			return nil, err
		}
		normalized = append(normalized, normalizedBlock)
	}
	return normalized, nil
}

func normalizeBlockItems(role string, blocks []any) ([]Block, error) {
	if len(blocks) == 0 {
		return nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	normalized := make([]Block, 0, len(blocks))
	for _, item := range blocks {
		block, ok := item.(map[string]any)
		if !ok {
			return nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		normalizedBlock, err := normalizeBlock(role, block)
		if err != nil {
			return nil, err
		}
		normalized = append(normalized, normalizedBlock)
	}
	return normalized, nil
}

func normalizeBlock(role string, block map[string]any) (Block, error) {
	typeValue, ok := block["type"]
	if !ok {
		return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	blockType, ok := typeValue.(string)
	if !ok {
		return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}

	switch blockType {
	case "input_text":
		if role != "user" && role != "system" && role != "developer" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		text, err := normalizeBlockText(block)
		if err != nil {
			return Block{}, err
		}
		return Block{Type: BlockTypeText, Text: text}, nil
	case "output_text":
		if role != "assistant" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		text, err := normalizeBlockText(block)
		if err != nil {
			return Block{}, err
		}
		return Block{Type: BlockTypeText, Text: text}, nil
	case "function_call_output":
		if role != "user" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callIDValue, ok := block["call_id"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callID, ok := callIDValue.(string)
		if !ok || callID == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		output, ok := block["output"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		return Block{
			Type: BlockTypeToolResult,
			ToolResult: &ToolResult{
				CallID: callID,
				Output: cloneValue(output),
			},
		}, nil
	case "custom_tool_call_output":
		if role != "user" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callIDValue, ok := block["call_id"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callID, ok := callIDValue.(string)
		if !ok || callID == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		output, ok := block["output"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		return Block{
			Type: BlockTypeToolResult,
			ToolResult: &ToolResult{
				CallID: callID,
				Output: cloneValue(output),
			},
		}, nil
	case "tool_search_output":
		if role != "user" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callIDValue, ok := block["call_id"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callID, ok := callIDValue.(string)
		if !ok || callID == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		statusValue, ok := block["status"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		status, ok := statusValue.(string)
		if !ok || status == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		executionValue, ok := block["execution"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		execution, ok := executionValue.(string)
		if !ok || execution == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		toolsValue, ok := block["tools"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		return Block{
			Type: BlockTypeToolResult,
			ToolResult: &ToolResult{
				CallID: callID,
				Output: map[string]any{
					"type":      "tool_search_output",
					"status":    status,
					"execution": execution,
					"tools":     cloneValue(toolsValue),
				},
			},
		}, nil
	case "function_call":
		if role != "assistant" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callIDValue, ok := block["call_id"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callID, ok := callIDValue.(string)
		if !ok || callID == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		nameValue, ok := block["name"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		name, ok := nameValue.(string)
		if !ok || name == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		arguments := ""
		if argumentsValue, ok := block["arguments"]; ok {
			arguments, ok = argumentsValue.(string)
			if !ok {
				return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
			}
		}
		return Block{
			Type: BlockTypeToolCall,
			ToolCall: &ToolCall{
				ID:        callID,
				Name:      name,
				Arguments: arguments,
			},
		}, nil
	case "tool_search_call":
		if role != "assistant" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callIDValue, ok := block["call_id"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callID, ok := callIDValue.(string)
		if !ok || callID == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		argumentsValue, ok := block["arguments"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		return Block{
			Type: BlockTypeToolCall,
			ToolCall: &ToolCall{
				ID:        callID,
				Name:      syntheticBuiltInToolName("tool_search"),
				Arguments: mustMarshalValue(argumentsValue),
			},
		}, nil
	case "local_shell_call":
		if role != "assistant" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callIDValue, ok := block["call_id"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callID, ok := callIDValue.(string)
		if !ok || callID == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		actionValue, ok := block["action"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		return Block{
			Type: BlockTypeToolCall,
			ToolCall: &ToolCall{
				ID:        callID,
				Name:      syntheticBuiltInToolName("local_shell"),
				Arguments: mustMarshalValue(actionValue),
			},
		}, nil
	case "custom_tool_call":
		if role != "assistant" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callIDValue, ok := block["call_id"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		callID, ok := callIDValue.(string)
		if !ok || callID == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		nameValue, ok := block["name"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		name, ok := nameValue.(string)
		if !ok || name == "" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		inputValue, ok := block["input"]
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		input, ok := inputValue.(string)
		if !ok {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		return Block{
			Type: BlockTypeToolCall,
			ToolCall: &ToolCall{
				ID:        callID,
				Name:      name,
				Arguments: mustMarshalCustomToolInput(input),
			},
		}, nil
	case "image_generation_call":
		if role != "assistant" {
			return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		return Block{
			Type: BlockTypeText,
			Text: mustMarshalTaggedText("image_generation_call", block),
		}, nil
	default:
		return Block{}, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
}

func normalizeBlockText(block map[string]any) (string, error) {
	textValue, ok := block["text"]
	if !ok {
		return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	text, ok := textValue.(string)
	if !ok || text == "" {
		return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	return text, nil
}

func normalizeTools(tools []openai.Tool) []ToolDefinition {
	if len(tools) == 0 {
		return nil
	}
	normalized := make([]ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		if tool.Type == "function" {
			definition := ToolDefinition{
				Type: "function",
			}
			if tool.Function != nil {
				definition.Name = tool.Function.Name
				definition.Description = tool.Function.Description
				definition.Parameters = cloneStringAnyMap(tool.Function.Parameters)
			}
			if definition.Name == "" {
				definition.Name = tool.Name
			}
			normalized = append(normalized, definition)
			continue
		}
		if tool.Type == "custom" {
			normalized = append(normalized, ToolDefinition{
				Type:        tool.Type,
				Name:        syntheticCustomToolName(tool.Name),
				Description: tool.Description,
				Config:      cloneRawMessageMap(tool.Config),
				BuiltIn:     true,
			})
			continue
		}
		normalized = append(normalized, ToolDefinition{
			Type:        tool.Type,
			Name:        syntheticBuiltInToolName(tool.Type),
			Description: tool.Description,
			Config:      cloneRawMessageMap(tool.Config),
			BuiltIn:     true,
		})
	}
	return normalized
}

func normalizeToolChoice(choice *openai.ToolChoice) ToolChoice {
	if choice == nil {
		return ToolChoice{}
	}

	if choice.Type == "none" || (choice.Mode == "string" && choice.Type == "none") {
		return ToolChoice{}
	}
	if choice.Type == "auto" {
		return ToolChoice{Type: "auto"}
	}
	if choice.Mode == "string" && choice.Type == "auto" {
		return ToolChoice{Type: "auto"}
	}
	if choice.Type == "function" {
		name := choice.Name
		if choice.Function != nil && choice.Function.Name != "" {
			name = choice.Function.Name
		}
		return ToolChoice{
			Type: "function",
			Name: name,
		}
	}
	if choice.Type == "" {
		return ToolChoice{}
	}
	if choice.Type == "custom" {
		if choice.Name == "" {
			return ToolChoice{Type: "custom"}
		}
		return ToolChoice{
			Type: choice.Type,
			Name: syntheticCustomToolName(choice.Name),
		}
	}
	return ToolChoice{
		Type: choice.Type,
		Name: syntheticBuiltInToolName(choice.Type),
	}
}

func syntheticBuiltInToolName(toolType string) string {
	return "__builtin_" + toolType
}

func syntheticCustomToolName(toolName string) string {
	return "__custom_" + toolName
}

func mustMarshalCustomToolInput(input string) string {
	encoded, err := json.Marshal(map[string]any{"input": input})
	if err != nil {
		return `{"input":""}`
	}
	return string(encoded)
}

func mustMarshalValue(value any) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		return `{}`
	}
	return string(encoded)
}

func mustMarshalTaggedText(tag string, value any) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "<" + tag + ">{}</" + tag + ">"
	}
	return "<" + tag + ">" + string(encoded) + "</" + tag + ">"
}

func systemTextFromBlocks(blocks []Block) string {
	var builder strings.Builder
	for _, block := range blocks {
		if block.Type == BlockTypeText {
			builder.WriteString(block.Text)
		}
	}
	return builder.String()
}

func cloneBlocks(blocks []Block) []Block {
	if len(blocks) == 0 {
		return nil
	}
	cloned := make([]Block, len(blocks))
	for idx, block := range blocks {
		cloned[idx] = block
		if block.ToolCall != nil {
			toolCall := *block.ToolCall
			cloned[idx].ToolCall = &toolCall
		}
		if block.ToolResult != nil {
			toolResult := *block.ToolResult
			toolResult.Output = cloneValue(toolResult.Output)
			cloned[idx].ToolResult = &toolResult
		}
	}
	return cloned
}

func cloneStringAnyMap(input map[string]any) map[string]any {
	if len(input) == 0 {
		return nil
	}
	cloned := make(map[string]any, len(input))
	for key, value := range input {
		cloned[key] = cloneValue(value)
	}
	return cloned
}

func cloneRawMessageMap(input map[string]json.RawMessage) map[string]json.RawMessage {
	if len(input) == 0 {
		return nil
	}
	cloned := make(map[string]json.RawMessage, len(input))
	for key, value := range input {
		cloned[key] = append(json.RawMessage(nil), value...)
	}
	return cloned
}

func cloneToolDefinitions(tools []ToolDefinition) []ToolDefinition {
	if len(tools) == 0 {
		return nil
	}
	cloned := make([]ToolDefinition, len(tools))
	for idx, tool := range tools {
		cloned[idx] = ToolDefinition{
			Type:        tool.Type,
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  cloneStringAnyMap(tool.Parameters),
			Config:      cloneRawMessageMap(tool.Config),
			BuiltIn:     tool.BuiltIn,
		}
	}
	return cloned
}

func cloneValue(value any) any {
	switch typed := value.(type) {
	case json.RawMessage:
		return append(json.RawMessage(nil), typed...)
	case []byte:
		return append([]byte(nil), typed...)
	case map[string]any:
		return cloneStringAnyMap(typed)
	case []any:
		cloned := make([]any, len(typed))
		for idx, item := range typed {
			cloned[idx] = cloneValue(item)
		}
		return cloned
	case []map[string]any:
		cloned := make([]map[string]any, len(typed))
		for idx, item := range typed {
			cloned[idx] = cloneStringAnyMap(item)
		}
		return cloned
	default:
		return value
	}
}
