package openai

func ValidateResponsesRequest(req ResponsesRequest) error {
	if req.Model == "" {
		return NewInvalidRequestError("model is required")
	}
	if isEmptyInput(req.Input) {
		return NewInvalidRequestError("input is required")
	}
	if !isSupportedInput(req.Input) {
		return NewInvalidRequestError("structured input is not supported")
	}
	if len(req.Tools) > 0 {
		return NewInvalidRequestError("tools are not supported")
	}
	if req.ToolChoice != nil {
		return NewInvalidRequestError("tool_choice is not supported")
	}
	if req.ParallelToolCalls != nil {
		return NewInvalidRequestError("parallel_tool_calls is not supported")
	}
	return nil
}

func isEmptyInput(input any) bool {
	if input == nil {
		return true
	}

	text, ok := input.(string)
	if ok {
		return text == ""
	}

	if items, ok := input.([]any); ok {
		return len(items) == 0
	}
	if items, ok := input.([]map[string]any); ok {
		return len(items) == 0
	}

	if item, ok := input.(map[string]any); ok {
		return len(item) == 0
	}

	return false
}

func isSupportedInput(input any) bool {
	if input == nil {
		return false
	}

	if _, ok := input.(string); ok {
		return true
	}

	if message, ok := input.(map[string]any); ok {
		return isValidMessage(message)
	}

	if messages, ok := input.([]map[string]any); ok {
		return isValidMessageSlice(messages)
	}

	if messages, ok := input.([]any); ok {
		return isValidMessageItems(messages)
	}

	return false
}

func isValidMessageSlice(messages []map[string]any) bool {
	if len(messages) == 0 {
		return false
	}
	for _, message := range messages {
		if !isValidMessage(message) {
			return false
		}
	}
	return true
}

func isValidMessageItems(messages []any) bool {
	if len(messages) == 0 {
		return false
	}
	for _, item := range messages {
		message, ok := item.(map[string]any)
		if !ok {
			return false
		}
		if !isValidMessage(message) {
			return false
		}
	}
	return true
}

func isValidMessage(message map[string]any) bool {
	roleValue, ok := message["role"]
	if !ok {
		return false
	}
	role, ok := roleValue.(string)
	if !ok {
		return false
	}
	allowedBlockType, ok := allowedBlockTypesByRole[role]
	if !ok {
		return false
	}

	if typeValue, ok := message["type"]; ok {
		messageType, ok := typeValue.(string)
		if !ok || messageType != "message" {
			return false
		}
	}

	contentValue, ok := message["content"]
	if !ok || contentValue == nil {
		return false
	}

	switch content := contentValue.(type) {
	case string:
		return true
	case []map[string]any:
		return isValidTextBlocks(allowedBlockType, content)
	case []any:
		return isValidTextBlockItems(allowedBlockType, content)
	default:
		return false
	}
}

func isValidTextBlocks(allowedBlockType string, blocks []map[string]any) bool {
	if len(blocks) == 0 {
		return false
	}
	for _, block := range blocks {
		if !isValidTextBlock(allowedBlockType, block) {
			return false
		}
	}
	return true
}

func isValidTextBlockItems(allowedBlockType string, blocks []any) bool {
	if len(blocks) == 0 {
		return false
	}
	for _, item := range blocks {
		block, ok := item.(map[string]any)
		if !ok {
			return false
		}
		if !isValidTextBlock(allowedBlockType, block) {
			return false
		}
	}
	return true
}

func isValidTextBlock(allowedBlockType string, block map[string]any) bool {
	typeValue, ok := block["type"]
	if !ok {
		return false
	}
	blockType, ok := typeValue.(string)
	if !ok || blockType != allowedBlockType {
		return false
	}
	textValue, ok := block["text"]
	if !ok {
		return false
	}
	_, ok = textValue.(string)
	return ok
}

var allowedBlockTypesByRole = map[string]string{
	"user":      "input_text",
	"system":    "input_text",
	"developer": "input_text",
	"assistant": "output_text",
}
