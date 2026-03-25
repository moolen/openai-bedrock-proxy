package openai

import "strconv"

const invalidResponsesInputErrorMessage = "input must be a non-empty string or supported message object/array"

var supportedBuiltInToolTypes = map[string]struct{}{
	"web_search_preview":   {},
	"file_search":          {},
	"computer_use_preview": {},
	"code_interpreter":     {},
}

func ValidateResponsesRequest(req ResponsesRequest) error {
	if req.Model == "" {
		return NewInvalidRequestError("model is required")
	}
	if isEmptyInput(req.Input) {
		return NewInvalidRequestError("input is required")
	}
	if !isSupportedInput(req.Input) {
		return NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	if err := validateTools(req.Tools); err != nil {
		return err
	}
	if err := validateToolChoice(req.ToolChoice); err != nil {
		return err
	}
	if req.ParallelToolCalls != nil {
		return NewInvalidRequestError("parallel_tool_calls is not supported")
	}
	return nil
}

func validateTools(tools []Tool) error {
	for idx, tool := range tools {
		if tool.Type == "" {
			return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].type is required")
		}
		if tool.Type == "function" {
			if tool.Function == nil || tool.Function.Name == "" {
				return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].function.name is required")
			}
			continue
		}
		if _, ok := supportedBuiltInToolTypes[tool.Type]; !ok {
			return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].type is not supported")
		}
	}
	return nil
}

func validateToolChoice(choice any) error {
	if choice == nil {
		return nil
	}
	switch v := choice.(type) {
	case string:
		if v != "auto" {
			return NewInvalidRequestError("tool_choice is invalid")
		}
		return nil
	case map[string]any:
		return validateToolChoiceMap(v)
	case ToolChoice:
		return validateToolChoiceStruct(v)
	case *ToolChoice:
		if v == nil {
			return nil
		}
		return validateToolChoiceStruct(*v)
	default:
		return NewInvalidRequestError("tool_choice is invalid")
	}
}

func validateToolChoiceMap(choice map[string]any) error {
	typeValue, ok := choice["type"]
	if !ok {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	choiceType, ok := typeValue.(string)
	if !ok || choiceType == "" {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	if choiceType == "function" {
		functionValue, ok := choice["function"]
		if !ok {
			return NewInvalidRequestError("tool_choice.function.name is required")
		}
		functionMap, ok := functionValue.(map[string]any)
		if !ok {
			return NewInvalidRequestError("tool_choice.function.name is required")
		}
		nameValue, ok := functionMap["name"]
		if !ok {
			return NewInvalidRequestError("tool_choice.function.name is required")
		}
		name, ok := nameValue.(string)
		if !ok || name == "" {
			return NewInvalidRequestError("tool_choice.function.name is required")
		}
		return nil
	}
	if _, ok := supportedBuiltInToolTypes[choiceType]; ok {
		return nil
	}
	return NewInvalidRequestError("tool_choice is invalid")
}

func validateToolChoiceStruct(choice ToolChoice) error {
	if choice.Type == "" {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	if choice.Type == "function" {
		if choice.Function == nil || choice.Function.Name == "" {
			return NewInvalidRequestError("tool_choice.function.name is required")
		}
		return nil
	}
	if _, ok := supportedBuiltInToolTypes[choice.Type]; ok {
		return nil
	}
	return NewInvalidRequestError("tool_choice is invalid")
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
		return content != ""
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
	text, ok := textValue.(string)
	return ok && text != ""
}

var allowedBlockTypesByRole = map[string]string{
	"user":      "input_text",
	"system":    "input_text",
	"developer": "input_text",
	"assistant": "output_text",
}
