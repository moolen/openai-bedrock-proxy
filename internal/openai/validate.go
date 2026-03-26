package openai

import (
	"strconv"
	"strings"
)

const invalidResponsesInputErrorMessage = "input must be a non-empty string or supported message object/array"

var supportedBuiltInToolTypes = map[string]struct{}{
	"web_search_preview":   {},
	"web_search":           {},
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
	functionToolNames := collectFunctionToolNames(req.Tools)
	if err := validateToolChoice(req.ToolChoice, functionToolNames); err != nil {
		return err
	}
	return nil
}

func collectFunctionToolNames(tools []Tool) map[string]struct{} {
	names := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		if tool.Type == "function" {
			if name := functionToolName(tool); name != "" {
				names[name] = struct{}{}
			}
		}
	}
	return names
}

func validateTools(tools []Tool) error {
	seenFunctionNames := make(map[string]struct{}, len(tools))
	for idx, tool := range tools {
		if tool.Type == "" {
			return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].type is required")
		}
		if tool.Type == "function" {
			name := functionToolName(tool)
			if name == "" {
				return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].function.name is required")
			}
			if tool.Name != "" && tool.Name != name {
				return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].name must match tools[" + strconv.Itoa(idx) + "].function.name")
			}
			if _, ok := seenFunctionNames[name]; ok {
				return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].function.name duplicates a previous tool")
			}
			seenFunctionNames[name] = struct{}{}
			continue
		}
		if tool.Type == "custom" {
			if tool.Function != nil || tool.hasFunctionField {
				return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].function is only allowed for function tools")
			}
			if strings.TrimSpace(tool.Name) == "" {
				return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].name is required")
			}
			continue
		}
		if _, ok := supportedBuiltInToolTypes[tool.Type]; !ok {
			return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].type is not supported")
		}
		if tool.Function != nil || tool.hasFunctionField {
			return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].function is only allowed for function tools")
		}
		if tool.Name != "" || tool.hasNameField {
			return NewInvalidRequestError("tools[" + strconv.Itoa(idx) + "].name is only allowed for function tools")
		}
	}
	return nil
}

func functionToolName(tool Tool) string {
	if tool.Function != nil && tool.Function.Name != "" {
		return tool.Function.Name
	}
	return tool.Name
}

func validateToolChoice(choice *ToolChoice, functionToolNames map[string]struct{}) error {
	if choice == nil {
		return nil
	}
	if choice.Mode == "" && choice.Type == "" && choice.Name == "" && choice.Function == nil {
		return nil
	}
	if choice.Mode == "invalid" {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	if choice.Mode == "string" {
		if choice.Type != "auto" {
			return NewInvalidRequestError("tool_choice is invalid")
		}
		if choice.Function != nil || choice.hasFunctionField {
			return NewInvalidRequestError("tool_choice.function is not allowed for this tool_choice type")
		}
		if choice.Name != "" || choice.hasNameField {
			return NewInvalidRequestError("tool_choice.name is not allowed for this tool_choice type")
		}
		return nil
	}
	if choice.Type == "auto" {
		if choice.Function != nil || choice.hasFunctionField {
			return NewInvalidRequestError("tool_choice.function is not allowed for this tool_choice type")
		}
		if choice.Name != "" || choice.hasNameField {
			return NewInvalidRequestError("tool_choice.name is not allowed for this tool_choice type")
		}
		return nil
	}
	return validateToolChoiceStruct(*choice, functionToolNames)
}

func validateToolChoiceStruct(choice ToolChoice, functionToolNames map[string]struct{}) error {
	if choice.Type == "" {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	if choice.Type == "function" {
		if choice.hasFunctionField && (choice.Function == nil || choice.Function.Name == "") {
			return NewInvalidRequestError("tool_choice.function.name is required")
		}
		name := choice.Name
		if choice.Function != nil && choice.Function.Name != "" {
			if name != "" && name != choice.Function.Name {
				return NewInvalidRequestError("tool_choice.function.name is invalid")
			}
			name = choice.Function.Name
		}
		if name == "" {
			return NewInvalidRequestError("tool_choice.function.name is required")
		}
		if _, ok := functionToolNames[name]; !ok {
			return NewInvalidRequestError("tool_choice.function.name is not present in tools")
		}
		return nil
	}
	if _, ok := supportedBuiltInToolTypes[choice.Type]; ok {
		if choice.Function != nil || choice.hasFunctionField {
			return NewInvalidRequestError("tool_choice.function is not allowed for this tool_choice type")
		}
		if choice.Name != "" || choice.hasNameField {
			return NewInvalidRequestError("tool_choice.name is not allowed for this tool_choice type")
		}
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
	switch role {
	case "user", "system", "developer", "assistant":
	default:
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
		return isValidBlocks(role, content)
	case []any:
		return isValidBlockItems(role, content)
	default:
		return false
	}
}

func isValidBlocks(role string, blocks []map[string]any) bool {
	if len(blocks) == 0 {
		return false
	}
	for _, block := range blocks {
		if !isValidBlock(role, block) {
			return false
		}
	}
	return true
}

func isValidBlockItems(role string, blocks []any) bool {
	if len(blocks) == 0 {
		return false
	}
	for _, item := range blocks {
		block, ok := item.(map[string]any)
		if !ok {
			return false
		}
		if !isValidBlock(role, block) {
			return false
		}
	}
	return true
}

func isValidBlock(role string, block map[string]any) bool {
	typeValue, ok := block["type"]
	if !ok {
		return false
	}
	blockType, ok := typeValue.(string)
	if !ok {
		return false
	}

	switch blockType {
	case "input_text":
		if role != "user" && role != "system" && role != "developer" {
			return false
		}
		textValue, ok := block["text"]
		if !ok {
			return false
		}
		text, ok := textValue.(string)
		return ok && text != ""
	case "output_text":
		if role != "assistant" {
			return false
		}
		textValue, ok := block["text"]
		if !ok {
			return false
		}
		text, ok := textValue.(string)
		return ok && text != ""
	case "function_call_output", "custom_tool_call_output":
		if role != "user" {
			return false
		}
		callIDValue, ok := block["call_id"]
		if !ok {
			return false
		}
		callID, ok := callIDValue.(string)
		if !ok || callID == "" {
			return false
		}
		_, ok = block["output"]
		return ok
	case "function_call":
		if role != "assistant" {
			return false
		}
		callIDValue, ok := block["call_id"]
		if !ok {
			return false
		}
		callID, ok := callIDValue.(string)
		if !ok || callID == "" {
			return false
		}
		nameValue, ok := block["name"]
		if !ok {
			return false
		}
		name, ok := nameValue.(string)
		if !ok || name == "" {
			return false
		}
		if argumentsValue, ok := block["arguments"]; ok {
			_, ok = argumentsValue.(string)
			return ok
		}
		return true
	case "custom_tool_call":
		if role != "assistant" {
			return false
		}
		callIDValue, ok := block["call_id"]
		if !ok {
			return false
		}
		callID, ok := callIDValue.(string)
		if !ok || callID == "" {
			return false
		}
		nameValue, ok := block["name"]
		if !ok {
			return false
		}
		name, ok := nameValue.(string)
		if !ok || name == "" {
			return false
		}
		inputValue, ok := block["input"]
		if !ok {
			return false
		}
		_, ok = inputValue.(string)
		return ok
	default:
		return false
	}
}
