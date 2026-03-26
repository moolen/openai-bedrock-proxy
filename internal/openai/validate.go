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
	"tool_search":          {},
	"local_shell":          {},
	"image_generation":     {},
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

	if item, ok := input.(map[string]any); ok {
		return isValidInputItem(item)
	}

	if items, ok := input.([]map[string]any); ok {
		return isValidInputItemSlice(items)
	}

	if items, ok := input.([]any); ok {
		return isValidInputItems(items)
	}

	return false
}

func isValidInputItemSlice(items []map[string]any) bool {
	if len(items) == 0 {
		return false
	}
	for _, item := range items {
		if !isValidInputItem(item) {
			return false
		}
	}
	return true
}

func isValidInputItems(items []any) bool {
	if len(items) == 0 {
		return false
	}
	for _, item := range items {
		object, ok := item.(map[string]any)
		if !ok {
			return false
		}
		if !isValidInputItem(object) {
			return false
		}
	}
	return true
}

func isValidInputItem(item map[string]any) bool {
	if isMessageLikeInputItem(item) {
		return isValidMessage(item)
	}
	return isValidStandaloneInputItem(item)
}

func isMessageLikeInputItem(item map[string]any) bool {
	if _, ok := item["role"]; ok {
		return true
	}
	typeValue, ok := item["type"]
	if !ok {
		return false
	}
	itemType, ok := typeValue.(string)
	return ok && itemType == "message"
}

func isValidStandaloneInputItem(item map[string]any) bool {
	typeValue, ok := item["type"]
	if !ok {
		return false
	}
	itemType, ok := typeValue.(string)
	if !ok {
		return false
	}
	role, ok := standaloneInputItemRole(itemType)
	if !ok {
		return false
	}
	return isValidBlock(role, item)
}

func standaloneInputItemRole(itemType string) (string, bool) {
	switch itemType {
	case "function_call_output", "custom_tool_call_output", "mcp_tool_call_output", "tool_search_output":
		return "user", true
	case "function_call", "tool_search_call", "custom_tool_call", "local_shell_call", "image_generation_call":
		return "assistant", true
	default:
		return "", false
	}
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
	case "function_call_output", "custom_tool_call_output", "mcp_tool_call_output":
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
	case "tool_search_output":
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
		statusValue, ok := block["status"]
		if !ok {
			return false
		}
		status, ok := statusValue.(string)
		if !ok || status == "" {
			return false
		}
		executionValue, ok := block["execution"]
		if !ok {
			return false
		}
		execution, ok := executionValue.(string)
		if !ok || execution == "" {
			return false
		}
		_, ok = block["tools"]
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
	case "tool_search_call":
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
		executionValue, ok := block["execution"]
		if !ok {
			return false
		}
		execution, ok := executionValue.(string)
		if !ok || execution == "" {
			return false
		}
		_, ok = block["arguments"]
		return ok
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
	case "local_shell_call":
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
		_, ok = block["action"]
		return ok
	case "image_generation_call":
		if role != "assistant" {
			return false
		}
		idValue, ok := block["id"]
		if !ok {
			return false
		}
		id, ok := idValue.(string)
		if !ok || id == "" {
			return false
		}
		statusValue, ok := block["status"]
		if !ok {
			return false
		}
		status, ok := statusValue.(string)
		if !ok || status == "" {
			return false
		}
		resultValue, ok := block["result"]
		if !ok {
			return false
		}
		result, ok := resultValue.(string)
		return ok && result != ""
	default:
		return false
	}
}
