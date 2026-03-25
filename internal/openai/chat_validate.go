package openai

import "strconv"

var supportedChatMessageRoles = map[string]struct{}{
	"system":    {},
	"developer": {},
	"user":      {},
	"assistant": {},
	"tool":      {},
}

func ValidateChatCompletionRequest(req ChatCompletionRequest) error {
	if req.Model == "" {
		return NewInvalidRequestError("model is required")
	}
	if len(req.Messages) == 0 {
		return NewInvalidRequestError("messages is required")
	}

	for idx, message := range req.Messages {
		if err := validateChatMessage(idx, message); err != nil {
			return err
		}
	}

	if err := validateTools(req.Tools); err != nil {
		return err
	}
	if err := validateChatToolChoice(req.ToolChoice, req.Tools); err != nil {
		return err
	}

	if err := validateResolvedMaxTokens(req); err != nil {
		return err
	}

	return nil
}

func validateChatMessage(index int, message ChatMessage) error {
	if _, ok := supportedChatMessageRoles[message.Role]; !ok {
		return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].role is invalid")
	}

	switch content := message.Content.(type) {
	case string:
		if content == "" {
			return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is required")
		}
	case nil:
		return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is required")
	default:
		return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is invalid")
	}

	if message.Role == "tool" && message.ToolCallID == "" {
		return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].tool_call_id is required")
	}

	return nil
}

func validateResolvedMaxTokens(req ChatCompletionRequest) error {
	maxTokens := resolvedMaxTokens(req)
	if maxTokens == nil {
		return nil
	}
	if *maxTokens <= 0 {
		if req.MaxCompletionTokens != nil {
			return NewInvalidRequestError("max_completion_tokens must be greater than 0")
		}
		return NewInvalidRequestError("max_tokens must be greater than 0")
	}
	return nil
}

func resolvedMaxTokens(req ChatCompletionRequest) *int {
	if req.MaxCompletionTokens != nil {
		return req.MaxCompletionTokens
	}
	return req.MaxTokens
}

func validateChatToolChoice(value any, tools []Tool) error {
	if value == nil {
		return nil
	}

	switch choice := value.(type) {
	case string:
		if choice == "auto" || choice == "required" {
			return nil
		}
		return NewInvalidRequestError("tool_choice is invalid")
	case map[string]any:
		return validateChatToolChoiceObject(choice, tools)
	default:
		return NewInvalidRequestError("tool_choice is invalid")
	}
}

func validateChatToolChoiceObject(choice map[string]any, tools []Tool) error {
	if len(choice) != 2 {
		return NewInvalidRequestError("tool_choice is invalid")
	}

	typeValue, ok := choice["type"]
	if !ok {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	typeName, ok := typeValue.(string)
	if !ok || typeName != "function" {
		return NewInvalidRequestError("tool_choice is invalid")
	}

	functionValue, ok := choice["function"]
	if !ok {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	functionChoice, ok := functionValue.(map[string]any)
	if !ok {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	if len(functionChoice) != 1 {
		return NewInvalidRequestError("tool_choice is invalid")
	}

	nameValue, ok := functionChoice["name"]
	if !ok {
		return NewInvalidRequestError("tool_choice.function.name is required")
	}
	name, ok := nameValue.(string)
	if !ok || name == "" {
		return NewInvalidRequestError("tool_choice.function.name is required")
	}

	functionToolNames := collectFunctionToolNames(tools)
	if _, exists := functionToolNames[name]; !exists {
		return NewInvalidRequestError("tool_choice.function.name is not present in tools")
	}

	return nil
}
