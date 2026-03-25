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
	if err := validateChatToolChoice(req.ToolChoice); err != nil {
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

	switch message.Content.Kind {
	case ChatMessageContentKindUnset:
		return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is required")
	case ChatMessageContentKindText:
		if message.Content.Text == "" {
			return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is required")
		}
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

func validateChatToolChoice(choice ChatToolChoice) error {
	if choice.Kind == ChatToolChoiceKindUnset {
		return nil
	}

	switch choice.Kind {
	case ChatToolChoiceKindString:
		if choice.StringValue == "auto" || choice.StringValue == "required" {
			return nil
		}
		return NewInvalidRequestError("tool_choice is invalid")
	case ChatToolChoiceKindFunction:
		return validateChatToolChoiceObject(choice)
	default:
		return NewInvalidRequestError("tool_choice is invalid")
	}
}

func validateChatToolChoiceObject(choice ChatToolChoice) error {
	if choice.FunctionValue == nil {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	if choice.FunctionValue.Type != "function" {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	if choice.FunctionValue.Function.Name == "" {
		return NewInvalidRequestError("tool_choice is invalid")
	}
	return nil
}
