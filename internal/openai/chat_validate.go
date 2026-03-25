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
	if err := validateChatStop(req.Stop); err != nil {
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
	if message.Role != "assistant" && len(message.ToolCalls) > 0 {
		return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].tool_calls is only allowed for assistant messages")
	}
	if message.Role == "assistant" && len(message.ToolCalls) > 0 {
		if err := validateAssistantToolCalls(index, message.ToolCalls); err != nil {
			return err
		}
	}

	switch message.Content.Kind {
	case ChatMessageContentKindUnset:
		if message.Role == "assistant" && len(message.ToolCalls) > 0 {
			break
		}
		return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is required")
	case ChatMessageContentKindText:
		if message.Content.Text == "" {
			if message.Role == "assistant" && len(message.ToolCalls) > 0 {
				break
			}
			return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is required")
		}
	case ChatMessageContentKindParts:
		if err := validateChatMessageParts(index, message.Content.Parts); err != nil {
			return err
		}
	default:
		return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is invalid")
	}

	if message.Role == "tool" && message.ToolCallID == "" {
		return NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].tool_call_id is required")
	}

	return nil
}

func validateAssistantToolCalls(messageIndex int, toolCalls []ChatToolCall) error {
	for toolCallIndex, toolCall := range toolCalls {
		path := "messages[" + strconv.Itoa(messageIndex) + "].tool_calls[" + strconv.Itoa(toolCallIndex) + "]"
		if toolCall.ID == "" {
			return NewInvalidRequestError(path + ".id is required")
		}
		if toolCall.Type != "function" {
			return NewInvalidRequestError(path + ".type is invalid")
		}
		if toolCall.Function.Name == "" {
			return NewInvalidRequestError(path + ".function.name is required")
		}
	}
	return nil
}

func validateChatMessageParts(messageIndex int, parts []ChatMessageContentPart) error {
	if len(parts) == 0 {
		return NewInvalidRequestError("messages[" + strconv.Itoa(messageIndex) + "].content is invalid")
	}
	for partIndex, part := range parts {
		if err := validateChatMessagePart(messageIndex, partIndex, part); err != nil {
			return err
		}
	}
	return nil
}

func validateChatMessagePart(messageIndex int, partIndex int, part ChatMessageContentPart) error {
	basePath := "messages[" + strconv.Itoa(messageIndex) + "].content[" + strconv.Itoa(partIndex) + "]"
	switch part.Type {
	case "text":
		if part.Text == "" {
			return NewInvalidRequestError(basePath + ".text is required")
		}
		return nil
	case "image_url":
		if part.ImageURL == nil {
			return NewInvalidRequestError(basePath + ".image_url.url is required")
		}
		urlValue, ok := part.ImageURL["url"]
		if !ok {
			return NewInvalidRequestError(basePath + ".image_url.url is required")
		}
		url, ok := urlValue.(string)
		if !ok || url == "" {
			return NewInvalidRequestError(basePath + ".image_url.url is required")
		}
		return nil
	default:
		return NewInvalidRequestError(basePath + ".type is invalid")
	}
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

func validateChatStop(stop ChatStop) error {
	switch stop.Kind {
	case ChatStopKindUnset, ChatStopKindString, ChatStopKindStrings:
		return nil
	default:
		return NewInvalidRequestError("stop is invalid")
	}
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
