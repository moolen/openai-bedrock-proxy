package conversation

import (
	"strings"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

const invalidResponsesInputErrorMessage = "input must be a non-empty string or supported message object/array"

var allowedBlockTypesByRole = map[string]string{
	"user":      "input_text",
	"system":    "input_text",
	"developer": "input_text",
	"assistant": "output_text",
}

func NormalizeRequest(req openai.ResponsesRequest) (Request, error) {
	system, messages, err := normalizeInput(req.Input)
	if err != nil {
		return Request{}, err
	}
	if req.Instructions != "" {
		system = append(system, req.Instructions)
	}
	return Request{
		System:   system,
		Messages: messages,
	}, nil
}

func Merge(base, current Request) Request {
	mergedMessages := make([]Message, 0, len(base.Messages)+len(current.Messages))
	mergedMessages = append(mergedMessages, base.Messages...)
	mergedMessages = append(mergedMessages, current.Messages...)
	return Request{
		System:   append([]string(nil), current.System...),
		Messages: mergedMessages,
	}
}

func AppendAssistantReply(req Request, assistantText string) Request {
	updated := Request{
		System:   append([]string(nil), req.System...),
		Messages: append([]Message(nil), req.Messages...),
	}
	updated.Messages = append(updated.Messages, Message{
		Role: "assistant",
		Text: assistantText,
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
		return nil, []Message{{Role: "user", Text: value}}, nil
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
	allowedBlockType, ok := allowedBlockTypesByRole[role]
	if !ok {
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

	text, err := normalizeContent(allowedBlockType, contentValue)
	if err != nil {
		return nil, nil, err
	}

	switch role {
	case "system", "developer":
		return []string{text}, nil, nil
	case "user", "assistant":
		return nil, []Message{{Role: role, Text: text}}, nil
	default:
		return nil, nil, openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
}

func normalizeContent(allowedBlockType string, content any) (string, error) {
	switch value := content.(type) {
	case string:
		if value == "" {
			return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		return value, nil
	case []map[string]any:
		return normalizeTextBlocks(allowedBlockType, value)
	case []any:
		return normalizeTextBlockItems(allowedBlockType, value)
	default:
		return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
}

func normalizeTextBlocks(allowedBlockType string, blocks []map[string]any) (string, error) {
	if len(blocks) == 0 {
		return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	var builder strings.Builder
	for _, block := range blocks {
		text, err := normalizeTextBlock(allowedBlockType, block)
		if err != nil {
			return "", err
		}
		builder.WriteString(text)
	}
	result := builder.String()
	if result == "" {
		return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	return result, nil
}

func normalizeTextBlockItems(allowedBlockType string, blocks []any) (string, error) {
	if len(blocks) == 0 {
		return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	var builder strings.Builder
	for _, item := range blocks {
		block, ok := item.(map[string]any)
		if !ok {
			return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
		}
		text, err := normalizeTextBlock(allowedBlockType, block)
		if err != nil {
			return "", err
		}
		builder.WriteString(text)
	}
	result := builder.String()
	if result == "" {
		return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	return result, nil
}

func normalizeTextBlock(allowedBlockType string, block map[string]any) (string, error) {
	typeValue, ok := block["type"]
	if !ok {
		return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
	blockType, ok := typeValue.(string)
	if !ok || blockType != allowedBlockType {
		return "", openai.NewInvalidRequestError(invalidResponsesInputErrorMessage)
	}
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
