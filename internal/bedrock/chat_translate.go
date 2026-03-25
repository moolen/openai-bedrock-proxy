package bedrock

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

func TranslateChatRequest(req openai.ChatCompletionRequest, model ModelRecord) (ConverseRequest, error) {
	out := ConverseRequest{
		ModelID:  req.Model,
		System:   make([]string, 0, len(req.Messages)),
		Messages: make([]Message, 0, len(req.Messages)),
	}
	if model.ID != "" {
		out.ModelID = model.ID
	}

	for idx, message := range req.Messages {
		switch message.Role {
		case "system", "developer":
			systemText, err := systemTextFromChatMessage(idx, message)
			if err != nil {
				return ConverseRequest{}, err
			}
			out.System = append(out.System, systemText)
		case "user":
			content, err := chatMessageContentBlocks(idx, message.Content, false)
			if err != nil {
				return ConverseRequest{}, err
			}
			out.Messages = append(out.Messages, Message{
				Role:    "user",
				Content: content,
			})
		case "assistant":
			content, err := assistantChatContentBlocks(idx, message)
			if err != nil {
				return ConverseRequest{}, err
			}
			out.Messages = append(out.Messages, Message{
				Role:    "assistant",
				Content: content,
			})
		case "tool":
			content, err := toolResultContentBlocks(idx, message)
			if err != nil {
				return ConverseRequest{}, err
			}
			out.Messages = append(out.Messages, Message{
				Role:    "user",
				Content: content,
			})
		default:
			return ConverseRequest{}, openai.NewInvalidRequestError("messages[" + strconv.Itoa(idx) + "].role is invalid")
		}
	}

	if maxTokens := resolvedChatMaxTokens(req); maxTokens != nil {
		if *maxTokens <= 0 || *maxTokens > math.MaxInt32 {
			return ConverseRequest{}, openai.NewInvalidRequestError("max_tokens is out of range")
		}
		translatedMaxTokens := int32(*maxTokens)
		out.MaxTokens = &translatedMaxTokens
	}

	if req.Temperature != nil {
		translatedTemperature := float32(*req.Temperature)
		out.Temperature = &translatedTemperature
	}

	toolConfig, err := toToolConfig(normalizeChatTools(req.Tools), normalizeChatToolChoice(req.ToolChoice))
	if err != nil {
		return ConverseRequest{}, err
	}
	out.ToolConfig = toolConfig

	return out, nil
}

func TranslateChatResponse(resp ConverseResponse, model string) openai.ChatCompletionResponse {
	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder
	toolCalls := make([]openai.ChatToolCall, 0, len(resp.Output))

	for _, block := range resp.Output {
		switch block.Type {
		case OutputBlockTypeToolCall:
			if block.ToolCall == nil {
				continue
			}
			toolCalls = append(toolCalls, openai.ChatToolCall{
				ID:   block.ToolCall.ID,
				Type: "function",
				Function: openai.ChatToolCallFunction{
					Name:      block.ToolCall.Name,
					Arguments: normalizeArgumentsString(block.ToolCall.Arguments),
				},
			})
		case OutputBlockTypeReasoning:
			reasoningBuilder.WriteString(block.Text)
		default:
			contentBuilder.WriteString(block.Text)
		}
	}

	message := openai.ChatMessage{
		Role:      "assistant",
		ToolCalls: toolCalls,
	}
	if content := contentBuilder.String(); content != "" {
		message.Content = openai.ChatMessageText(content)
	}
	if reasoning := reasoningBuilder.String(); reasoning != "" {
		message.ReasoningContent = reasoning
	}

	responseID := resp.ResponseID
	if responseID == "" {
		responseID = fallbackResponseID()
	}

	return openai.ChatCompletionResponse{
		ID:      "chatcmpl_" + responseID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openai.ChatCompletionChoice{{
			Index:        0,
			Message:      message,
			FinishReason: mapChatFinishReason(resp.StopReason),
		}},
	}
}

func mapChatFinishReason(stopReason string) string {
	switch stopReason {
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "guardrail_intervened", "content_filtered":
		return "content_filter"
	case "end_turn", "stop_sequence":
		return "stop"
	case "":
		return ""
	default:
		return "stop"
	}
}

func systemTextFromChatMessage(index int, message openai.ChatMessage) (string, error) {
	content, err := chatMessageContentBlocks(index, message.Content, false)
	if err != nil {
		return "", err
	}
	var builder strings.Builder
	for _, block := range content {
		builder.WriteString(block.Text)
	}
	if builder.Len() == 0 {
		return "", openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is required")
	}
	return builder.String(), nil
}

func assistantChatContentBlocks(index int, message openai.ChatMessage) ([]ContentBlock, error) {
	content, err := chatMessageContentBlocks(index, message.Content, true)
	if err != nil {
		return nil, err
	}
	for toolCallIndex, toolCall := range message.ToolCalls {
		if toolCall.Type != "" && toolCall.Type != "function" {
			return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].tool_calls[" + strconv.Itoa(toolCallIndex) + "].type is invalid")
		}
		if strings.TrimSpace(toolCall.ID) == "" {
			return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].tool_calls[" + strconv.Itoa(toolCallIndex) + "].id is required")
		}
		if strings.TrimSpace(toolCall.Function.Name) == "" {
			return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].tool_calls[" + strconv.Itoa(toolCallIndex) + "].function.name is required")
		}

		input, err := parseToolUseInput(toolCall.Function.Arguments)
		if err != nil {
			return nil, err
		}

		content = append(content, ContentBlock{
			ToolUse: &ToolUseBlock{
				ToolUseID: toolCall.ID,
				Name:      toolCall.Function.Name,
				Input:     input,
			},
		})
	}

	if len(content) == 0 {
		return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is required")
	}
	return content, nil
}

func toolResultContentBlocks(index int, message openai.ChatMessage) ([]ContentBlock, error) {
	if strings.TrimSpace(message.ToolCallID) == "" {
		return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].tool_call_id is required")
	}

	content, err := chatMessageContentBlocks(index, message.Content, false)
	if err != nil {
		return nil, err
	}
	resultContent := make([]ToolResultContentBlock, 0, len(content))
	for _, block := range content {
		resultContent = append(resultContent, ToolResultContentBlock{
			Type: toolResultContentTypeText,
			Text: block.Text,
		})
	}

	return []ContentBlock{{
		ToolResult: &ToolResultBlock{
			ToolUseID: message.ToolCallID,
			Content:   resultContent,
		},
	}}, nil
}

func chatMessageContentBlocks(index int, content openai.ChatMessageContent, allowUnset bool) ([]ContentBlock, error) {
	switch content.Kind {
	case openai.ChatMessageContentKindUnset:
		if allowUnset {
			return nil, nil
		}
		return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is required")
	case openai.ChatMessageContentKindText:
		if strings.TrimSpace(content.Text) == "" {
			if allowUnset {
				return nil, nil
			}
			return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is required")
		}
		return []ContentBlock{{Text: content.Text}}, nil
	case openai.ChatMessageContentKindParts:
		if len(content.Parts) == 0 {
			return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is invalid")
		}
		blocks := make([]ContentBlock, 0, len(content.Parts))
		for partIndex, part := range content.Parts {
			if part.Type != "text" {
				return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content[" + strconv.Itoa(partIndex) + "].type is unsupported")
			}
			if strings.TrimSpace(part.Text) == "" {
				return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content[" + strconv.Itoa(partIndex) + "].text is required")
			}
			blocks = append(blocks, ContentBlock{Text: part.Text})
		}
		return blocks, nil
	default:
		return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is invalid")
	}
}

func normalizeChatTools(tools []openai.Tool) []conversation.ToolDefinition {
	if len(tools) == 0 {
		return nil
	}

	normalized := make([]conversation.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		if tool.Type == "function" {
			definition := conversation.ToolDefinition{
				Type: "function",
			}
			if tool.Function != nil {
				definition.Name = tool.Function.Name
				definition.Description = tool.Function.Description
				definition.Parameters = tool.Function.Parameters
			}
			if definition.Name == "" {
				definition.Name = tool.Name
			}
			normalized = append(normalized, definition)
			continue
		}

		normalized = append(normalized, conversation.ToolDefinition{
			Type:    tool.Type,
			Name:    syntheticBuiltInToolName(tool.Type),
			Config:  cloneChatRawMessageMap(tool.Config),
			BuiltIn: true,
		})
	}
	return normalized
}

func normalizeChatToolChoice(choice openai.ChatToolChoice) conversation.ToolChoice {
	if choice.Kind == openai.ChatToolChoiceKindUnset {
		return conversation.ToolChoice{}
	}

	switch choice.Kind {
	case openai.ChatToolChoiceKindString:
		switch choice.StringValue {
		case "", "none":
			return conversation.ToolChoice{}
		case "auto", "required":
			return conversation.ToolChoice{Type: "auto"}
		default:
			return conversation.ToolChoice{
				Type: choice.StringValue,
				Name: syntheticBuiltInToolName(choice.StringValue),
			}
		}
	case openai.ChatToolChoiceKindFunction:
		if choice.FunctionValue == nil {
			return conversation.ToolChoice{}
		}
		return conversation.ToolChoice{
			Type: "function",
			Name: choice.FunctionValue.Function.Name,
		}
	default:
		return conversation.ToolChoice{}
	}
}

func resolvedChatMaxTokens(req openai.ChatCompletionRequest) *int {
	if req.MaxCompletionTokens != nil {
		return req.MaxCompletionTokens
	}
	return req.MaxTokens
}

func syntheticBuiltInToolName(toolType string) string {
	return "__builtin_" + toolType
}

func cloneChatRawMessageMap(input map[string]json.RawMessage) map[string]json.RawMessage {
	if len(input) == 0 {
		return nil
	}
	cloned := make(map[string]json.RawMessage, len(input))
	for key, value := range input {
		cloned[key] = append(json.RawMessage(nil), value...)
	}
	return cloned
}
