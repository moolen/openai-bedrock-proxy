package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

func TranslateChatRequest(ctx context.Context, req openai.ChatCompletionRequest, model ModelRecord) (ConverseRequest, error) {
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
			systemText, err := systemTextFromChatMessage(ctx, idx, message, model)
			if err != nil {
				return ConverseRequest{}, err
			}
			out.System = append(out.System, systemText)
		case "user":
			content, err := chatMessageContentBlocks(ctx, idx, message.Content, false, model, true)
			if err != nil {
				return ConverseRequest{}, err
			}
			out.Messages = append(out.Messages, Message{
				Role:    "user",
				Content: content,
			})
		case "assistant":
			content, err := assistantChatContentBlocks(ctx, idx, message, model)
			if err != nil {
				return ConverseRequest{}, err
			}
			out.Messages = append(out.Messages, Message{
				Role:    "assistant",
				Content: content,
			})
		case "tool":
			content, err := toolResultContentBlocks(ctx, idx, message, model)
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

	toolChoice, err := normalizeChatToolChoice(req.ToolChoice)
	if err != nil {
		return ConverseRequest{}, err
	}

	toolConfig, err := toToolConfig(normalizeChatTools(req.Tools), toolChoice)
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

func systemTextFromChatMessage(ctx context.Context, index int, message openai.ChatMessage, model ModelRecord) (string, error) {
	content, err := chatMessageContentBlocks(ctx, index, message.Content, false, model, false)
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

func assistantChatContentBlocks(ctx context.Context, index int, message openai.ChatMessage, model ModelRecord) ([]ContentBlock, error) {
	content, err := chatMessageContentBlocks(ctx, index, message.Content, true, model, false)
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

func toolResultContentBlocks(ctx context.Context, index int, message openai.ChatMessage, model ModelRecord) ([]ContentBlock, error) {
	if strings.TrimSpace(message.ToolCallID) == "" {
		return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].tool_call_id is required")
	}

	content, err := chatMessageContentBlocks(ctx, index, message.Content, false, model, false)
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

func chatMessageContentBlocks(ctx context.Context, index int, content openai.ChatMessageContent, allowUnset bool, model ModelRecord, allowImages bool) ([]ContentBlock, error) {
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
			switch part.Type {
			case "text":
				if strings.TrimSpace(part.Text) == "" {
					return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content[" + strconv.Itoa(partIndex) + "].text is required")
				}
				blocks = append(blocks, ContentBlock{Text: part.Text})
			case "image_url":
				if !allowImages {
					return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content[" + strconv.Itoa(partIndex) + "].type is unsupported")
				}
				if !modelSupportsInputModality(model, "IMAGE") {
					return nil, openai.NewInvalidRequestError("multimodal message is not supported by this model")
				}
				image, err := imageContentBlock(ctx, index, partIndex, part)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, image)
			default:
				return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content[" + strconv.Itoa(partIndex) + "].type is unsupported")
			}
		}
		return blocks, nil
	default:
		return nil, openai.NewInvalidRequestError("messages[" + strconv.Itoa(index) + "].content is invalid")
	}
}

func imageContentBlock(ctx context.Context, messageIndex int, partIndex int, part openai.ChatMessageContentPart) (ContentBlock, error) {
	basePath := "messages[" + strconv.Itoa(messageIndex) + "].content[" + strconv.Itoa(partIndex) + "].image_url.url"
	if part.ImageURL == nil {
		return ContentBlock{}, openai.NewInvalidRequestError(basePath + " is required")
	}
	rawURL, ok := part.ImageURL["url"].(string)
	if !ok || strings.TrimSpace(rawURL) == "" {
		return ContentBlock{}, openai.NewInvalidRequestError(basePath + " is required")
	}

	data, contentType, err := ParseImageURL(ctx, rawURL, nil)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return ContentBlock{}, err
		}
		return ContentBlock{}, openai.NewInvalidRequestError(basePath + " is invalid")
	}
	format, err := imageFormatFromContentType(contentType)
	if err != nil {
		return ContentBlock{}, openai.NewInvalidRequestError(basePath + " is invalid")
	}
	return ContentBlock{
		Image: &ImageBlock{
			Format: format,
			Bytes:  data,
		},
	}, nil
}

func modelSupportsInputModality(model ModelRecord, want string) bool {
	for _, modality := range model.InputModalities {
		if strings.EqualFold(strings.TrimSpace(modality), want) {
			return true
		}
	}
	return false
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

func normalizeChatToolChoice(choice openai.ChatToolChoice) (conversation.ToolChoice, error) {
	if choice.Kind == openai.ChatToolChoiceKindUnset {
		return conversation.ToolChoice{}, nil
	}

	switch choice.Kind {
	case openai.ChatToolChoiceKindString:
		switch choice.StringValue {
		case "", "none":
			return conversation.ToolChoice{}, nil
		case "auto":
			return conversation.ToolChoice{Type: "auto"}, nil
		case "required":
			return conversation.ToolChoice{}, openai.NewInvalidRequestError("tool_choice \"required\" is not supported for chat completions")
		default:
			return conversation.ToolChoice{
				Type: choice.StringValue,
				Name: syntheticBuiltInToolName(choice.StringValue),
			}, nil
		}
	case openai.ChatToolChoiceKindFunction:
		if choice.FunctionValue == nil {
			return conversation.ToolChoice{}, nil
		}
		return conversation.ToolChoice{
			Type: "function",
			Name: choice.FunctionValue.Function.Name,
		}, nil
	default:
		return conversation.ToolChoice{}, nil
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
