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

var promptCachingEnabledByDefault bool

func SetPromptCachingEnabledByDefault(enabled bool) {
	promptCachingEnabledByDefault = enabled
}

type PromptCachingConfig struct {
	System   *bool
	Messages *bool
}

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
	if req.TopP != nil {
		translatedTopP := float32(*req.TopP)
		out.TopP = &translatedTopP
	}
	out.StopSequences = normalizeChatStopSequences(req.Stop)
	if err := validateChatReasoningControlConflicts(req); err != nil {
		return ConverseRequest{}, err
	}

	promptCaching, additionalFields, err := consumePromptCaching(req.ExtraBody)
	if err != nil {
		return ConverseRequest{}, err
	}
	if len(additionalFields) > 0 {
		out.AdditionalModelRequestFields = additionalFields
		if _, ok := additionalFields["thinking"]; ok && supportsThinkingTopPConflict(model) {
			out.TopP = nil
		}
	}

	if err := applyReasoning(&out, req, model); err != nil {
		return ConverseRequest{}, err
	}
	applyPromptCaching(&out, promptCaching, model, promptCachingEnabledByDefault)

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

func consumePromptCaching(extra map[string]any) (PromptCachingConfig, map[string]any, error) {
	if len(extra) == 0 {
		return PromptCachingConfig{}, nil, nil
	}

	remaining := make(map[string]any, len(extra))
	for key, value := range extra {
		remaining[key] = value
	}

	rawPromptCaching, ok := remaining["prompt_caching"]
	if !ok {
		return PromptCachingConfig{}, remaining, nil
	}
	delete(remaining, "prompt_caching")

	configMap, ok := rawPromptCaching.(map[string]any)
	if !ok {
		return PromptCachingConfig{}, nil, openai.NewInvalidRequestError("extra_body.prompt_caching is invalid")
	}

	var cfg PromptCachingConfig
	if rawSystem, ok := configMap["system"]; ok {
		value, ok := rawSystem.(bool)
		if !ok {
			return PromptCachingConfig{}, nil, openai.NewInvalidRequestError("extra_body.prompt_caching.system is invalid")
		}
		cfg.System = &value
	}
	if rawMessages, ok := configMap["messages"]; ok {
		value, ok := rawMessages.(bool)
		if !ok {
			return PromptCachingConfig{}, nil, openai.NewInvalidRequestError("extra_body.prompt_caching.messages is invalid")
		}
		cfg.Messages = &value
	}

	if len(remaining) == 0 {
		remaining = nil
	}
	return cfg, remaining, nil
}

func applyReasoning(out *ConverseRequest, req openai.ChatCompletionRequest, model ModelRecord) error {
	effort, err := normalizeReasoningEffort(req.ReasoningEffort)
	if err != nil {
		return err
	}
	if effort == "" {
		return nil
	}

	switch {
	case supportsAnthropicReasoning(model):
		maxTokens := resolvedChatMaxTokens(req)
		if maxTokens == nil {
			return openai.NewInvalidRequestError("reasoning_effort requires max_tokens or max_completion_tokens")
		}
		ensureAdditionalModelRequestFields(out)["reasoning_config"] = map[string]any{
			"type":          "enabled",
			"budget_tokens": calcReasoningBudgetTokens(*maxTokens, effort),
		}
		out.TopP = nil
	case supportsDeepSeekV3Reasoning(model):
		ensureAdditionalModelRequestFields(out)["reasoning_config"] = effort
	default:
		return openai.NewInvalidRequestError("reasoning_effort is not supported by this model")
	}
	return nil
}

func ensureAdditionalModelRequestFields(out *ConverseRequest) map[string]any {
	if out.AdditionalModelRequestFields == nil {
		out.AdditionalModelRequestFields = map[string]any{}
	}
	return out.AdditionalModelRequestFields
}

func calcReasoningBudgetTokens(maxTokens int, effort string) int {
	switch effort {
	case "low":
		return int(float64(maxTokens) * 0.3)
	case "medium":
		return int(float64(maxTokens) * 0.6)
	default:
		return maxTokens - 1
	}
}

func applyPromptCaching(req *ConverseRequest, cfg PromptCachingConfig, model ModelRecord, enabledByDefault bool) {
	if !supportsPromptCaching(model) {
		return
	}

	systemEnabled := enabledByDefault
	if cfg.System != nil {
		systemEnabled = *cfg.System
	}
	if systemEnabled && len(req.System) > 0 {
		req.SystemCachePoint = true
	}

	messagesEnabled := enabledByDefault
	if cfg.Messages != nil {
		messagesEnabled = *cfg.Messages
	}
	if !messagesEnabled {
		return
	}

	for idx := len(req.Messages) - 1; idx >= 0; idx-- {
		if req.Messages[idx].Role != "user" || len(req.Messages[idx].Content) == 0 {
			continue
		}
		req.Messages[idx].Content = append(req.Messages[idx].Content, ContentBlock{
			CachePoint: &CachePointBlock{Type: "default"},
		})
		return
	}
}

func normalizeChatStopSequences(stop openai.ChatStop) []string {
	switch stop.Kind {
	case openai.ChatStopKindString:
		return []string{stop.Value}
	case openai.ChatStopKindStrings:
		return append([]string(nil), stop.Values...)
	default:
		return nil
	}
}

func validateChatReasoningControlConflicts(req openai.ChatCompletionRequest) error {
	if strings.TrimSpace(req.ReasoningEffort) == "" || len(req.ExtraBody) == 0 {
		return nil
	}
	if _, ok := req.ExtraBody["thinking"]; ok {
		return openai.NewInvalidRequestError("reasoning_effort cannot be combined with provider-specific reasoning controls")
	}
	if _, ok := req.ExtraBody["reasoning_config"]; ok {
		return openai.NewInvalidRequestError("reasoning_effort cannot be combined with provider-specific reasoning controls")
	}
	return nil
}

func supportsPromptCaching(model ModelRecord) bool {
	resolved := strings.ToLower(resolvedModelID(model))
	if strings.Contains(resolved, "anthropic.claude") {
		excluded := []string{"claude-instant", "claude-v1", "claude-v2"}
		for _, pattern := range excluded {
			if strings.Contains(resolved, pattern) {
				return false
			}
		}
		return true
	}
	return strings.Contains(resolved, "amazon.nova")
}

func supportsAnthropicReasoning(model ModelRecord) bool {
	resolved := strings.ToLower(resolvedModelID(model))
	if !strings.Contains(resolved, "anthropic.claude") {
		return false
	}
	excluded := []string{"claude-instant", "claude-v1", "claude-v2"}
	for _, pattern := range excluded {
		if strings.Contains(resolved, pattern) {
			return false
		}
	}
	return true
}

func supportsDeepSeekV3Reasoning(model ModelRecord) bool {
	resolved := strings.ToLower(resolvedModelID(model))
	return strings.Contains(resolved, "deepseek.v3") || strings.Contains(resolved, "deepseek.deepseek-v3")
}

func supportsThinkingTopPConflict(model ModelRecord) bool {
	resolved := strings.ToLower(resolvedModelID(model))
	conflicts := []string{
		"claude-sonnet-4-5",
		"claude-haiku-4-5",
		"claude-opus-4-5",
	}
	for _, pattern := range conflicts {
		if strings.Contains(resolved, pattern) {
			return true
		}
	}
	return false
}

func normalizeReasoningEffort(value string) (string, error) {
	switch strings.TrimSpace(value) {
	case "":
		return "", nil
	case "low", "medium", "high":
		return strings.TrimSpace(value), nil
	default:
		return "", openai.NewInvalidRequestError("reasoning_effort is invalid")
	}
}

func resolvedModelID(model ModelRecord) string {
	if model.ResolvedFoundationModelID != "" {
		return model.ResolvedFoundationModelID
	}
	return model.ID
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
		Usage: chatCompletionUsage(resp.Usage),
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
