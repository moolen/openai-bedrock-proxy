package bedrock

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsmiddleware "github.com/aws/aws-sdk-go-v2/aws/middleware"
	"github.com/aws/aws-sdk-go-v2/config"
	bedrockcatalog "github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrockcatalogtypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	bedrockdocument "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	applog "github.com/moolen/openai-bedrock-proxy/internal/logging"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type RuntimeAPI interface {
	Converse(context.Context, *bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
	ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error)
}

type CatalogAPI interface {
	ListFoundationModels(context.Context, *bedrockcatalog.ListFoundationModelsInput, ...func(*bedrockcatalog.Options)) (*bedrockcatalog.ListFoundationModelsOutput, error)
}

type LoadConfigFunc func(context.Context, ...func(*config.LoadOptions) error) (aws.Config, error)

type Client struct {
	runtime       RuntimeAPI
	catalog       CatalogAPI
	streamAdapter streamAdapterFunc
}

type ModelSummary struct {
	ID       string
	Name     string
	Provider string
}

type streamEvents interface {
	Events() <-chan bedrocktypes.ConverseStreamOutput
	Close() error
	Err() error
}

type streamAdapterFunc func(*bedrockruntime.ConverseStreamOutput) (streamEvents, error)

func NewClient(ctx context.Context, region string, loadConfig LoadConfigFunc) (*Client, error) {
	logger := bedrockLogger(ctx).With("aws_region", region)
	if loadConfig == nil {
		logger.Error("load config function is required")
		return nil, errors.New("load config function is required")
	}

	opts := []func(*config.LoadOptions) error{}
	if region != "" {
		opts = append(opts, config.WithRegion(region))
	}

	logger.Debug("loading aws config for bedrock client")
	awsCfg, err := loadConfig(ctx, opts...)
	if err != nil {
		logger.Error("failed to load aws config", "error", err)
		return nil, err
	}

	logger.Info("initialized bedrock runtime client", "resolved_region", awsCfg.Region)
	return &Client{
		runtime: bedrockruntime.NewFromConfig(awsCfg),
		catalog: bedrockcatalog.NewFromConfig(awsCfg),
	}, nil
}

func (c *Client) Converse(ctx context.Context, input *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	return c.runtime.Converse(ctx, input, optFns...)
}

func (c *Client) ConverseStream(ctx context.Context, input *bedrockruntime.ConverseStreamInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error) {
	return c.runtime.ConverseStream(ctx, input, optFns...)
}

func (c *Client) ListModels(ctx context.Context) ([]ModelSummary, error) {
	if c.catalog == nil {
		return nil, errors.New("bedrock catalog client is not configured")
	}

	resp, err := c.catalog.ListFoundationModels(ctx, &bedrockcatalog.ListFoundationModelsInput{
		ByOutputModality: bedrockcatalogtypes.ModelModalityText,
	})
	if err != nil {
		return nil, err
	}

	models := make([]ModelSummary, 0, len(resp.ModelSummaries))
	for _, model := range resp.ModelSummaries {
		modelID := aws.ToString(model.ModelId)
		if modelID == "" {
			continue
		}
		models = append(models, ModelSummary{
			ID:       modelID,
			Name:     aws.ToString(model.ModelName),
			Provider: aws.ToString(model.ProviderName),
		})
	}
	return models, nil
}

func (c *Client) RespondConversation(ctx context.Context, modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64) (ConverseResponse, error) {
	logger := bedrockLogger(ctx).With("model_id", modelID)
	logger.Debug("preparing bedrock converse request",
		"system", req.System,
		"messages", req.Messages,
		"max_output_tokens", maxOutputTokens,
		"temperature", temperature,
	)

	translated, err := TranslateConversation(modelID, req, maxOutputTokens, temperature)
	if err != nil {
		logger.Error("failed to translate conversation for converse", "error", err)
		return ConverseResponse{}, err
	}

	logger.Debug("translated bedrock converse request",
		"system", translated.System,
		"messages", translated.Messages,
		"max_tokens", translated.MaxTokens,
		"translated_temperature", translated.Temperature,
	)

	input := toConverseInput(translated)
	resp, err := c.runtime.Converse(ctx, input)
	if err != nil {
		logger.Error("bedrock converse failed", "error", err)
		return ConverseResponse{}, err
	}

	result := ConverseResponse{
		ResponseID: responseIDFromMetadata(resp),
		Text:       extractText(resp),
	}
	logger.Info("bedrock converse completed", "response_id", result.ResponseID)
	logger.Debug("bedrock converse response text",
		"response_id", result.ResponseID,
		"text", result.Text,
	)
	return result, nil
}

func (c *Client) StreamConversation(ctx context.Context, modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64, w http.ResponseWriter) (ConverseResponse, error) {
	logger := bedrockLogger(ctx).With("model_id", modelID)
	logger.Debug("preparing bedrock converse stream request",
		"system", req.System,
		"messages", req.Messages,
		"max_output_tokens", maxOutputTokens,
		"temperature", temperature,
	)

	translated, err := TranslateConversation(modelID, req, maxOutputTokens, temperature)
	if err != nil {
		logger.Error("failed to translate conversation for stream", "error", err)
		return ConverseResponse{}, err
	}

	logger.Debug("translated bedrock converse stream request",
		"system", translated.System,
		"messages", translated.Messages,
		"max_tokens", translated.MaxTokens,
		"translated_temperature", translated.Temperature,
	)

	resp, err := c.runtime.ConverseStream(ctx, toConverseStreamInput(translated))
	if err != nil {
		logger.Error("bedrock converse stream failed", "error", err)
		return ConverseResponse{}, err
	}

	adapter := c.streamAdapter
	if adapter == nil {
		adapter = defaultStreamAdapter
	}
	stream, err := adapter(resp)
	if err != nil {
		logger.Error("failed to adapt bedrock stream", "error", err)
		return ConverseResponse{}, err
	}
	defer stream.Close()

	logger.Info("bedrock stream started")
	text, stopReason, err := processStream(stream, w, logger)
	result := ConverseResponse{
		ResponseID: responseIDFromStreamMetadata(resp),
		Text:       text,
		StopReason: stopReason,
	}
	if err != nil {
		logger.Error("bedrock stream processing failed",
			"response_id", result.ResponseID,
			"stop_reason", stopReason,
			"error", err,
		)
		return result, err
	}

	logger.Info("bedrock stream completed",
		"response_id", result.ResponseID,
		"stop_reason", stopReason,
	)
	logger.Debug("bedrock stream final text",
		"response_id", result.ResponseID,
		"text", text,
	)
	return result, nil
}

func (c *Client) Respond(ctx context.Context, req openai.ResponsesRequest) (openai.Response, error) {
	normalized, err := conversation.NormalizeRequest(req)
	if err != nil {
		return openai.Response{}, err
	}
	resp, err := c.RespondConversation(ctx, req.Model, normalized, req.MaxOutputTokens, req.Temperature)
	if err != nil {
		return openai.Response{}, err
	}
	return TranslateResponse(resp, req.Model), nil
}

func (c *Client) Stream(ctx context.Context, req openai.ResponsesRequest, w http.ResponseWriter) error {
	normalized, err := conversation.NormalizeRequest(req)
	if err != nil {
		return err
	}
	_, err = c.StreamConversation(ctx, req.Model, normalized, req.MaxOutputTokens, req.Temperature, w)
	return err
}

func toConverseInput(req ConverseRequest) *bedrockruntime.ConverseInput {
	return &bedrockruntime.ConverseInput{
		ModelId:         aws.String(req.ModelID),
		Messages:        toBedrockMessages(req.Messages),
		System:          toBedrockSystem(req.System),
		InferenceConfig: toInferenceConfig(req),
		ToolConfig:      toSDKToolConfig(req.ToolConfig),
	}
}

func toConverseStreamInput(req ConverseRequest) *bedrockruntime.ConverseStreamInput {
	return &bedrockruntime.ConverseStreamInput{
		ModelId:         aws.String(req.ModelID),
		Messages:        toBedrockMessages(req.Messages),
		System:          toBedrockSystem(req.System),
		InferenceConfig: toInferenceConfig(req),
		ToolConfig:      toSDKToolConfig(req.ToolConfig),
	}
}

func toBedrockMessages(messages []Message) []bedrocktypes.Message {
	out := make([]bedrocktypes.Message, 0, len(messages))
	for _, message := range messages {
		content := make([]bedrocktypes.ContentBlock, 0, len(message.Content))
		for _, block := range message.Content {
			switch {
			case block.ToolUse != nil:
				content = append(content, &bedrocktypes.ContentBlockMemberToolUse{
					Value: bedrocktypes.ToolUseBlock{
						ToolUseId: aws.String(block.ToolUse.ToolUseID),
						Name:      aws.String(block.ToolUse.Name),
						Input:     bedrockdocument.NewLazyDocument(block.ToolUse.Input),
					},
				})
			case block.ToolResult != nil:
				content = append(content, &bedrocktypes.ContentBlockMemberToolResult{
					Value: bedrocktypes.ToolResultBlock{
						ToolUseId: aws.String(block.ToolResult.ToolUseID),
						Content:   toSDKToolResultContent(block.ToolResult.Content),
					},
				})
			default:
				content = append(content, &bedrocktypes.ContentBlockMemberText{Value: block.Text})
			}
		}
		out = append(out, bedrocktypes.Message{
			Role:    bedrocktypes.ConversationRole(message.Role),
			Content: content,
		})
	}
	return out
}

func toBedrockSystem(system []string) []bedrocktypes.SystemContentBlock {
	out := make([]bedrocktypes.SystemContentBlock, 0, len(system))
	for _, text := range system {
		out = append(out, &bedrocktypes.SystemContentBlockMemberText{Value: text})
	}
	return out
}

func toInferenceConfig(req ConverseRequest) *bedrocktypes.InferenceConfiguration {
	if req.MaxTokens == nil && req.Temperature == nil {
		return nil
	}

	return &bedrocktypes.InferenceConfiguration{
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	}
}

func toSDKToolConfig(config *ToolConfig) *bedrocktypes.ToolConfiguration {
	if config == nil || len(config.Tools) == 0 {
		return nil
	}

	out := &bedrocktypes.ToolConfiguration{
		Tools: make([]bedrocktypes.Tool, 0, len(config.Tools)),
	}
	for _, tool := range config.Tools {
		spec := bedrocktypes.ToolSpecification{
			Name: aws.String(tool.Name),
			InputSchema: &bedrocktypes.ToolInputSchemaMemberJson{
				Value: bedrockdocument.NewLazyDocument(tool.InputSchema),
			},
		}
		if tool.Description != "" {
			spec.Description = aws.String(tool.Description)
		}
		out.Tools = append(out.Tools, &bedrocktypes.ToolMemberToolSpec{
			Value: spec,
		})
	}
	if config.ToolChoice != nil {
		out.ToolChoice = toSDKToolChoice(config.ToolChoice)
	}
	return out
}

func toSDKToolChoice(choice *ToolChoice) bedrocktypes.ToolChoice {
	if choice == nil {
		return nil
	}
	if choice.Auto {
		return &bedrocktypes.ToolChoiceMemberAuto{
			Value: bedrocktypes.AutoToolChoice{},
		}
	}
	if choice.Tool != "" {
		return &bedrocktypes.ToolChoiceMemberTool{
			Value: bedrocktypes.SpecificToolChoice{
				Name: aws.String(choice.Tool),
			},
		}
	}
	return nil
}

func toSDKToolResultContent(content []ToolResultContentBlock) []bedrocktypes.ToolResultContentBlock {
	out := make([]bedrocktypes.ToolResultContentBlock, 0, len(content))
	for _, block := range content {
		switch block.Type {
		case toolResultContentTypeText:
			out = append(out, &bedrocktypes.ToolResultContentBlockMemberText{Value: block.Text})
		case toolResultContentTypeJSON:
			fallthrough
		default:
			out = append(out, &bedrocktypes.ToolResultContentBlockMemberJson{
				Value: bedrockdocument.NewLazyDocument(block.JSON),
			})
		}
	}
	return out
}

func extractText(resp *bedrockruntime.ConverseOutput) string {
	if resp == nil {
		return ""
	}

	output, ok := resp.Output.(*bedrocktypes.ConverseOutputMemberMessage)
	if !ok {
		return ""
	}

	text := ""
	for _, block := range output.Value.Content {
		if textBlock, ok := block.(*bedrocktypes.ContentBlockMemberText); ok {
			text += textBlock.Value
		}
	}
	return text
}

func responseIDFromMetadata(resp *bedrockruntime.ConverseOutput) string {
	if resp != nil {
		if requestID, ok := awsmiddleware.GetRequestIDMetadata(resp.ResultMetadata); ok && requestID != "" {
			return requestID
		}
	}
	return fallbackResponseID()
}

func responseIDFromStreamMetadata(resp *bedrockruntime.ConverseStreamOutput) string {
	if resp != nil {
		if requestID, ok := awsmiddleware.GetRequestIDMetadata(resp.ResultMetadata); ok && requestID != "" {
			return requestID
		}
	}
	return fallbackResponseID()
}

func defaultStreamAdapter(resp *bedrockruntime.ConverseStreamOutput) (streamEvents, error) {
	if resp == nil {
		return nil, errors.New("bedrock stream response was nil")
	}
	stream := resp.GetStream()
	if stream == nil {
		return nil, errors.New("bedrock stream was nil")
	}
	return stream, nil
}

func processStream(stream streamEvents, w http.ResponseWriter, logger *slog.Logger) (string, string, error) {
	var accumulator TextAccumulator
	stopReason := ""

	for event := range stream.Events() {
		switch typed := event.(type) {
		case *bedrocktypes.ConverseStreamOutputMemberContentBlockDelta:
			textDelta, ok := typed.Value.Delta.(*bedrocktypes.ContentBlockDeltaMemberText)
			if !ok {
				continue
			}
			accumulator.Add(textDelta.Value)
			logger.Debug("bedrock stream text delta", "delta", textDelta.Value)
			if err := openai.WriteEvent(w, "response.output_text.delta", map[string]any{
				"delta": textDelta.Value,
			}); err != nil {
				logger.Error("failed to write text delta event", "error", err)
				return "", "", err
			}
		case *bedrocktypes.ConverseStreamOutputMemberMessageStop:
			stopReason = string(typed.Value.StopReason)
			logger.Debug("bedrock stream stop", "stop_reason", stopReason)
			if err := openai.WriteEvent(w, "response.completed", map[string]any{
				"status": stopReason,
			}); err != nil {
				logger.Error("failed to write response completed event", "error", err)
				return "", "", err
			}
		}
	}

	return accumulator.Text(), stopReason, stream.Err()
}

func fallbackResponseID() string {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "local"
	}
	return fmt.Sprintf("%x", buf[:])
}

func bedrockLogger(ctx context.Context) *slog.Logger {
	return applog.FromContext(ctx).With("component", "bedrock")
}
