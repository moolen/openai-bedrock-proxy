package bedrock

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"net/http"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsmiddleware "github.com/aws/aws-sdk-go-v2/aws/middleware"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type RuntimeAPI interface {
	Converse(context.Context, *bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
	ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error)
}

type LoadConfigFunc func(context.Context, ...func(*config.LoadOptions) error) (aws.Config, error)

type Client struct {
	runtime       RuntimeAPI
	streamAdapter streamAdapterFunc
}

type streamEvents interface {
	Events() <-chan bedrocktypes.ConverseStreamOutput
	Close() error
	Err() error
}

type streamAdapterFunc func(*bedrockruntime.ConverseStreamOutput) (streamEvents, error)

func NewClient(ctx context.Context, region string, loadConfig LoadConfigFunc) (*Client, error) {
	if loadConfig == nil {
		return nil, errors.New("load config function is required")
	}

	opts := []func(*config.LoadOptions) error{}
	if region != "" {
		opts = append(opts, config.WithRegion(region))
	}

	awsCfg, err := loadConfig(ctx, opts...)
	if err != nil {
		return nil, err
	}

	return &Client{
		runtime: bedrockruntime.NewFromConfig(awsCfg),
	}, nil
}

func (c *Client) Converse(ctx context.Context, input *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	return c.runtime.Converse(ctx, input, optFns...)
}

func (c *Client) ConverseStream(ctx context.Context, input *bedrockruntime.ConverseStreamInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error) {
	return c.runtime.ConverseStream(ctx, input, optFns...)
}

func (c *Client) RespondConversation(ctx context.Context, modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64) (ConverseResponse, error) {
	translated, err := TranslateConversation(modelID, req, maxOutputTokens, temperature)
	if err != nil {
		return ConverseResponse{}, err
	}

	input := toConverseInput(translated)

	resp, err := c.runtime.Converse(ctx, input)
	if err != nil {
		return ConverseResponse{}, err
	}

	return ConverseResponse{
		ResponseID: responseIDFromMetadata(resp),
		Text:       extractText(resp),
	}, nil
}

func (c *Client) StreamConversation(ctx context.Context, modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64, w http.ResponseWriter) (ConverseResponse, error) {
	translated, err := TranslateConversation(modelID, req, maxOutputTokens, temperature)
	if err != nil {
		return ConverseResponse{}, err
	}

	resp, err := c.runtime.ConverseStream(ctx, toConverseStreamInput(translated))
	if err != nil {
		return ConverseResponse{}, err
	}
	adapter := c.streamAdapter
	if adapter == nil {
		adapter = defaultStreamAdapter
	}
	stream, err := adapter(resp)
	if err != nil {
		return ConverseResponse{}, err
	}
	defer stream.Close()

	text, stopReason, err := processStream(stream, w)
	return ConverseResponse{
		ResponseID: responseIDFromStreamMetadata(resp),
		Text:       text,
		StopReason: stopReason,
	}, err
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
	}
}

func toConverseStreamInput(req ConverseRequest) *bedrockruntime.ConverseStreamInput {
	return &bedrockruntime.ConverseStreamInput{
		ModelId:         aws.String(req.ModelID),
		Messages:        toBedrockMessages(req.Messages),
		System:          toBedrockSystem(req.System),
		InferenceConfig: toInferenceConfig(req),
	}
}

func toBedrockMessages(messages []Message) []bedrocktypes.Message {
	out := make([]bedrocktypes.Message, 0, len(messages))
	for _, message := range messages {
		content := make([]bedrocktypes.ContentBlock, 0, len(message.Content))
		for _, block := range message.Content {
			content = append(content, &bedrocktypes.ContentBlockMemberText{Value: block.Text})
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

func processStream(stream streamEvents, w http.ResponseWriter) (string, string, error) {
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
			if err := openai.WriteEvent(w, "response.output_text.delta", map[string]any{
				"delta": textDelta.Value,
			}); err != nil {
				return "", "", err
			}
		case *bedrocktypes.ConverseStreamOutputMemberMessageStop:
			stopReason = string(typed.Value.StopReason)
			if err := openai.WriteEvent(w, "response.completed", map[string]any{
				"status": stopReason,
			}); err != nil {
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
