package bedrock

import (
	"fmt"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type ContentBlock struct {
	Text string
}

type Message struct {
	Role    string
	Content []ContentBlock
}

type ToolConfig struct{}

type ConverseRequest struct {
	ModelID     string
	System      []string
	Messages    []Message
	MaxTokens   *int32
	Temperature *float32
	ToolConfig  *ToolConfig
}

func TranslateRequest(req openai.ResponsesRequest) (ConverseRequest, error) {
	msgs, err := normalizeInput(req.Input)
	if err != nil {
		return ConverseRequest{}, err
	}

	out := ConverseRequest{
		ModelID:  req.Model,
		Messages: msgs,
	}
	if req.Instructions != "" {
		out.System = []string{req.Instructions}
	}
	if req.MaxOutputTokens != nil {
		maxTokens := int32(*req.MaxOutputTokens)
		out.MaxTokens = &maxTokens
	}
	if req.Temperature != nil {
		temperature := float32(*req.Temperature)
		out.Temperature = &temperature
	}
	return out, nil
}

func normalizeInput(input any) ([]Message, error) {
	text, ok := input.(string)
	if !ok {
		return nil, openai.NewInvalidRequestError(fmt.Sprintf("unsupported input type %T", input))
	}

	return []Message{
		{
			Role: "user",
			Content: []ContentBlock{
				{Text: text},
			},
		},
	}, nil
}
