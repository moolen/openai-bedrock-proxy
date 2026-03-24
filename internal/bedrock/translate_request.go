package bedrock

import (
	"math"

	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
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

func TranslateConversation(modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64) (ConverseRequest, error) {
	out := ConverseRequest{
		ModelID:  modelID,
		System:   append([]string(nil), req.System...),
		Messages: toBedrockConversationMessages(req.Messages),
	}
	if maxOutputTokens != nil {
		if *maxOutputTokens < 0 || *maxOutputTokens > math.MaxInt32 {
			return ConverseRequest{}, openai.NewInvalidRequestError("max_output_tokens is out of range")
		}
		maxTokens := int32(*maxOutputTokens)
		out.MaxTokens = &maxTokens
	}
	if temperature != nil {
		temperature32 := float32(*temperature)
		out.Temperature = &temperature32
	}
	return out, nil
}

func TranslateRequest(req openai.ResponsesRequest) (ConverseRequest, error) {
	normalized, err := conversation.NormalizeRequest(req)
	if err != nil {
		return ConverseRequest{}, err
	}
	return TranslateConversation(req.Model, normalized, req.MaxOutputTokens, req.Temperature)
}

func toBedrockConversationMessages(messages []conversation.Message) []Message {
	out := make([]Message, 0, len(messages))
	for _, message := range messages {
		out = append(out, Message{
			Role: message.Role,
			Content: []ContentBlock{
				{Text: message.Text},
			},
		})
	}
	return out
}
