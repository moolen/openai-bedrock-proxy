package proxy

import (
	"context"

	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type BedrockChatAPI interface {
	LookupModel(context.Context, string) (bedrock.ModelRecord, error)
	Chat(context.Context, bedrock.ConverseRequest) (bedrock.ConverseResponse, error)
}

type ChatService struct {
	client BedrockChatAPI
}

func NewChatService(client BedrockChatAPI) *ChatService {
	return &ChatService{client: client}
}

func (s *ChatService) Complete(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	if err := openai.ValidateChatCompletionRequest(req); err != nil {
		return openai.ChatCompletionResponse{}, err
	}

	record, err := s.client.LookupModel(ctx, req.Model)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}

	translated, err := bedrock.TranslateChatRequest(req, record)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}

	resp, err := s.client.Chat(ctx, translated)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}

	return bedrock.TranslateChatResponse(resp, req.Model), nil
}
