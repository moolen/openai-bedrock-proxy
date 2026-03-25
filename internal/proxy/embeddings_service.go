package proxy

import (
	"context"

	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type BedrockEmbeddingsAPI interface {
	LookupModel(context.Context, string) (bedrock.ModelRecord, error)
	Embed(context.Context, openai.EmbeddingsRequest, bedrock.ModelRecord) (openai.EmbeddingsResponse, error)
}

type EmbeddingsService struct {
	client BedrockEmbeddingsAPI
}

func NewEmbeddingsService(client BedrockEmbeddingsAPI) *EmbeddingsService {
	return &EmbeddingsService{client: client}
}

func (s *EmbeddingsService) Create(ctx context.Context, req openai.EmbeddingsRequest) (openai.EmbeddingsResponse, error) {
	if err := openai.ValidateEmbeddingsRequest(req); err != nil {
		return openai.EmbeddingsResponse{}, err
	}

	record, err := s.client.LookupModel(ctx, req.Model)
	if err != nil {
		return openai.EmbeddingsResponse{}, err
	}

	return s.client.Embed(ctx, req, record)
}
