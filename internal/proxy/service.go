package proxy

import (
	"context"
	"net/http"

	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type BedrockConversation interface {
	RespondConversation(ctx context.Context, modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64) (bedrock.ConverseResponse, error)
	StreamConversation(ctx context.Context, modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64, w http.ResponseWriter) (bedrock.ConverseResponse, error)
	ListModels(context.Context) ([]bedrock.ModelSummary, error)
}

type Service struct {
	client BedrockConversation
	store  conversation.Store
}

func NewService(client BedrockConversation, store conversation.Store) *Service {
	return &Service{client: client, store: store}
}

func (s *Service) Respond(ctx context.Context, req openai.ResponsesRequest) (openai.Response, error) {
	base, err := s.loadPrevious(req.PreviousResponseID)
	if err != nil {
		return openai.Response{}, err
	}
	normalized, err := conversation.NormalizeRequest(req)
	if err != nil {
		return openai.Response{}, err
	}
	merged := conversation.Merge(base, normalized)

	resp, err := s.client.RespondConversation(ctx, req.Model, merged, req.MaxOutputTokens, req.Temperature)
	if err != nil {
		return openai.Response{}, err
	}
	response := bedrock.TranslateResponse(resp, req.Model)
	snapshot := conversation.AppendAssistantReply(merged, resp.Text)
	s.store.Save(conversation.RecordFromResponse(response.ID, req.Model, snapshot))
	return response, nil
}

func (s *Service) Stream(ctx context.Context, req openai.ResponsesRequest, w http.ResponseWriter) error {
	base, err := s.loadPrevious(req.PreviousResponseID)
	if err != nil {
		return err
	}
	normalized, err := conversation.NormalizeRequest(req)
	if err != nil {
		return err
	}
	merged := conversation.Merge(base, normalized)

	resp, err := s.client.StreamConversation(ctx, req.Model, merged, req.MaxOutputTokens, req.Temperature, w)
	if err != nil {
		return err
	}

	snapshot := conversation.AppendAssistantReply(merged, resp.Text)
	s.store.Save(conversation.RecordFromResponse(openAIResponseID(resp.ResponseID), req.Model, snapshot))
	return nil
}

func (s *Service) ListModels(ctx context.Context) (openai.ModelsList, error) {
	models, err := s.client.ListModels(ctx)
	if err != nil {
		return openai.ModelsList{}, err
	}

	data := make([]openai.Model, 0, len(models))
	for _, model := range models {
		owner := model.Provider
		if owner == "" {
			owner = "bedrock"
		}
		data = append(data, openai.Model{
			ID:      model.ID,
			Object:  "model",
			OwnedBy: owner,
			Name:    model.Name,
		})
	}

	return openai.ModelsList{
		Object: "list",
		Data:   data,
	}, nil
}

func (s *Service) loadPrevious(previousResponseID string) (conversation.Request, error) {
	if previousResponseID == "" {
		return conversation.Request{}, nil
	}
	record, ok := s.store.Get(previousResponseID)
	if !ok {
		return conversation.Request{}, openai.NewInvalidRequestError("unknown previous_response_id")
	}
	return conversation.Request{Messages: record.Messages}, nil
}

func openAIResponseID(bedrockResponseID string) string {
	return "resp_" + bedrockResponseID
}
