package proxy

import (
	"context"
	"log/slog"
	"net/http"

	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	applog "github.com/moolen/openai-bedrock-proxy/internal/logging"
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
	logger := proxyLogger(ctx).With(
		"model", req.Model,
		"previous_response_id", req.PreviousResponseID,
	)
	logger.Debug("starting proxy respond",
		"input", req.Input,
		"instructions", req.Instructions,
		"max_output_tokens", req.MaxOutputTokens,
		"temperature", req.Temperature,
	)

	record, err := s.loadPrevious(req.PreviousResponseID)
	if err != nil {
		logger.Warn("failed to load previous response", "error", err)
		return openai.Response{}, err
	}

	base := conversation.Request{Messages: record.Messages}
	if req.PreviousResponseID != "" {
		logger.Debug("loaded previous response snapshot",
			"stored_response_id", record.ResponseID,
			"stored_model", record.ModelID,
			"stored_messages", record.Messages,
			"stored_message_count", len(record.Messages),
		)
		if record.ModelID != "" && record.ModelID != req.Model {
			logger.Debug("incoming model overrides stored model",
				"stored_model", record.ModelID,
				"incoming_model", req.Model,
			)
		}
	}

	normalized, err := conversation.NormalizeRequest(req)
	if err != nil {
		logger.Warn("failed to normalize request", "error", err)
		return openai.Response{}, err
	}
	logger.Debug("normalized request",
		"system", normalized.System,
		"messages", normalized.Messages,
		"system_count", len(normalized.System),
		"message_count", len(normalized.Messages),
	)

	merged := conversation.Merge(base, normalized)
	logger.Debug("merged conversation for bedrock",
		"system", merged.System,
		"messages", merged.Messages,
		"message_count", len(merged.Messages),
	)

	resp, err := s.client.RespondConversation(ctx, req.Model, merged, req.MaxOutputTokens, req.Temperature)
	if err != nil {
		logger.Error("bedrock respond failed", "error", err)
		return openai.Response{}, err
	}

	response := bedrock.TranslateResponse(resp, req.Model)
	snapshot := conversation.AppendAssistantReply(merged, resp.Text)
	s.store.Save(conversation.RecordFromResponse(response.ID, req.Model, snapshot))

	logger.Info("proxy respond completed",
		"response_id", response.ID,
		"bedrock_response_id", resp.ResponseID,
	)
	logger.Debug("persisted response snapshot",
		"response_id", response.ID,
		"assistant_text", resp.Text,
		"snapshot_messages", snapshot.Messages,
		"snapshot_message_count", len(snapshot.Messages),
	)

	return response, nil
}

func (s *Service) Stream(ctx context.Context, req openai.ResponsesRequest, w http.ResponseWriter) error {
	logger := proxyLogger(ctx).With(
		"model", req.Model,
		"previous_response_id", req.PreviousResponseID,
	)
	logger.Debug("starting proxy stream",
		"input", req.Input,
		"instructions", req.Instructions,
		"max_output_tokens", req.MaxOutputTokens,
		"temperature", req.Temperature,
	)

	record, err := s.loadPrevious(req.PreviousResponseID)
	if err != nil {
		logger.Warn("failed to load previous response", "error", err)
		return err
	}

	base := conversation.Request{Messages: record.Messages}
	if req.PreviousResponseID != "" {
		logger.Debug("loaded previous response snapshot",
			"stored_response_id", record.ResponseID,
			"stored_model", record.ModelID,
			"stored_messages", record.Messages,
			"stored_message_count", len(record.Messages),
		)
		if record.ModelID != "" && record.ModelID != req.Model {
			logger.Debug("incoming model overrides stored model",
				"stored_model", record.ModelID,
				"incoming_model", req.Model,
			)
		}
	}

	normalized, err := conversation.NormalizeRequest(req)
	if err != nil {
		logger.Warn("failed to normalize stream request", "error", err)
		return err
	}
	logger.Debug("normalized stream request",
		"system", normalized.System,
		"messages", normalized.Messages,
		"system_count", len(normalized.System),
		"message_count", len(normalized.Messages),
	)

	merged := conversation.Merge(base, normalized)
	logger.Debug("merged conversation for bedrock stream",
		"system", merged.System,
		"messages", merged.Messages,
		"message_count", len(merged.Messages),
	)

	resp, err := s.client.StreamConversation(ctx, req.Model, merged, req.MaxOutputTokens, req.Temperature, w)
	if err != nil {
		logger.Error("bedrock stream failed", "error", err)
		logger.Debug("stream persistence skipped because stream failed")
		return err
	}

	snapshot := conversation.AppendAssistantReply(merged, resp.Text)
	responseID := openAIResponseID(resp.ResponseID)
	s.store.Save(conversation.RecordFromResponse(responseID, req.Model, snapshot))

	logger.Info("proxy stream completed",
		"response_id", responseID,
		"bedrock_response_id", resp.ResponseID,
		"stop_reason", resp.StopReason,
	)
	logger.Debug("persisted streamed response snapshot",
		"response_id", responseID,
		"assistant_text", resp.Text,
		"snapshot_messages", snapshot.Messages,
		"snapshot_message_count", len(snapshot.Messages),
	)

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

func (s *Service) loadPrevious(previousResponseID string) (conversation.Record, error) {
	if previousResponseID == "" {
		return conversation.Record{}, nil
	}
	record, ok := s.store.Get(previousResponseID)
	if !ok {
		return conversation.Record{}, openai.NewInvalidRequestError("unknown previous_response_id")
	}
	return record, nil
}

func openAIResponseID(bedrockResponseID string) string {
	return "resp_" + bedrockResponseID
}

func proxyLogger(ctx context.Context) *slog.Logger {
	return applog.FromContext(ctx).With("component", "proxy")
}
