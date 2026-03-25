package proxy

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type recordingStore struct {
	records map[string]conversation.Record
	saved   int
	last    conversation.Record
}

func newRecordingStore() *recordingStore {
	return &recordingStore{records: make(map[string]conversation.Record)}
}

func (s *recordingStore) Get(id string) (conversation.Record, bool) {
	record, ok := s.records[id]
	if !ok {
		return conversation.Record{}, false
	}
	record.Messages = append([]conversation.Message(nil), record.Messages...)
	return record, true
}

func (s *recordingStore) Save(record conversation.Record) {
	s.saved++
	record.Messages = append([]conversation.Message(nil), record.Messages...)
	s.records[record.ResponseID] = record
	s.last = record
}

type fakeBedrock struct {
	respondResp bedrock.ConverseResponse
	respondErr  error
	streamResp  bedrock.ConverseResponse
	streamErr   error
	models      []bedrock.ModelSummary
	modelsErr   error

	respondCalls int
	streamCalls  int
	lastModel    string
	lastRequest  conversation.Request
}

func (f *fakeBedrock) RespondConversation(_ context.Context, modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64) (bedrock.ConverseResponse, error) {
	f.respondCalls++
	f.lastModel = modelID
	f.lastRequest = req
	return f.respondResp, f.respondErr
}

func (f *fakeBedrock) StreamConversation(_ context.Context, modelID string, req conversation.Request, maxOutputTokens *int, temperature *float64, w http.ResponseWriter) (bedrock.ConverseResponse, error) {
	f.streamCalls++
	f.lastModel = modelID
	f.lastRequest = req
	if w != nil {
		_, _ = w.Write([]byte(""))
	}
	return f.streamResp, f.streamErr
}

func (f *fakeBedrock) ListModels(context.Context) ([]bedrock.ModelSummary, error) {
	return f.models, f.modelsErr
}

func textMessage(role, text string) conversation.Message {
	return conversation.Message{
		Role: role,
		Blocks: []conversation.Block{
			{Type: conversation.BlockTypeText, Text: text},
		},
	}
}

func TestServiceRespondUsesPreviousResponseSnapshot(t *testing.T) {
	store := newRecordingStore()
	store.Save(conversation.Record{
		ResponseID: "resp_prev",
		ModelID:    "old-model",
		Messages: []conversation.Message{
			textMessage("user", "hi"),
			textMessage("assistant", "hello"),
		},
	})
	client := &fakeBedrock{respondResp: bedrock.ConverseResponse{
		ResponseID: "1",
		Output: []bedrock.OutputBlock{
			{Type: bedrock.OutputBlockTypeText, Text: "ok"},
		},
	}}
	svc := NewService(client, store)

	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model:              "model",
		Input:              "next",
		PreviousResponseID: "resp_prev",
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got := client.lastRequest.Messages
	if len(got) != 3 {
		t.Fatalf("expected 3 merged messages, got %d", len(got))
	}
	if got[0].Role != "user" || got[0].Blocks[0].Text != "hi" {
		t.Fatalf("expected first message to be previous user, got %#v", got[0])
	}
	if got[1].Role != "assistant" || got[1].Blocks[0].Text != "hello" {
		t.Fatalf("expected second message to be previous assistant, got %#v", got[1])
	}
	if got[2].Role != "user" || got[2].Blocks[0].Text != "next" {
		t.Fatalf("expected third message to be current user, got %#v", got[2])
	}
}

func TestServiceRespondRebuildsTurnLocalSystemContext(t *testing.T) {
	store := newRecordingStore()
	store.Save(conversation.Record{
		ResponseID: "resp_prev",
		ModelID:    "old-model",
		Messages: []conversation.Message{
			textMessage("user", "hi"),
			textMessage("assistant", "hello"),
		},
	})
	client := &fakeBedrock{respondResp: bedrock.ConverseResponse{
		ResponseID: "1",
		Output: []bedrock.OutputBlock{
			{Type: bedrock.OutputBlockTypeText, Text: "ok"},
		},
	}}
	svc := NewService(client, store)

	input := []any{
		map[string]any{"role": "system", "content": "sys"},
		map[string]any{"role": "user", "content": "next"},
	}
	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model:              "model",
		Input:              input,
		Instructions:       "rules",
		PreviousResponseID: "resp_prev",
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	got := client.lastRequest.System
	if len(got) != 2 {
		t.Fatalf("expected 2 system entries, got %d", len(got))
	}
	if got[0] != "sys" || got[1] != "rules" {
		t.Fatalf("expected system to be rebuilt from current request, got %#v", got)
	}
}

func TestServiceRespondPersistsAssistantReplyInSnapshot(t *testing.T) {
	store := newRecordingStore()
	store.records["resp_prev"] = conversation.Record{
		ResponseID: "resp_prev",
		ModelID:    "old-model",
		Messages: []conversation.Message{
			textMessage("user", "prior"),
			textMessage("assistant", "context"),
		},
	}
	client := &fakeBedrock{respondResp: bedrock.ConverseResponse{
		ResponseID: "abc",
		Output: []bedrock.OutputBlock{
			{Type: bedrock.OutputBlockTypeText, Text: "assistant"},
		},
	}}
	svc := NewService(client, store)

	resp, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model:              "model",
		Input:              "hi",
		PreviousResponseID: "resp_prev",
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if store.saved != 1 {
		t.Fatalf("expected 1 saved record, got %d", store.saved)
	}
	if store.last.ResponseID != resp.ID {
		t.Fatalf("expected record to use response id %q, got %q", resp.ID, store.last.ResponseID)
	}
	if len(store.last.Messages) != 4 {
		t.Fatalf("expected 4 messages in snapshot, got %d", len(store.last.Messages))
	}
	if store.last.Messages[0].Role != "user" || store.last.Messages[0].Blocks[0].Text != "prior" {
		t.Fatalf("expected prior user message to be persisted, got %#v", store.last.Messages[0])
	}
	if store.last.Messages[1].Role != "assistant" || store.last.Messages[1].Blocks[0].Text != "context" {
		t.Fatalf("expected prior assistant message to be persisted, got %#v", store.last.Messages[1])
	}
	if store.last.Messages[2].Role != "user" || store.last.Messages[2].Blocks[0].Text != "hi" {
		t.Fatalf("expected current user message to be persisted, got %#v", store.last.Messages[2])
	}
	if store.last.Messages[3].Role != "assistant" || store.last.Messages[3].Blocks[0].Text != "assistant" {
		t.Fatalf("expected assistant reply to be persisted, got %#v", store.last.Messages[3])
	}
}

func TestServiceRespondPersistsAssistantToolCallInSnapshot(t *testing.T) {
	store := newRecordingStore()
	client := &fakeBedrock{respondResp: bedrock.ConverseResponse{
		ResponseID: "abc",
		Output: []bedrock.OutputBlock{
			{
				Type: bedrock.OutputBlockTypeToolCall,
				ToolCall: &bedrock.ToolCall{
					ID:        "call_1",
					Name:      "lookup",
					Arguments: `{"q":"weather"}`,
				},
			},
		},
	}}
	svc := NewService(client, store)

	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []openai.Tool{
			{
				Type: "function",
				Function: &openai.ToolFunction{
					Name: "lookup",
					Parameters: map[string]any{
						"type": "object",
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(client.lastRequest.Tools) != 1 || client.lastRequest.Tools[0].Name != "lookup" {
		t.Fatalf("expected tool definitions to reach bedrock, got %#v", client.lastRequest.Tools)
	}
	if len(store.last.Messages) != 2 {
		t.Fatalf("expected user turn and assistant tool call, got %#v", store.last.Messages)
	}
	if store.last.Messages[1].Role != "assistant" {
		t.Fatalf("expected assistant message, got %#v", store.last.Messages[1])
	}
	if len(store.last.Messages[1].Blocks) != 1 || store.last.Messages[1].Blocks[0].ToolCall == nil {
		t.Fatalf("expected assistant tool call block, got %#v", store.last.Messages[1].Blocks)
	}
	if store.last.Messages[1].Blocks[0].ToolCall.Name != "lookup" {
		t.Fatalf("expected tool call name to persist, got %#v", store.last.Messages[1].Blocks[0].ToolCall)
	}
}

func TestServiceRespondMergesToolResultFollowUpWithPreviousToolCall(t *testing.T) {
	store := newRecordingStore()
	store.records["resp_prev"] = conversation.Record{
		ResponseID: "resp_prev",
		ModelID:    "old-model",
		Messages: []conversation.Message{
			textMessage("user", "use a tool"),
			{
				Role: "assistant",
				Blocks: []conversation.Block{
					{
						Type: conversation.BlockTypeToolCall,
						ToolCall: &conversation.ToolCall{
							ID:        "call_1",
							Name:      "lookup",
							Arguments: `{"q":"weather"}`,
						},
					},
				},
			},
		},
	}
	client := &fakeBedrock{respondResp: bedrock.ConverseResponse{
		ResponseID: "next",
		Output: []bedrock.OutputBlock{
			{Type: bedrock.OutputBlockTypeText, Text: "thanks"},
		},
	}}
	svc := NewService(client, store)

	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model:              "model",
		PreviousResponseID: "resp_prev",
		Input: []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type":    "function_call_output",
						"call_id": "call_1",
						"output": map[string]any{
							"answer": "sunny",
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(client.lastRequest.Messages) != 3 {
		t.Fatalf("expected previous turns plus tool result, got %#v", client.lastRequest.Messages)
	}
	last := client.lastRequest.Messages[2]
	if last.Role != "user" || len(last.Blocks) != 1 || last.Blocks[0].ToolResult == nil {
		t.Fatalf("expected merged user tool result block, got %#v", last)
	}
	if last.Blocks[0].ToolResult.CallID != "call_1" {
		t.Fatalf("expected merged tool result call id, got %#v", last.Blocks[0].ToolResult)
	}
	output, ok := last.Blocks[0].ToolResult.Output.(map[string]any)
	if !ok || output["answer"] != "sunny" {
		t.Fatalf("expected merged tool result payload, got %#v", last.Blocks[0].ToolResult.Output)
	}
}

func TestServiceRespondUsesIncomingModelForContinuation(t *testing.T) {
	store := newRecordingStore()
	store.Save(conversation.Record{
		ResponseID: "resp_prev",
		ModelID:    "old-model",
		Messages: []conversation.Message{
			textMessage("user", "hi"),
		},
	})
	client := &fakeBedrock{respondResp: bedrock.ConverseResponse{
		ResponseID: "1",
		Output: []bedrock.OutputBlock{
			{Type: bedrock.OutputBlockTypeText, Text: "ok"},
		},
	}}
	svc := NewService(client, store)

	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model:              "new-model",
		Input:              "next",
		PreviousResponseID: "resp_prev",
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if client.lastModel != "new-model" {
		t.Fatalf("expected model to come from incoming request, got %q", client.lastModel)
	}
}

func TestServiceRespondDoesNotPersistFailedResponse(t *testing.T) {
	store := newRecordingStore()
	client := &fakeBedrock{respondErr: errors.New("bedrock fail")}
	svc := NewService(client, store)

	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model: "model",
		Input: "hi",
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if store.saved != 0 {
		t.Fatalf("expected no saved record, got %d", store.saved)
	}
}

func TestServiceRespondReturnsInvalidRequestForUnknownPreviousResponseID(t *testing.T) {
	store := newRecordingStore()
	client := &fakeBedrock{}
	svc := NewService(client, store)

	_, err := svc.Respond(context.Background(), openai.ResponsesRequest{
		Model:              "model",
		Input:              "hi",
		PreviousResponseID: "resp_missing",
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var invalid openai.InvalidRequestError
	if !errors.As(err, &invalid) {
		t.Fatalf("expected invalid request error, got %T", err)
	}
	if invalid.Message != "unknown previous_response_id" {
		t.Fatalf("expected canonical message, got %q", invalid.Message)
	}
	if client.respondCalls != 0 {
		t.Fatalf("expected bedrock not to be called, got %d", client.respondCalls)
	}
	if store.saved != 0 {
		t.Fatalf("expected no saved record, got %d", store.saved)
	}
}

func TestServiceStreamPersistsOnlyAfterCleanCompletion(t *testing.T) {
	store := newRecordingStore()
	client := &fakeBedrock{streamResp: bedrock.ConverseResponse{
		ResponseID: "stream",
		Output: []bedrock.OutputBlock{
			{Type: bedrock.OutputBlockTypeText, Text: "done"},
		},
	}}
	svc := NewService(client, store)

	err := svc.Stream(context.Background(), openai.ResponsesRequest{
		Model: "model",
		Input: "hi",
	}, httptest.NewRecorder())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if store.saved != 1 {
		t.Fatalf("expected 1 saved record, got %d", store.saved)
	}
	if store.last.ResponseID != "resp_stream" {
		t.Fatalf("expected stored response id resp_stream, got %q", store.last.ResponseID)
	}
}

func TestServiceStreamDoesNotPersistFailedStream(t *testing.T) {
	store := newRecordingStore()
	client := &fakeBedrock{streamErr: errors.New("stream failed")}
	svc := NewService(client, store)

	err := svc.Stream(context.Background(), openai.ResponsesRequest{
		Model: "model",
		Input: "hi",
	}, httptest.NewRecorder())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if store.saved != 0 {
		t.Fatalf("expected no saved record, got %d", store.saved)
	}
}

func TestServiceListModelsMapsBedrockModelsToOpenAIList(t *testing.T) {
	store := newRecordingStore()
	client := &fakeBedrock{
		models: []bedrock.ModelSummary{
			{ID: "anthropic.claude-3-7-sonnet-20250219-v1:0", Provider: "Anthropic"},
			{ID: "amazon.nova-pro-v1:0", Provider: "Amazon"},
		},
	}
	svc := NewService(client, store)

	got, err := svc.ListModels(context.Background())
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if got.Object != "list" {
		t.Fatalf("expected list object, got %q", got.Object)
	}
	if len(got.Data) != 2 {
		t.Fatalf("expected 2 models, got %d", len(got.Data))
	}
	if got.Data[0].ID != "anthropic.claude-3-7-sonnet-20250219-v1:0" {
		t.Fatalf("expected first model id to map, got %q", got.Data[0].ID)
	}
	if got.Data[0].OwnedBy != "Anthropic" {
		t.Fatalf("expected provider to map to owned_by, got %q", got.Data[0].OwnedBy)
	}
	if got.Data[0].Object != "model" {
		t.Fatalf("expected model object, got %q", got.Data[0].Object)
	}
}

func TestListModelsPreservesOpenAIListShapeWhileExpandingCatalog(t *testing.T) {
	svc := NewService(fakeCatalogClientWithProfiles(), conversation.NewInMemoryStore(16))
	got, err := svc.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Object != "list" || len(got.Data) == 0 {
		t.Fatalf("unexpected list payload: %#v", got)
	}
	first := got.Data[0]
	if first.ID == "" || first.Object != "model" || first.OwnedBy == "" {
		t.Fatalf("unexpected model payload: %#v", first)
	}
}

func fakeCatalogClientWithProfiles() *fakeBedrock {
	return &fakeBedrock{
		models: []bedrock.ModelSummary{
			{
				ID:       "anthropic.claude-3-7-sonnet-20250219-v1:0",
				Name:     "Claude 3.7 Sonnet",
				Provider: "Anthropic",
			},
			{
				ID:       "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
				Name:     "US Claude 3.7 Sonnet Profile",
				Provider: "Anthropic",
			},
		},
	}
}
