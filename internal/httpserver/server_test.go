package httpserver

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/smithy-go"
	"github.com/klauspost/compress/zstd"
	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
	"github.com/moolen/openai-bedrock-proxy/internal/proxy"
	"github.com/moolen/openai-bedrock-proxy/internal/session"
)

func TestResponsesHandlerIgnoresAuthorizationHeader(t *testing.T) {
	svc := &fakeService{
		response: openai.Response{ID: "resp_1", Object: "response", Model: "model"},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi"}`))
	req.Header.Set("Authorization", "Bearer ignored")
	rec := httptest.NewRecorder()

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if svc.calls != 1 {
		t.Fatalf("expected service to be called once, got %d", svc.calls)
	}
	if svc.lastRequest.Model != "model" {
		t.Fatalf("expected request model to pass through, got %q", svc.lastRequest.Model)
	}
}

func TestModelsHandlerReturnsOpenAIListShape(t *testing.T) {
	svc := &fakeService{
		models: openai.ModelsList{
			Object: "list",
			Data: []openai.Model{
				{ID: "anthropic.claude-3-7-sonnet-20250219-v1:0", Object: "model", OwnedBy: "bedrock"},
			},
			Models: []openai.CodexModelInfo{
				{
					Slug:                          "anthropic.claude-3-7-sonnet-20250219-v1:0",
					DisplayName:                   "Claude 3.7 Sonnet",
					SupportedReasoningLevels:      []openai.CodexReasoningLevel{{Effort: "medium", Description: "medium"}},
					ShellType:                     "shell_command",
					Visibility:                    "list",
					SupportedInAPI:                true,
					Priority:                      1,
					BaseInstructions:              "You are Codex, a coding assistant.",
					SupportsReasoningSummaries:    false,
					DefaultReasoningSummary:       "auto",
					SupportVerbosity:              false,
					WebSearchToolType:             "text",
					TruncationPolicy:              openai.CodexTruncationPolicy{Mode: "bytes", Limit: 10000},
					SupportsParallelToolCalls:     false,
					EffectiveContextWindowPercent: 95,
					ExperimentalSupportedTools:    []string{},
					InputModalities:               []string{"text"},
					SupportsSearchTool:            false,
				},
			},
		},
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var body openai.ModelsList
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("expected valid json response, got %v", err)
	}
	if body.Object != "list" {
		t.Fatalf("expected list object, got %q", body.Object)
	}
	if len(body.Data) != 1 {
		t.Fatalf("expected one model, got %d", len(body.Data))
	}
	if body.Data[0].ID != "anthropic.claude-3-7-sonnet-20250219-v1:0" {
		t.Fatalf("expected model id to come from service, got %q", body.Data[0].ID)
	}
	if len(body.Models) != 1 {
		t.Fatalf("expected one codex model, got %#v", body.Models)
	}
	if body.Models[0].Slug != "anthropic.claude-3-7-sonnet-20250219-v1:0" {
		t.Fatalf("expected codex model slug to come from service, got %#v", body.Models[0])
	}
}

func TestResponsesHandlerReturnsUpgradeRequiredForWebsocketGET(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/responses", nil)
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", "websocket")

	NewServer(&fakeService{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusUpgradeRequired {
		t.Fatalf("expected 426, got %d", rec.Code)
	}
}

func TestResponsesHandlerRejectsInvalidJSON(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":`))

	NewServer(&fakeService{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}

	var body openai.ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("expected valid error response, got %v", err)
	}
	if body.Error.Type != "invalid_request_error" {
		t.Fatalf("expected invalid request error type, got %q", body.Error.Type)
	}
}

func TestResponsesHandlerDecodesZstdEncodedJSON(t *testing.T) {
	svc := &fakeService{
		response: openai.Response{ID: "resp_1", Object: "response", Model: "model"},
	}
	body := encodeZstd(t, `{"model":"model","input":"hi"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	req.Header.Set("Content-Encoding", "zstd")
	rec := httptest.NewRecorder()

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if svc.calls != 1 {
		t.Fatalf("expected service to be called once, got %d", svc.calls)
	}
	if svc.lastRequest.Model != "model" {
		t.Fatalf("expected decoded model, got %q", svc.lastRequest.Model)
	}
}

func TestResponsesHandlerRejectsUnsupportedContentEncoding(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi"}`))
	req.Header.Set("Content-Encoding", "br")

	NewServer(&fakeService{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusUnsupportedMediaType {
		t.Fatalf("expected 415, got %d", rec.Code)
	}
	var body openai.ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("expected valid error response, got %v", err)
	}
	if body.Error.Type != "invalid_request_error" {
		t.Fatalf("expected invalid request error type, got %q", body.Error.Type)
	}
}

func TestResponsesHandlerAcceptsResponsesRequestWithTools(t *testing.T) {
	svc := &fakeService{
		response: openai.Response{ID: "resp_1", Object: "response", Model: "model"},
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"model",
		"input":"hi",
		"tools":[
			{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}
		]
	}`))

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if svc.calls != 1 {
		t.Fatalf("expected service to be called once, got %d", svc.calls)
	}
	if len(svc.lastRequest.Tools) != 1 {
		t.Fatalf("expected one decoded tool, got %#v", svc.lastRequest.Tools)
	}
	if svc.lastRequest.Tools[0].Type != "function" || svc.lastRequest.Tools[0].Function == nil || svc.lastRequest.Tools[0].Function.Name != "lookup" {
		t.Fatalf("expected decoded function tool, got %#v", svc.lastRequest.Tools[0])
	}
}

func TestResponsesHandlerAcceptsResponsesAPIToolShape(t *testing.T) {
	svc := &fakeService{
		response: openai.Response{ID: "resp_1", Object: "response", Model: "model"},
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"model",
		"input":"hi",
		"parallel_tool_calls": false,
		"tools":[
			{"type":"function","name":"lookup","description":"A demo tool","strict":false,"parameters":{"type":"object"}}
		]
	}`))

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if len(svc.lastRequest.Tools) != 1 {
		t.Fatalf("expected one decoded tool, got %#v", svc.lastRequest.Tools)
	}
	if svc.lastRequest.Tools[0].Function == nil || svc.lastRequest.Tools[0].Function.Name != "lookup" {
		t.Fatalf("expected Responses API function tool shape to populate function metadata, got %#v", svc.lastRequest.Tools[0])
	}
	if got := svc.lastRequest.Tools[0].Function.Parameters["type"]; got != "object" {
		t.Fatalf("expected top-level parameters to be preserved, got %#v", svc.lastRequest.Tools[0].Function.Parameters)
	}
}

func TestResponsesHandlerRejectsMalformedToolDefinition(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"model",
		"input":"hi",
		"tools":[{"type":"function","function":{"parameters":{"type":"object"}}}]
	}`))

	NewServer(&fakeService{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
	var body openai.ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("expected valid error response, got %v", err)
	}
	if body.Error.Type != "invalid_request_error" {
		t.Fatalf("expected invalid request error type, got %q", body.Error.Type)
	}
}

func TestResponsesHandlerSerializesToolCallOutputs(t *testing.T) {
	svc := &fakeService{
		response: openai.Response{
			ID:     "resp_1",
			Object: "response",
			Model:  "model",
			Output: []openai.OutputItem{
				{
					Type:      "function_call",
					CallID:    "call_1",
					Name:      "lookup",
					Arguments: `{"q":"weather"}`,
				},
				{
					Type:   "web_search_call",
					CallID: "call_web",
					Action: map[string]any{"query": "golang"},
				},
			},
		},
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi"}`))

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var body map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("expected valid json response, got %v", err)
	}
	output, ok := body["output"].([]any)
	if !ok || len(output) != 2 {
		t.Fatalf("expected serialized output items, got %#v", body["output"])
	}
	functionCall, ok := output[0].(map[string]any)
	if !ok {
		t.Fatalf("expected first output item object, got %#v", output[0])
	}
	if functionCall["type"] != "function_call" || functionCall["call_id"] != "call_1" || functionCall["name"] != "lookup" {
		t.Fatalf("expected serialized function call item, got %#v", functionCall)
	}
	if functionCall["arguments"] != `{"q":"weather"}` {
		t.Fatalf("expected serialized function arguments, got %#v", functionCall)
	}
	webSearchCall, ok := output[1].(map[string]any)
	if !ok {
		t.Fatalf("expected second output item object, got %#v", output[1])
	}
	action, ok := webSearchCall["action"].(map[string]any)
	if !ok || action["query"] != "golang" {
		t.Fatalf("expected serialized built-in action payload, got %#v", webSearchCall)
	}
}

func TestChatCompletionsHandlerReturnsJSON(t *testing.T) {
	svc := &fakeService{
		chatResponse: openai.ChatCompletionResponse{
			ID:     "chatcmpl_123",
			Object: "chat.completion",
			Model:  "model",
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatMessage{Role: "assistant", Content: openai.ChatMessageText("hello")},
			}},
		},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"model","messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"object":"chat.completion"`) {
		t.Fatalf("expected chat completion payload, got %s", rec.Body.String())
	}
}

func TestChatCompletionsHandlerStreamsSSEWhenRequested(t *testing.T) {
	svc := &fakeService{chatStreamBody: "data: {\"object\":\"chat.completion.chunk\"}\n\ndata: [DONE]\n\n"}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"model","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`))
	rec := httptest.NewRecorder()

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Fatalf("expected SSE content type, got %q", got)
	}
	if !strings.Contains(rec.Body.String(), "[DONE]") {
		t.Fatalf("expected stream terminator, got %s", rec.Body.String())
	}
}

func TestEmbeddingsHandlerReturnsJSON(t *testing.T) {
	svc := &fakeService{
		embeddingsResponse: openai.EmbeddingsResponse{
			Object: "list",
			Model:  "model",
			Data: []openai.Embedding{
				{Object: "embedding", Index: 0, Embedding: []float64{0.1, 0.2}},
			},
			Usage: openai.EmbeddingsUsage{PromptTokens: 1, TotalTokens: 1},
		},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"model","input":"hi"}`))
	rec := httptest.NewRecorder()

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"object":"list"`) {
		t.Fatalf("expected embeddings payload, got %s", rec.Body.String())
	}
}

func TestChatCompletionsHandlerPropagatesSessionHeader(t *testing.T) {
	svc := &fakeService{
		chatResponse: openai.ChatCompletionResponse{
			ID:     "chatcmpl_123",
			Object: "chat.completion",
			Model:  "model",
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatMessage{Role: "assistant", Content: openai.ChatMessageText("hello")},
			}},
		},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"model","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("X-Session-ID", "sess-123")
	rec := httptest.NewRecorder()

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if got := session.IDFromContext(svc.lastChatContext); got != "sess-123" {
		t.Fatalf("expected session id in context, got %q", got)
	}
}

func TestUsageHandlerReturnsOverallTotals(t *testing.T) {
	svc := &fakeService{
		usageSummary: openai.UsageSummary{
			Object: "usage.summary",
			Totals: openai.UsageTotals{
				Requests:         3,
				PromptTokens:     12,
				CompletionTokens: 5,
				TotalTokens:      17,
			},
		},
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/usage", nil)
	rec := httptest.NewRecorder()

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var body openai.UsageSummary
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("expected usage json, got %v", err)
	}
	if body.Totals.Requests != 3 || body.Totals.TotalTokens != 17 {
		t.Fatalf("unexpected usage payload: %#v", body)
	}
}

func TestUsageSessionHandlerReturnsPerSessionTotals(t *testing.T) {
	svc := &fakeService{
		sessionUsage: openai.UsageSession{
			Object:    "usage.session",
			SessionID: "sess-123",
			Totals: openai.UsageTotals{
				Requests:         2,
				PromptTokens:     9,
				CompletionTokens: 3,
				TotalTokens:      12,
			},
		},
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/usage/sess-123", nil)
	rec := httptest.NewRecorder()

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var body openai.UsageSession
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("expected usage session json, got %v", err)
	}
	if body.SessionID != "sess-123" || body.Totals.TotalTokens != 12 {
		t.Fatalf("unexpected session usage payload: %#v", body)
	}
}

func TestModelByIDHandlerReturns404ForUnknownModel(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/v1/models/missing", nil)
	rec := httptest.NewRecorder()

	NewServer(&fakeService{lookupModelErr: openai.NewNotFoundError("model not found")}).ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", rec.Code)
	}
}

func TestModelByIDHandlerReturnsModelPayloadOnSuccess(t *testing.T) {
	svc := &fakeService{
		modelLookup: openai.Model{
			ID:      "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
			Object:  "model",
			OwnedBy: "Anthropic",
			Name:    "Claude 3.7 Sonnet",
		},
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/models/us.anthropic.claude-3-7-sonnet-20250219-v1:0", nil)
	rec := httptest.NewRecorder()

	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"object":"model"`) || !strings.Contains(rec.Body.String(), `"id":"us.anthropic.claude-3-7-sonnet-20250219-v1:0"`) {
		t.Fatalf("unexpected model payload: %s", rec.Body.String())
	}
}

func TestHealthHandlerReturnsOK(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	NewServer(&fakeService{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "\"status\":\"ok\"") {
		t.Fatalf("unexpected health response: %d %s", rec.Code, rec.Body.String())
	}
}

type fakeService struct {
	response           openai.Response
	models             openai.ModelsList
	modelLookup        openai.Model
	chatResponse       openai.ChatCompletionResponse
	embeddingsResponse openai.EmbeddingsResponse
	usageSummary       openai.UsageSummary
	sessionUsage       openai.UsageSession
	err                error
	modelsErr          error
	lookupModelErr     error
	chatErr            error
	embeddingsErr      error
	usageErr           error
	sessionUsageErr    error
	calls              int
	chatCalls          int
	embedCalls         int
	lastRequest        openai.ResponsesRequest
	lastChatRequest    openai.ChatCompletionRequest
	lastEmbedRequest   openai.EmbeddingsRequest
	lastChatContext    context.Context
	streamErr          error
	chatStreamErr      error
	streamBody         string
	chatStreamBody     string
}

func (s *fakeService) Respond(_ context.Context, req openai.ResponsesRequest) (openai.Response, error) {
	s.calls++
	s.lastRequest = req
	return s.response, s.err
}

func (s *fakeService) ListModels(_ context.Context) (openai.ModelsList, error) {
	return s.models, s.modelsErr
}

func (s *fakeService) GetModel(_ context.Context, _ string) (openai.Model, error) {
	return s.modelLookup, s.lookupModelErr
}

func (s *fakeService) CompleteChat(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	s.chatCalls++
	s.lastChatRequest = req
	s.lastChatContext = ctx
	return s.chatResponse, s.chatErr
}

func (s *fakeService) Embed(_ context.Context, req openai.EmbeddingsRequest) (openai.EmbeddingsResponse, error) {
	s.embedCalls++
	s.lastEmbedRequest = req
	return s.embeddingsResponse, s.embeddingsErr
}

func (s *fakeService) Stream(_ context.Context, req openai.ResponsesRequest, w http.ResponseWriter) error {
	s.calls++
	s.lastRequest = req
	if s.streamBody != "" {
		if _, err := w.Write([]byte(s.streamBody)); err != nil {
			return err
		}
	}
	return s.streamErr
}

func (s *fakeService) StreamChat(ctx context.Context, req openai.ChatCompletionRequest, w http.ResponseWriter) error {
	s.chatCalls++
	s.lastChatRequest = req
	s.lastChatContext = ctx
	if s.chatStreamBody != "" {
		if _, err := w.Write([]byte(s.chatStreamBody)); err != nil {
			return err
		}
	}
	return s.chatStreamErr
}

func (s *fakeService) GetUsage(context.Context) (openai.UsageSummary, error) {
	return s.usageSummary, s.usageErr
}

func (s *fakeService) GetSessionUsage(_ context.Context, sessionID string) (openai.UsageSession, error) {
	s.sessionUsage.SessionID = sessionID
	return s.sessionUsage, s.sessionUsageErr
}

type fakeAPIError struct {
	message string
}

func (e fakeAPIError) Error() string {
	return e.message
}

func (e fakeAPIError) ErrorCode() string {
	return "UpstreamError"
}

func (e fakeAPIError) ErrorFault() smithy.ErrorFault {
	return smithy.FaultServer
}

func (e fakeAPIError) ErrorMessage() string {
	return e.message
}

func TestResponsesHandlerReturnsNotImplementedForStreamingWhenServiceDoesNotSupportIt(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi","stream":true}`))

	NewServer(struct{ Service }{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d", rec.Code)
	}
}

func TestResponsesHandlerReturnsJSONErrorForEarlyStreamingFailure(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi","stream":true}`))

	NewServer(&fakeService{streamErr: openai.NewInvalidRequestError("stream failed")}).ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
	var body openai.ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("expected valid error response, got %v", err)
	}
	if body.Error.Type != "invalid_request_error" {
		t.Fatalf("expected invalid request error type, got %q", body.Error.Type)
	}
}

func TestResponsesHandlerStreamsWhenServiceSupportsIt(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi","stream":true}`))

	NewServer(&fakeService{streamBody: "event: response.completed\ndata: {\"status\":\"ok\"}\n\n"}).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("expected text/event-stream content type, got %q", got)
	}
	if !strings.Contains(rec.Body.String(), "event: response.completed") {
		t.Fatalf("expected streamed body, got %q", rec.Body.String())
	}
}

func TestResponsesHandlerReturnsBadGatewayForUpstreamErrors(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi"}`))

	NewServer(&fakeService{err: fakeAPIError{message: "bedrock failed"}}).ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", rec.Code)
	}
}

func TestResponsesHandlerReturnsBadRequestForUnknownPreviousResponseID(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"model","input":"hi","previous_response_id":"resp_missing"}`))

	store := conversation.NewInMemoryStore(10)
	bedrockProxy := &fakeBedrockProxy{}
	svc := proxy.NewService(bedrockProxy, store)
	NewServer(svc).ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
	var body openai.ErrorResponse
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("expected valid error response, got %v", err)
	}
	if body.Error.Message != "unknown previous_response_id" {
		t.Fatalf("expected canonical message, got %q", body.Error.Message)
	}
	if bedrockProxy.respondCalls != 0 {
		t.Fatalf("expected bedrock not to be called, got %d", bedrockProxy.respondCalls)
	}
}

type fakeBedrockProxy struct {
	respondCalls int
}

func (f *fakeBedrockProxy) RespondConversation(context.Context, string, conversation.Request, *int, *float64) (bedrock.ConverseResponse, error) {
	f.respondCalls++
	return bedrock.ConverseResponse{}, nil
}

func (f *fakeBedrockProxy) StreamConversation(context.Context, string, conversation.Request, *int, *float64, http.ResponseWriter) (bedrock.ConverseResponse, error) {
	return bedrock.ConverseResponse{}, nil
}

func (f *fakeBedrockProxy) ListModels(context.Context) ([]bedrock.ModelSummary, error) {
	return nil, nil
}

func (f *fakeBedrockProxy) LookupModel(context.Context, string) (bedrock.ModelRecord, error) {
	return bedrock.ModelRecord{}, nil
}

func (f *fakeBedrockProxy) Chat(context.Context, bedrock.ConverseRequest) (bedrock.ConverseResponse, error) {
	return bedrock.ConverseResponse{}, nil
}

func (f *fakeBedrockProxy) ChatStream(context.Context, bedrock.ConverseRequest) (bedrock.ChatStreamResponse, error) {
	return bedrock.ChatStreamResponse{}, nil
}

func (f *fakeBedrockProxy) Embed(context.Context, openai.EmbeddingsRequest, bedrock.ModelRecord) (openai.EmbeddingsResponse, error) {
	return openai.EmbeddingsResponse{}, nil
}

func encodeZstd(t *testing.T, payload string) []byte {
	t.Helper()

	var buf bytes.Buffer
	encoder, err := zstd.NewWriter(&buf)
	if err != nil {
		t.Fatalf("expected zstd encoder, got %v", err)
	}
	if _, err := encoder.Write([]byte(payload)); err != nil {
		t.Fatalf("expected zstd write to succeed, got %v", err)
	}
	if err := encoder.Close(); err != nil {
		t.Fatalf("expected zstd close to succeed, got %v", err)
	}
	return buf.Bytes()
}
