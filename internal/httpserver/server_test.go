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

type fakeService struct {
	response    openai.Response
	models      openai.ModelsList
	err         error
	modelsErr   error
	calls       int
	lastRequest openai.ResponsesRequest
	streamErr   error
	streamBody  string
}

func (s *fakeService) Respond(_ context.Context, req openai.ResponsesRequest) (openai.Response, error) {
	s.calls++
	s.lastRequest = req
	return s.response, s.err
}

func (s *fakeService) ListModels(_ context.Context) (openai.ModelsList, error) {
	return s.models, s.modelsErr
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
