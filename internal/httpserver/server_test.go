package httpserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/smithy-go"
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
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)

	NewServer(&fakeService{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var body struct {
		Object string `json:"object"`
		Data   []any  `json:"data"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("expected valid json response, got %v", err)
	}
	if body.Object != "list" {
		t.Fatalf("expected list object, got %q", body.Object)
	}
	if body.Data == nil {
		t.Fatal("expected data field to be present")
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

type fakeService struct {
	response    openai.Response
	err         error
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
	svc := proxy.NewService(&fakeBedrockProxy{}, store)
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
