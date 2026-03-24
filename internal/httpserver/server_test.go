package httpserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
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
}

func (s *fakeService) Respond(_ context.Context, req openai.ResponsesRequest) (openai.Response, error) {
	s.calls++
	s.lastRequest = req
	return s.response, s.err
}
