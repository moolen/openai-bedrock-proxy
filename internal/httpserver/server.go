package httpserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type Service interface {
	Respond(context.Context, openai.ResponsesRequest) (openai.Response, error)
}

type StreamingService interface {
	Stream(context.Context, openai.ResponsesRequest, http.ResponseWriter) error
}

func NewServer(svc Service) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", handleResponses(svc))
	mux.HandleFunc("GET /v1/models", handleModels())
	return mux
}

func handleResponses(svc Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req openai.ResponsesRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, openai.NewInvalidRequestError(err.Error()))
			return
		}
		if err := openai.ValidateResponsesRequest(req); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		if req.Stream {
			streamSvc, ok := svc.(StreamingService)
			if !ok {
				writeError(w, http.StatusNotImplemented, openai.NewInvalidRequestError("streaming is not supported"))
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			if err := streamSvc.Stream(r.Context(), req, w); err != nil {
				_ = openai.WriteEvent(w, "error", openai.ErrorResponseFrom(err))
			}
			return
		}

		resp, err := svc.Respond(r.Context(), req)
		if err != nil {
			writeError(w, statusCodeFor(err), err)
			return
		}
		writeJSON(w, http.StatusOK, resp)
	}
}

func handleModels() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"object": "list",
			"data":   []any{},
		})
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, openai.ErrorResponseFrom(err))
}

func statusCodeFor(err error) int {
	var invalidRequest openai.InvalidRequestError
	if errors.As(err, &invalidRequest) {
		return http.StatusBadRequest
	}
	return http.StatusInternalServerError
}
