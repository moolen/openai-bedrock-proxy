package httpserver

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/aws/smithy-go"
	applog "github.com/moolen/openai-bedrock-proxy/internal/logging"
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
		startedAt := time.Now()
		requestID := newRequestID()
		logger := slog.Default().With(
			"component", "httpserver",
			"request_id", requestID,
			"method", r.Method,
			"path", r.URL.Path,
		)
		tw := &trackingResponseWriter{
			ResponseWriter: w,
			statusCode:     http.StatusOK,
		}
		r = r.WithContext(applog.WithLogger(r.Context(), logger))

		logger.Info("received responses request")
		defer func() {
			logger.Info("completed responses request",
				"status_code", tw.statusCode,
				"bytes_written", tw.bytesWritten,
				"duration", time.Since(startedAt),
			)
		}()

		var req openai.ResponsesRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			logger.Error("failed to decode responses request", "error", err)
			writeError(tw, http.StatusBadRequest, openai.NewInvalidRequestError(err.Error()))
			return
		}

		logger.Debug("decoded responses request",
			"model", req.Model,
			"stream", req.Stream,
			"previous_response_id", req.PreviousResponseID,
			"input", req.Input,
			"instructions", req.Instructions,
			"max_output_tokens", req.MaxOutputTokens,
			"temperature", req.Temperature,
		)

		if err := openai.ValidateResponsesRequest(req); err != nil {
			logger.Warn("responses request validation failed", "error", err)
			writeError(tw, http.StatusBadRequest, err)
			return
		}

		logger.Debug("validated responses request",
			"model", req.Model,
			"stream", req.Stream,
			"previous_response_id", req.PreviousResponseID,
		)

		if req.Stream {
			logger.Info("handling streaming responses request",
				"model", req.Model,
				"previous_response_id", req.PreviousResponseID,
			)
			streamSvc, ok := svc.(StreamingService)
			if !ok {
				err := openai.NewInvalidRequestError("streaming is not supported")
				logger.Error("streaming not supported by service", "error", err)
				writeError(tw, http.StatusNotImplemented, err)
				return
			}

			tw.Header().Set("Content-Type", "text/event-stream")
			tw.Header().Set("Cache-Control", "no-cache")
			tw.Header().Set("Connection", "keep-alive")

			logger.Debug("routing responses request to streaming service")
			if err := streamSvc.Stream(r.Context(), req, tw); err != nil {
				if !tw.started {
					logger.Error("streaming request failed before response started", "error", err)
					writeError(tw, statusCodeFor(err), err)
					return
				}
				logger.Error("streaming request failed after response started", "error", err)
				_ = openai.WriteEvent(tw, "error", openai.ErrorResponseFrom(err))
				return
			}

			logger.Info("streaming responses request completed",
				"model", req.Model,
				"previous_response_id", req.PreviousResponseID,
			)
			return
		}

		resp, err := svc.Respond(r.Context(), req)
		if err != nil {
			logger.Error("responses request failed", "error", err, "status_code", statusCodeFor(err))
			writeError(tw, statusCodeFor(err), err)
			return
		}

		logger.Debug("responses response payload", "response", resp)
		writeJSON(tw, http.StatusOK, resp)
	}
}

func handleModels() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		startedAt := time.Now()
		requestID := newRequestID()
		logger := slog.Default().With(
			"component", "httpserver",
			"request_id", requestID,
			"method", r.Method,
			"path", r.URL.Path,
		)
		tw := &trackingResponseWriter{
			ResponseWriter: w,
			statusCode:     http.StatusOK,
		}

		logger.Info("received models request")
		defer func() {
			logger.Info("completed models request",
				"status_code", tw.statusCode,
				"bytes_written", tw.bytesWritten,
				"duration", time.Since(startedAt),
			)
		}()

		payload := map[string]any{
			"object": "list",
			"data":   []any{},
		}
		logger.Debug("returning models payload", "payload", payload)
		writeJSON(tw, http.StatusOK, payload)
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
	var apiErr smithy.APIError
	if errors.As(err, &apiErr) {
		return http.StatusBadGateway
	}
	return http.StatusInternalServerError
}

type trackingResponseWriter struct {
	http.ResponseWriter
	started      bool
	statusCode   int
	bytesWritten int
}

func (w *trackingResponseWriter) WriteHeader(statusCode int) {
	w.started = true
	w.statusCode = statusCode
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *trackingResponseWriter) Write(p []byte) (int, error) {
	w.started = true
	if w.statusCode == 0 {
		w.statusCode = http.StatusOK
	}
	n, err := w.ResponseWriter.Write(p)
	w.bytesWritten += n
	return n, err
}

func (w *trackingResponseWriter) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func newRequestID() string {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "local"
	}
	return fmt.Sprintf("%x", buf[:])
}
