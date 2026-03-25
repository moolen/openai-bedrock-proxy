package httpserver

import (
	"compress/gzip"
	"compress/zlib"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/aws/smithy-go"
	"github.com/klauspost/compress/zstd"
	applog "github.com/moolen/openai-bedrock-proxy/internal/logging"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type Service interface {
	Respond(context.Context, openai.ResponsesRequest) (openai.Response, error)
	ListModels(context.Context) (openai.ModelsList, error)
}

type StreamingService interface {
	Stream(context.Context, openai.ResponsesRequest, http.ResponseWriter) error
}

func NewServer(svc Service) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/responses", handleResponses(svc))
	mux.HandleFunc("GET /v1/models", handleModels(svc))
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

		body, err := decodedRequestBody(r)
		if err != nil {
			status := http.StatusBadRequest
			var unsupported unsupportedContentEncodingError
			if errors.As(err, &unsupported) {
				status = http.StatusUnsupportedMediaType
			}
			logger.Error("failed to decode responses request body", "error", err, "status_code", status)
			writeError(tw, status, openai.NewInvalidRequestError(err.Error()))
			return
		}
		defer body.Close()

		var req openai.ResponsesRequest
		if err := json.NewDecoder(body).Decode(&req); err != nil {
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

func handleModels(svc Service) http.HandlerFunc {
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

		models, err := svc.ListModels(r.Context())
		if err != nil {
			logger.Error("models request failed", "error", err, "status_code", statusCodeFor(err))
			writeError(tw, statusCodeFor(err), err)
			return
		}
		logger.Debug("returning models payload", "payload", models)
		writeJSON(tw, http.StatusOK, models)
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

type unsupportedContentEncodingError struct {
	encoding string
}

func (e unsupportedContentEncodingError) Error() string {
	return fmt.Sprintf("unsupported content-encoding: %s", e.encoding)
}

type multiCloserReader struct {
	io.Reader
	closers []io.Closer
}

type noErrorCloser struct {
	close func()
}

func (c noErrorCloser) Close() error {
	c.close()
	return nil
}

func (r *multiCloserReader) Close() error {
	var closeErr error
	for i := len(r.closers) - 1; i >= 0; i-- {
		if err := r.closers[i].Close(); err != nil && closeErr == nil {
			closeErr = err
		}
	}
	return closeErr
}

func decodedRequestBody(r *http.Request) (io.ReadCloser, error) {
	reader := io.Reader(r.Body)
	closers := []io.Closer{r.Body}
	encodings := contentEncodings(r.Header.Values("Content-Encoding"))

	for i := len(encodings) - 1; i >= 0; i-- {
		encoding := encodings[i]
		switch encoding {
		case "", "identity":
			continue
		case "gzip":
			gzipReader, err := gzip.NewReader(reader)
			if err != nil {
				return nil, err
			}
			reader = gzipReader
			closers = append(closers, gzipReader)
		case "deflate":
			zlibReader, err := zlib.NewReader(reader)
			if err != nil {
				return nil, err
			}
			reader = zlibReader
			closers = append(closers, zlibReader)
		case "zstd":
			zstdReader, err := zstd.NewReader(reader)
			if err != nil {
				return nil, err
			}
			reader = zstdReader
			closers = append(closers, noErrorCloser{close: zstdReader.Close})
		default:
			return nil, unsupportedContentEncodingError{encoding: encoding}
		}
	}

	return &multiCloserReader{Reader: reader, closers: closers}, nil
}

func contentEncodings(values []string) []string {
	var encodings []string
	for _, value := range values {
		for _, part := range strings.Split(value, ",") {
			encoding := strings.TrimSpace(strings.ToLower(part))
			if encoding != "" {
				encodings = append(encodings, encoding)
			}
		}
	}
	return encodings
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
