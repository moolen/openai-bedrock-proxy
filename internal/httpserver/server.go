package httpserver

import (
	"compress/gzip"
	"compress/zlib"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/aws/smithy-go"
	"github.com/klauspost/compress/zstd"
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
		body, err := decodedRequestBody(r)
		if err != nil {
			status := http.StatusBadRequest
			var unsupported unsupportedContentEncodingError
			if errors.As(err, &unsupported) {
				status = http.StatusUnsupportedMediaType
			}
			writeError(w, status, openai.NewInvalidRequestError(err.Error()))
			return
		}
		defer body.Close()

		var req openai.ResponsesRequest
		if err := json.NewDecoder(body).Decode(&req); err != nil {
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
			tw := &trackingResponseWriter{ResponseWriter: w}
			tw.Header().Set("Content-Type", "text/event-stream")
			tw.Header().Set("Cache-Control", "no-cache")
			tw.Header().Set("Connection", "keep-alive")
			if err := streamSvc.Stream(r.Context(), req, tw); err != nil {
				if !tw.started {
					writeError(w, statusCodeFor(err), err)
					return
				}
				_ = openai.WriteEvent(tw, "error", openai.ErrorResponseFrom(err))
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

func handleModels(svc Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		models, err := svc.ListModels(r.Context())
		if err != nil {
			writeError(w, statusCodeFor(err), err)
			return
		}
		writeJSON(w, http.StatusOK, models)
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
	started bool
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
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *trackingResponseWriter) Write(p []byte) (int, error) {
	w.started = true
	return w.ResponseWriter.Write(p)
}

func (w *trackingResponseWriter) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}
