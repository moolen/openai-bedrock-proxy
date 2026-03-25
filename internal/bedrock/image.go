package bedrock

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"time"
)

const maxRemoteImageBytes = 20 << 20

type imageFetchFunc func(context.Context, string) ([]byte, string, error)

var defaultImageFetcher imageFetchFunc = fetchRemoteImage

func ParseImageURL(ctx context.Context, raw string, fetch imageFetchFunc) ([]byte, string, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil, "", errors.New("image URL is required")
	}

	if hasDataURLScheme(trimmed) {
		return parseImageDataURL(trimmed)
	}

	if fetch == nil {
		fetch = defaultImageFetcher
	}

	data, contentType, err := fetch(ctx, trimmed)
	if err != nil {
		return nil, "", err
	}
	return finalizeFetchedImage(data, contentType)
}

func fetchRemoteImage(ctx context.Context, raw string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, raw, nil)
	if err != nil {
		return nil, "", err
	}

	client := &http.Client{
		Timeout: 30 * time.Second,
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, "", fmt.Errorf("unexpected image response status: %d", resp.StatusCode)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxRemoteImageBytes+1))
	if err != nil {
		return nil, "", err
	}
	if len(data) > maxRemoteImageBytes {
		return nil, "", fmt.Errorf("image response is too large")
	}

	return data, resp.Header.Get("Content-Type"), nil
}

func parseImageDataURL(raw string) ([]byte, string, error) {
	header, payload, ok := strings.Cut(raw, ",")
	if !ok {
		return nil, "", errors.New("image data URL is invalid")
	}
	if !hasDataURLScheme(header) {
		return nil, "", errors.New("image data URL is invalid")
	}

	mediaType := strings.TrimSpace(header[len("data:"):])
	if !strings.HasSuffix(strings.ToLower(mediaType), ";base64") {
		return nil, "", errors.New("image data URL must be base64 encoded")
	}
	mediaType = strings.TrimSpace(mediaType[:len(mediaType)-len(";base64")])
	contentType, err := normalizeImageContentType(mediaType)
	if err != nil {
		return nil, "", err
	}

	decoder := base64.NewDecoder(base64.StdEncoding, strings.NewReader(strings.TrimSpace(payload)))
	data, err := io.ReadAll(io.LimitReader(decoder, maxRemoteImageBytes+1))
	if err != nil {
		return nil, "", err
	}
	if len(data) > maxRemoteImageBytes {
		return nil, "", fmt.Errorf("image data URL is too large")
	}
	if err := validateImagePayloadContentType(contentType, data); err != nil {
		return nil, "", err
	}
	return data, contentType, nil
}

func finalizeFetchedImage(data []byte, rawContentType string) ([]byte, string, error) {
	declaredContentType, declaredErr := normalizeImageContentType(rawContentType)
	detectedContentType, detectedErr := normalizeImageContentType(http.DetectContentType(data))

	switch {
	case declaredErr == nil && detectedErr == nil:
		if declaredContentType != detectedContentType {
			return nil, "", fmt.Errorf("image content type does not match payload")
		}
		return data, declaredContentType, nil
	case declaredErr == nil:
		return nil, "", detectedErr
	case detectedErr == nil:
		return data, detectedContentType, nil
	default:
		return nil, "", detectedErr
	}
}

func validateImagePayloadContentType(contentType string, data []byte) error {
	detectedContentType, err := normalizeImageContentType(http.DetectContentType(data))
	if err != nil {
		return err
	}
	if detectedContentType != contentType {
		return fmt.Errorf("image content type does not match payload")
	}
	return nil
}

func hasDataURLScheme(raw string) bool {
	return len(raw) >= len("data:") && strings.EqualFold(raw[:len("data:")], "data:")
}

func normalizeImageContentType(raw string) (string, error) {
	mediaType, _, err := mime.ParseMediaType(strings.TrimSpace(raw))
	if err != nil {
		return "", err
	}

	switch strings.ToLower(mediaType) {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return strings.ToLower(mediaType), nil
	default:
		return "", fmt.Errorf("unsupported image content type: %s", mediaType)
	}
}

func imageFormatFromContentType(contentType string) (string, error) {
	switch contentType {
	case "image/png":
		return "png", nil
	case "image/jpeg":
		return "jpeg", nil
	case "image/gif":
		return "gif", nil
	case "image/webp":
		return "webp", nil
	default:
		return "", fmt.Errorf("unsupported image content type: %s", contentType)
	}
}
