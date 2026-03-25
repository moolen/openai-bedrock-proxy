package bedrock

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseImageURLDecodesDataURL(t *testing.T) {
	fetchCalled := false
	wantData := testPNGBytes(t)

	data, contentType, err := ParseImageURL(context.Background(), testPNGDataURL(t), func(context.Context, string) ([]byte, string, error) {
		fetchCalled = true
		return nil, "", errors.New("unexpected fetch")
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fetchCalled {
		t.Fatal("expected data URLs to bypass remote fetch")
	}
	if !bytes.Equal(data, wantData) {
		t.Fatalf("expected decoded image bytes, got %#v", data)
	}
	if contentType != "image/png" {
		t.Fatalf("expected png content type, got %q", contentType)
	}
}

func TestParseImageURLAcceptsUppercaseDataScheme(t *testing.T) {
	data, contentType, err := ParseImageURL(context.Background(), strings.ToUpper(testPNGDataURL(t)[:4])+testPNGDataURL(t)[4:], nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !bytes.Equal(data, testPNGBytes(t)) {
		t.Fatalf("expected decoded image bytes, got %#v", data)
	}
	if contentType != "image/png" {
		t.Fatalf("expected png content type, got %q", contentType)
	}
}

func TestParseImageURLFetchesRemoteURL(t *testing.T) {
	fetchCalls := 0
	wantData := testPNGBytes(t)

	data, contentType, err := ParseImageURL(context.Background(), "https://example.com/cat.png", func(_ context.Context, url string) ([]byte, string, error) {
		fetchCalls++
		if url != "https://example.com/cat.png" {
			t.Fatalf("unexpected fetch url: %q", url)
		}
		return wantData, "image/png", nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fetchCalls != 1 {
		t.Fatalf("expected one remote fetch, got %d", fetchCalls)
	}
	if !bytes.Equal(data, wantData) {
		t.Fatalf("expected fetched image bytes, got %#v", data)
	}
	if contentType != "image/png" {
		t.Fatalf("expected fetched content type, got %q", contentType)
	}
}

func TestParseImageURLRejectsRemoteNonImageContentType(t *testing.T) {
	_, _, err := ParseImageURL(context.Background(), "https://example.com/not-image", func(_ context.Context, _ string) ([]byte, string, error) {
		return []byte("not an image"), "text/plain", nil
	})
	if err == nil {
		t.Fatal("expected non-image remote response to be rejected")
	}
}

func TestParseImageURLRejectsFalseRemoteImageContentType(t *testing.T) {
	_, _, err := ParseImageURL(context.Background(), "https://example.com/not-image", func(_ context.Context, _ string) ([]byte, string, error) {
		return []byte("not an image"), "image/png", nil
	})
	if err == nil {
		t.Fatal("expected false image content type to be rejected")
	}
}

func TestParseImageURLHonorsContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, _, err := ParseImageURL(ctx, "https://example.com/cat.png", func(ctx context.Context, _ string) ([]byte, string, error) {
		return nil, "", ctx.Err()
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context cancellation, got %v", err)
	}
}

func TestFetchRemoteImageRejectsOversizedBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write(bytes.Repeat([]byte("a"), maxRemoteImageBytes+1))
	}))
	defer server.Close()

	_, _, err := fetchRemoteImage(context.Background(), server.URL)
	if err == nil {
		t.Fatal("expected oversized image response to be rejected")
	}
	if !strings.Contains(err.Error(), "too large") {
		t.Fatalf("expected size limit error, got %v", err)
	}
}

func TestParseImageURLRejectsOversizedDataURL(t *testing.T) {
	raw := "data:image/png;base64," + base64.StdEncoding.EncodeToString(bytes.Repeat([]byte("a"), maxRemoteImageBytes+1))

	_, _, err := ParseImageURL(context.Background(), raw, nil)
	if err == nil {
		t.Fatal("expected oversized data URL to be rejected")
	}
	if !strings.Contains(err.Error(), "too large") {
		t.Fatalf("expected size limit error, got %v", err)
	}
}

func TestParseImageURLRejectsFalseDataURLContentType(t *testing.T) {
	raw := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("not an image"))

	_, _, err := ParseImageURL(context.Background(), raw, nil)
	if err == nil {
		t.Fatal("expected false data URL content type to be rejected")
	}
}

func testPNGBytes(t *testing.T) []byte {
	t.Helper()

	data, err := base64.StdEncoding.DecodeString("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aF9sAAAAASUVORK5CYII=")
	if err != nil {
		t.Fatalf("expected embedded png fixture to decode, got %v", err)
	}
	return data
}

func testPNGDataURL(t *testing.T) string {
	t.Helper()
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(testPNGBytes(t))
}
