package openai

import (
	"bytes"
	"strings"
	"testing"
)

func TestWriteEventFormatsSSEFrame(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteEvent(&buf, "response.output_text.delta", map[string]any{"delta": "hi"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "event: response.output_text.delta\n") {
		t.Fatalf("unexpected SSE payload: %q", buf.String())
	}
	if !strings.Contains(buf.String(), "data: {\"delta\":\"hi\"}\n\n") {
		t.Fatalf("unexpected SSE payload: %q", buf.String())
	}
}
