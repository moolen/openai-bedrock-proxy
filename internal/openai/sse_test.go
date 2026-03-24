package openai

import (
	"bytes"
	"testing"
)

func TestWriteEventFormatsSSEFrame(t *testing.T) {
	writer := &fakeFlushWriter{}
	if err := WriteEvent(writer, "response.output_text.delta", map[string]any{"delta": "hi"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "event: response.output_text.delta\ndata: {\"delta\":\"hi\"}\n\n"
	if writer.String() != want {
		t.Fatalf("unexpected SSE payload: %q", writer.String())
	}
	if !writer.flushed {
		t.Fatal("expected writer to be flushed")
	}
	if writer.flushCount != 1 {
		t.Fatalf("expected one flush call, got %d", writer.flushCount)
	}
	if writer.flushedBeforeWrite {
		t.Fatal("expected write to happen before flush")
	}
}

type fakeFlushWriter struct {
	bytes.Buffer
	flushed           bool
	flushedBeforeWrite bool
	flushCount        int
}

func (w *fakeFlushWriter) Flush() {
	if w.Len() == 0 {
		w.flushedBeforeWrite = true
	}
	w.flushed = true
	w.flushCount++
}
