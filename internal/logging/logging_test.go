package logging

import (
	"context"
	"log/slog"
	"testing"
)

func TestParseLevelAcceptsKnownLevels(t *testing.T) {
	cases := []string{"debug", "info", "warn", "error"}
	for _, tc := range cases {
		if _, err := ParseLevel(tc); err != nil {
			t.Fatalf("expected %q to parse, got %v", tc, err)
		}
	}
}

func TestParseLevelRejectsUnknownValue(t *testing.T) {
	if _, err := ParseLevel("trace"); err == nil {
		t.Fatal("expected invalid level error")
	}
}

func TestParseLevelTrimsAndLowercases(t *testing.T) {
	level, err := ParseLevel(" DEBUG ")
	if err != nil {
		t.Fatalf("expected normalized level to parse, got %v", err)
	}
	if level != slog.LevelDebug {
		t.Fatalf("expected %v, got %v", slog.LevelDebug, level)
	}
}

func TestWithLoggerRoundTripsViaContext(t *testing.T) {
	logger := NewLogger(slog.LevelInfo)
	ctx := WithLogger(context.Background(), logger)
	fromCtx := FromContext(ctx)
	if fromCtx != logger {
		t.Fatal("expected logger from context to match provided logger")
	}
}

func TestFromContextNilFallsBackToDefaultLogger(t *testing.T) {
	logger := FromContext(nil)
	if logger == nil {
		t.Fatal("expected fallback logger")
	}
}

func TestFromContextFallsBackToDefaultLogger(t *testing.T) {
	logger := FromContext(context.Background())
	if logger == nil {
		t.Fatal("expected fallback logger")
	}
}
