package logging

import (
	"context"
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

func TestFromContextFallsBackToDefaultLogger(t *testing.T) {
	logger := FromContext(context.Background())
	if logger == nil {
		t.Fatal("expected fallback logger")
	}
}
