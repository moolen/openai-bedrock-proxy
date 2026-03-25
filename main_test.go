package main

import (
	"log/slog"
	"os/exec"
	"testing"

	"github.com/moolen/openai-bedrock-proxy/internal/config"
	applog "github.com/moolen/openai-bedrock-proxy/internal/logging"
)

func TestResolveLogLevelUsesCLIValueFirst(t *testing.T) {
	cfg := config.Config{LogLevel: "error"}
	level, err := resolveLogLevel("debug", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if level != slog.LevelDebug {
		t.Fatalf("expected debug level, got %v", level)
	}
}

func TestResolveLogLevelFallsBackToConfigValue(t *testing.T) {
	cfg := config.Config{LogLevel: "warn"}
	level, err := resolveLogLevel("", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if level != slog.LevelWarn {
		t.Fatalf("expected warn level, got %v", level)
	}
}

func TestResolveLogLevelRejectsInvalidValue(t *testing.T) {
	cfg := config.Config{LogLevel: "debug"}
	if _, err := resolveLogLevel("loud", cfg); err == nil {
		t.Fatal("expected invalid level error")
	}
}

func TestMainServerBuilds(t *testing.T) {
	cmd := exec.Command("go", "build", "./cmd/openai-bedrock-proxy")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}
}

func resolveLogLevel(flagValue string, cfg config.Config) (slog.Level, error) {
	value := cfg.LogLevel
	if flagValue != "" {
		value = flagValue
	}
	return applog.ParseLevel(value)
}
