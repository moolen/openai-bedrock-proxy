package config

import "testing"

func TestLoadConfigDefaults(t *testing.T) {
	cfg := LoadFromEnv(func(string) string { return "" })
	if cfg.ListenAddr != "0.0.0.0" {
		t.Fatalf("expected default listen addr, got %q", cfg.ListenAddr)
	}
	if cfg.ListenPort != "8080" {
		t.Fatalf("expected default listen port, got %q", cfg.ListenPort)
	}
}

func TestLoadConfigOverrides(t *testing.T) {
	env := map[string]string{
		"LISTEN_ADDR": "127.0.0.1",
		"LISTEN_PORT": "9000",
		"AWS_REGION":  "us-east-1",
		"LOG_LEVEL":   "debug",
	}
	cfg := LoadFromEnv(func(key string) string { return env[key] })
	if cfg.ListenPort != "9000" {
		t.Fatalf("expected override port, got %q", cfg.ListenPort)
	}
}
