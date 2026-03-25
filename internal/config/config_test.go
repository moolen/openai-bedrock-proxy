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
	if cfg.AWSRegion != "" {
		t.Fatalf("expected default aws region to be empty, got %q", cfg.AWSRegion)
	}
	if cfg.LogLevel != "debug" {
		t.Fatalf("expected default log level, got %q", cfg.LogLevel)
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
	if cfg.ListenAddr != "127.0.0.1" {
		t.Fatalf("expected override listen addr, got %q", cfg.ListenAddr)
	}
	if cfg.ListenPort != "9000" {
		t.Fatalf("expected override port, got %q", cfg.ListenPort)
	}
	if cfg.AWSRegion != "us-east-1" {
		t.Fatalf("expected override aws region, got %q", cfg.AWSRegion)
	}
	if cfg.LogLevel != "debug" {
		t.Fatalf("expected override log level, got %q", cfg.LogLevel)
	}
}

func TestLoadConfigPromptCaching(t *testing.T) {
	cfg := LoadFromEnv(func(string) string { return "" })
	if cfg.EnablePromptCaching {
		t.Fatal("expected prompt caching to default disabled")
	}

	env := map[string]string{
		"ENABLE_PROMPT_CACHING": "true",
	}
	cfg = LoadFromEnv(func(key string) string { return env[key] })
	if !cfg.EnablePromptCaching {
		t.Fatal("expected prompt caching env override to enable caching")
	}

	env["ENABLE_PROMPT_CACHING"] = "0"
	cfg = LoadFromEnv(func(key string) string { return env[key] })
	if cfg.EnablePromptCaching {
		t.Fatal("expected \"0\" to keep prompt caching disabled")
	}

	env["ENABLE_PROMPT_CACHING"] = "treu"
	cfg = LoadFromEnv(func(key string) string { return env[key] })
	if cfg.EnablePromptCaching {
		t.Fatal("expected invalid prompt caching value to remain disabled")
	}
}
