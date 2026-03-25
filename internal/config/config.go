package config

import "strings"

type Config struct {
	ListenAddr          string
	ListenPort          string
	AWSRegion           string
	LogLevel            string
	EnablePromptCaching bool
}

func LoadFromEnv(getenv func(string) string) Config {
	return Config{
		ListenAddr:          firstNonEmpty(getenv("LISTEN_ADDR"), "0.0.0.0"),
		ListenPort:          firstNonEmpty(getenv("LISTEN_PORT"), "8080"),
		AWSRegion:           getenv("AWS_REGION"),
		LogLevel:            firstNonEmpty(getenv("LOG_LEVEL"), "debug"),
		EnablePromptCaching: envBoolFalseDefault(getenv("ENABLE_PROMPT_CACHING")),
	}
}

func firstNonEmpty(value string, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func envBoolFalseDefault(value string) bool {
	trimmed := strings.TrimSpace(value)
	return trimmed == "1" ||
		strings.EqualFold(trimmed, "true") ||
		strings.EqualFold(trimmed, "yes") ||
		strings.EqualFold(trimmed, "on")
}
