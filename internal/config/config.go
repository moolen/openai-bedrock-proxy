package config

type Config struct {
	ListenAddr string
	ListenPort string
	AWSRegion  string
	LogLevel   string
}

func LoadFromEnv(getenv func(string) string) Config {
	return Config{
		ListenAddr: firstNonEmpty(getenv("LISTEN_ADDR"), "0.0.0.0"),
		ListenPort: firstNonEmpty(getenv("LISTEN_PORT"), "8080"),
		AWSRegion:  getenv("AWS_REGION"),
		LogLevel:   firstNonEmpty(getenv("LOG_LEVEL"), "debug"),
	}
}

func firstNonEmpty(value string, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
