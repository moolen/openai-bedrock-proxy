package main

import (
	"context"
	"flag"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	appconfig "github.com/moolen/openai-bedrock-proxy/internal/config"
	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	"github.com/moolen/openai-bedrock-proxy/internal/httpserver"
	applog "github.com/moolen/openai-bedrock-proxy/internal/logging"
	"github.com/moolen/openai-bedrock-proxy/internal/proxy"
)

const defaultResponseStoreCapacity = 4096

func resolveLogLevel(flagValue string, cfg appconfig.Config) (slog.Level, error) {
	value := cfg.LogLevel
	if flagValue != "" {
		value = flagValue
	}
	return applog.ParseLevel(value)
}

func main() {
	var logLevelFlag string
	flag.StringVar(&logLevelFlag, "log-level", "", "log level: debug|info|warn|error")
	flag.Parse()

	cfg := appconfig.LoadFromEnv(os.Getenv)
	level, err := resolveLogLevel(logLevelFlag, cfg)
	if err != nil {
		log.Fatal(err)
	}

	logger := applog.NewLogger(level)
	slog.SetDefault(logger)
	addr := net.JoinHostPort(cfg.ListenAddr, cfg.ListenPort)
	bedrock.SetPromptCachingEnabledByDefault(cfg.EnablePromptCaching)

	logger.Info("starting proxy", "listen_addr", addr, "aws_region", cfg.AWSRegion, "log_level", level.String())
	logger.Debug("initializing bedrock client")

	ctx := applog.WithLogger(context.Background(), logger.With("component", "bedrock"))

	client, err := bedrock.NewClient(ctx, cfg.AWSRegion, awsconfig.LoadDefaultConfig)
	if err != nil {
		logger.Error("failed to initialize bedrock client", "error", err)
		os.Exit(1)
	}
	logger.Debug("bedrock client initialized")

	store := conversation.NewInMemoryStore(defaultResponseStoreCapacity)
	logger.Debug("constructing proxy service", "store_capacity", defaultResponseStoreCapacity)
	svc := proxy.NewService(client, store)

	logger.Debug("constructing http server")
	srv := &http.Server{
		Addr:    addr,
		Handler: httpserver.NewServer(svc),
	}
	logger.Info("listening", "listen_addr", addr)

	if err := srv.ListenAndServe(); err != nil {
		logger.Error("server stopped", "error", err)
		os.Exit(1)
	}
}
