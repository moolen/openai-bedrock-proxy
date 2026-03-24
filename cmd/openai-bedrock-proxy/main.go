package main

import (
	"context"
	"log"
	"net"
	"net/http"
	"os"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	appconfig "github.com/moolen/openai-bedrock-proxy/internal/config"
	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
	"github.com/moolen/openai-bedrock-proxy/internal/httpserver"
	"github.com/moolen/openai-bedrock-proxy/internal/proxy"
)

const defaultResponseStoreCapacity = 4096

func main() {
	ctx := context.Background()
	cfg := appconfig.LoadFromEnv(os.Getenv)

	client, err := bedrock.NewClient(ctx, cfg.AWSRegion, awsconfig.LoadDefaultConfig)
	if err != nil {
		log.Fatal(err)
	}

	store := conversation.NewInMemoryStore(defaultResponseStoreCapacity)
	svc := proxy.NewService(client, store)

	srv := &http.Server{
		Addr:    net.JoinHostPort(cfg.ListenAddr, cfg.ListenPort),
		Handler: httpserver.NewServer(svc),
	}

	log.Fatal(srv.ListenAndServe())
}
