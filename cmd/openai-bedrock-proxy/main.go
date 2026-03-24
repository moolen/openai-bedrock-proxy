package main

import (
	"context"
	"log"
	"net"
	"net/http"
	"os"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	appconfig "github.com/moolen/openai-bedrock-proxy/internal/config"
	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	"github.com/moolen/openai-bedrock-proxy/internal/httpserver"
)

func main() {
	ctx := context.Background()
	cfg := appconfig.LoadFromEnv(os.Getenv)

	client, err := bedrock.NewClient(ctx, cfg.AWSRegion, awsconfig.LoadDefaultConfig)
	if err != nil {
		log.Fatal(err)
	}

	srv := &http.Server{
		Addr:    net.JoinHostPort(cfg.ListenAddr, cfg.ListenPort),
		Handler: httpserver.NewServer(client),
	}

	log.Fatal(srv.ListenAndServe())
}
