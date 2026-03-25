package proxy

import (
	"context"
	"errors"
	"testing"

	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type fakeBedrockEmbeddingsAPI struct {
	record    bedrock.ModelRecord
	lookupErr error
	embedResp openai.EmbeddingsResponse
	embedErr  error

	lookupCalls int
	embedCalls  int
	lastModel   string
	lastReq     openai.EmbeddingsRequest
	lastRecord  bedrock.ModelRecord
}

func (f *fakeBedrockEmbeddingsAPI) LookupModel(_ context.Context, id string) (bedrock.ModelRecord, error) {
	f.lookupCalls++
	f.lastModel = id
	return f.record, f.lookupErr
}

func (f *fakeBedrockEmbeddingsAPI) Embed(_ context.Context, req openai.EmbeddingsRequest, record bedrock.ModelRecord) (openai.EmbeddingsResponse, error) {
	f.embedCalls++
	f.lastReq = req
	f.lastRecord = record
	return f.embedResp, f.embedErr
}

func TestEmbeddingsServiceReturnsValidationErrors(t *testing.T) {
	service := NewEmbeddingsService(&fakeBedrockEmbeddingsAPI{})

	_, err := service.Create(context.Background(), openai.EmbeddingsRequest{
		Model:      "cohere.embed-english-v3",
		Input:      "hello",
		Dimensions: intPtr(256),
	})
	if err == nil {
		t.Fatal("expected validation error")
	}
	var invalid openai.InvalidRequestError
	if !errors.As(err, &invalid) {
		t.Fatalf("expected invalid request error, got %T", err)
	}
}

func TestEmbeddingsServiceResolvesModelAndReturnsResponse(t *testing.T) {
	client := &fakeBedrockEmbeddingsAPI{
		record: bedrock.ModelRecord{
			ID:                        "us.amazon.nova-2-multimodal-embeddings-v1:0",
			ResolvedFoundationModelID: "amazon.nova-2-multimodal-embeddings-v1:0",
		},
		embedResp: openai.EmbeddingsResponse{
			Object: "list",
			Model:  "amazon.nova-2-multimodal-embeddings-v1:0",
			Data: []openai.Embedding{
				{Object: "embedding", Index: 0, Embedding: []float64{0.1, 0.2}},
			},
			Usage: openai.EmbeddingsUsage{PromptTokens: 1, TotalTokens: 1},
		},
	}
	service := NewEmbeddingsService(client)

	got, err := service.Create(context.Background(), openai.EmbeddingsRequest{
		Model: "us.amazon.nova-2-multimodal-embeddings-v1:0",
		Input: "hello",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client.lookupCalls != 1 || client.lastModel != "us.amazon.nova-2-multimodal-embeddings-v1:0" {
		t.Fatalf("expected model lookup, got calls=%d model=%q", client.lookupCalls, client.lastModel)
	}
	if client.embedCalls != 1 {
		t.Fatalf("expected one embed call, got %d", client.embedCalls)
	}
	if client.lastRecord.ResolvedFoundationModelID != "amazon.nova-2-multimodal-embeddings-v1:0" {
		t.Fatalf("expected resolved model record to reach embed call, got %#v", client.lastRecord)
	}
	if got.Model != "amazon.nova-2-multimodal-embeddings-v1:0" || len(got.Data) != 1 {
		t.Fatalf("expected embeddings response, got %#v", got)
	}
}

func intPtr(value int) *int {
	return &value
}
