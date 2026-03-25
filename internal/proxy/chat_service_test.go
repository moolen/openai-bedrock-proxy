package proxy

import (
	"context"
	"errors"
	"testing"

	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type fakeBedrockChatAPI struct {
	record    bedrock.ModelRecord
	lookupErr error
	chatResp  bedrock.ConverseResponse
	chatErr   error

	lookupCalls int
	chatCalls   int
	lastModel   string
	lastChatReq bedrock.ConverseRequest
}

func (f *fakeBedrockChatAPI) LookupModel(_ context.Context, id string) (bedrock.ModelRecord, error) {
	f.lookupCalls++
	f.lastModel = id
	return f.record, f.lookupErr
}

func (f *fakeBedrockChatAPI) Chat(_ context.Context, req bedrock.ConverseRequest) (bedrock.ConverseResponse, error) {
	f.chatCalls++
	f.lastChatReq = req
	return f.chatResp, f.chatErr
}

func TestChatServiceCompleteTranslatesRequestAndResponse(t *testing.T) {
	client := &fakeBedrockChatAPI{
		record: bedrock.ModelRecord{ID: "resolved-model"},
		chatResp: bedrock.ConverseResponse{
			ResponseID: "resp_1",
			Output: []bedrock.OutputBlock{
				{Type: bedrock.OutputBlockTypeText, Text: "hello"},
			},
			StopReason: "end_turn",
		},
	}
	service := NewChatService(client)

	got, err := service.Complete(context.Background(), openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: openai.ChatMessageText("hi")},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client.lookupCalls != 1 || client.lastModel != "model" {
		t.Fatalf("expected model lookup call, got calls=%d model=%q", client.lookupCalls, client.lastModel)
	}
	if client.chatCalls != 1 {
		t.Fatalf("expected chat call, got %d", client.chatCalls)
	}
	if client.lastChatReq.ModelID != "resolved-model" {
		t.Fatalf("expected resolved model id in translated request, got %#v", client.lastChatReq)
	}
	if len(got.Choices) != 1 || got.Choices[0].Message.Content.Text != "hello" {
		t.Fatalf("expected translated chat response, got %#v", got)
	}
}

func TestChatServiceCompleteReturnsLookupErrors(t *testing.T) {
	lookupErr := errors.New("catalog down")
	client := &fakeBedrockChatAPI{lookupErr: lookupErr}
	service := NewChatService(client)

	_, err := service.Complete(context.Background(), openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: openai.ChatMessageText("hi")},
		},
	})
	if !errors.Is(err, lookupErr) {
		t.Fatalf("expected lookup error, got %v", err)
	}
	if client.chatCalls != 0 {
		t.Fatalf("expected no chat call after lookup failure, got %d", client.chatCalls)
	}
}

func TestChatServiceCompleteReturnsTranslationErrors(t *testing.T) {
	client := &fakeBedrockChatAPI{record: bedrock.ModelRecord{ID: "resolved-model"}}
	service := NewChatService(client)

	_, err := service.Complete(context.Background(), openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{
			{
				Role: "user",
				Content: openai.ChatMessageContent{
					Kind: openai.ChatMessageContentKindParts,
					Parts: []openai.ChatMessageContentPart{
						{Type: "image_url", ImageURL: map[string]any{"url": "https://example.com/cat.png"}},
					},
				},
			},
		},
	})
	if err == nil {
		t.Fatal("expected translation error")
	}
	var invalidRequest openai.InvalidRequestError
	if !errors.As(err, &invalidRequest) {
		t.Fatalf("expected invalid request error, got %T", err)
	}
	if client.chatCalls != 0 {
		t.Fatalf("expected no chat call after translation failure, got %d", client.chatCalls)
	}
}

func TestChatServiceCompleteReturnsChatErrors(t *testing.T) {
	chatErr := errors.New("converse failed")
	client := &fakeBedrockChatAPI{
		record:  bedrock.ModelRecord{ID: "resolved-model"},
		chatErr: chatErr,
	}
	service := NewChatService(client)

	_, err := service.Complete(context.Background(), openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: openai.ChatMessageText("hi")},
		},
	})
	if !errors.Is(err, chatErr) {
		t.Fatalf("expected chat error, got %v", err)
	}
	if client.chatCalls != 1 {
		t.Fatalf("expected one chat call, got %d", client.chatCalls)
	}
}
