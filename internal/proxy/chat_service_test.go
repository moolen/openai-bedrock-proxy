package proxy

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/moolen/openai-bedrock-proxy/internal/bedrock"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type fakeBedrockChatAPI struct {
	record     bedrock.ModelRecord
	lookupErr  error
	chatResp   bedrock.ConverseResponse
	chatErr    error
	streamResp bedrock.ChatStreamResponse
	streamErr  error

	lookupCalls   int
	chatCalls     int
	streamCalls   int
	lastModel     string
	lastChatReq   bedrock.ConverseRequest
	lastStreamReq bedrock.ConverseRequest
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

func (f *fakeBedrockChatAPI) ChatStream(_ context.Context, req bedrock.ConverseRequest) (bedrock.ChatStreamResponse, error) {
	f.streamCalls++
	f.lastStreamReq = req
	return f.streamResp, f.streamErr
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

func TestChatServiceStreamTranslatesRequestAndWritesChunks(t *testing.T) {
	client := &fakeBedrockChatAPI{
		record: bedrock.ModelRecord{ID: "resolved-model"},
		streamResp: bedrock.ChatStreamResponse{
			ResponseID: "bedrock_1",
			Stream: newProxyFakeChatStream(
				proxyTextDelta("hel"),
				proxyTextDelta("lo"),
				proxyMessageStop("end_turn"),
				proxyMetadataUsage(10, 2),
			),
		},
	}
	service := NewChatService(client)
	var buf bytes.Buffer

	err := service.Stream(context.Background(), openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: openai.ChatMessageText("hi")},
		},
		Stream: true,
		StreamOptions: &openai.StreamOptions{
			IncludeUsage: true,
		},
	}, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client.lookupCalls != 1 || client.lastModel != "model" {
		t.Fatalf("expected model lookup call, got calls=%d model=%q", client.lookupCalls, client.lastModel)
	}
	if client.streamCalls != 1 {
		t.Fatalf("expected one chat stream call, got %d", client.streamCalls)
	}
	if client.lastStreamReq.ModelID != "resolved-model" {
		t.Fatalf("expected resolved model id in stream request, got %#v", client.lastStreamReq)
	}

	got := buf.String()
	if !strings.Contains(got, "\"object\":\"chat.completion.chunk\"") {
		t.Fatalf("expected chunk output, got %s", got)
	}
	if !strings.Contains(got, "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}") {
		t.Fatalf("expected usage chunk, got %s", got)
	}
	if !strings.HasSuffix(got, "data: [DONE]\n\n") {
		t.Fatalf("expected done marker, got %s", got)
	}
}

func TestChatServiceStreamReturnsChatStreamErrors(t *testing.T) {
	chatErr := errors.New("converse stream failed")
	client := &fakeBedrockChatAPI{
		record:    bedrock.ModelRecord{ID: "resolved-model"},
		streamErr: chatErr,
	}
	service := NewChatService(client)

	err := service.Stream(context.Background(), openai.ChatCompletionRequest{
		Model: "model",
		Messages: []openai.ChatMessage{
			{Role: "user", Content: openai.ChatMessageText("hi")},
		},
		Stream: true,
	}, &bytes.Buffer{})
	if !errors.Is(err, chatErr) {
		t.Fatalf("expected chat stream error, got %v", err)
	}
	if client.streamCalls != 1 {
		t.Fatalf("expected one chat stream call, got %d", client.streamCalls)
	}
}

type proxyFakeChatStream struct {
	events chan bedrocktypes.ConverseStreamOutput
}

func newProxyFakeChatStream(events ...bedrocktypes.ConverseStreamOutput) *proxyFakeChatStream {
	ch := make(chan bedrocktypes.ConverseStreamOutput, len(events))
	for _, event := range events {
		ch <- event
	}
	close(ch)
	return &proxyFakeChatStream{events: ch}
}

func (f *proxyFakeChatStream) Events() <-chan bedrocktypes.ConverseStreamOutput {
	return f.events
}

func (f *proxyFakeChatStream) Close() error {
	return nil
}

func (f *proxyFakeChatStream) Err() error {
	return nil
}

func proxyTextDelta(value string) bedrocktypes.ConverseStreamOutput {
	return &bedrocktypes.ConverseStreamOutputMemberContentBlockDelta{
		Value: bedrocktypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(0),
			Delta:             &bedrocktypes.ContentBlockDeltaMemberText{Value: value},
		},
	}
}

func proxyMessageStop(reason string) bedrocktypes.ConverseStreamOutput {
	return &bedrocktypes.ConverseStreamOutputMemberMessageStop{
		Value: bedrocktypes.MessageStopEvent{
			StopReason: bedrocktypes.StopReason(reason),
		},
	}
}

func proxyMetadataUsage(prompt int32, completion int32) bedrocktypes.ConverseStreamOutput {
	total := prompt + completion
	return &bedrocktypes.ConverseStreamOutputMemberMetadata{
		Value: bedrocktypes.ConverseStreamMetadataEvent{
			Usage: &bedrocktypes.TokenUsage{
				InputTokens:  aws.Int32(prompt),
				OutputTokens: aws.Int32(completion),
				TotalTokens:  aws.Int32(total),
			},
		},
	}
}
