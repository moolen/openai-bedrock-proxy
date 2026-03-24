package bedrock

import (
	"context"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

func TestNewClientUsesDefaultAWSConfig(t *testing.T) {
	called := false
	loader := func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error) {
		called = true
		return aws.Config{Region: "us-west-2"}, nil
	}

	client, err := NewClient(context.Background(), "", loader)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatal("expected AWS config loader to be called")
	}
	if client == nil {
		t.Fatal("expected client to be created")
	}
}

func TestNewClientAppliesRegionOverride(t *testing.T) {
	var captured config.LoadOptions
	loader := func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error) {
		for _, optFn := range optFns {
			if err := optFn(&captured); err != nil {
				t.Fatalf("unexpected option error: %v", err)
			}
		}
		return aws.Config{Region: "ignored-by-test"}, nil
	}

	_, err := NewClient(context.Background(), "eu-central-1", loader)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if captured.Region != "eu-central-1" {
		t.Fatalf("expected region override, got %q", captured.Region)
	}
}

func TestNewClientRejectsNilLoader(t *testing.T) {
	client, err := NewClient(context.Background(), "", nil)
	if err == nil {
		t.Fatal("expected nil loader error")
	}
	if client != nil {
		t.Fatal("expected no client when loader is nil")
	}
}

func TestNewClientPropagatesLoaderError(t *testing.T) {
	want := errors.New("load failed")
	loader := func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error) {
		return aws.Config{}, want
	}

	client, err := NewClient(context.Background(), "", loader)
	if !errors.Is(err, want) {
		t.Fatalf("expected loader error, got %v", err)
	}
	if client != nil {
		t.Fatal("expected no client on loader error")
	}
}

func TestClientRespondTranslatesRequestAndParsesText(t *testing.T) {
	maxTokens := 64
	temperature := 0.25
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberText{Value: "hello back"},
					},
				},
			},
		},
	}
	client := &Client{runtime: runtime}

	resp, err := client.Respond(context.Background(), openai.ResponsesRequest{
		Model:           "model-id",
		Input:           "hello",
		Instructions:    "be terse",
		MaxOutputTokens: &maxTokens,
		Temperature:     &temperature,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if runtime.lastConverseInput == nil {
		t.Fatal("expected converse input to be captured")
	}
	if got := aws.ToString(runtime.lastConverseInput.ModelId); got != "model-id" {
		t.Fatalf("expected model id passthrough, got %q", got)
	}
	if len(runtime.lastConverseInput.Messages) != 1 {
		t.Fatalf("expected one message, got %d", len(runtime.lastConverseInput.Messages))
	}
	if len(runtime.lastConverseInput.System) != 1 {
		t.Fatalf("expected one system block, got %d", len(runtime.lastConverseInput.System))
	}
	if runtime.lastConverseInput.InferenceConfig == nil || runtime.lastConverseInput.InferenceConfig.MaxTokens == nil {
		t.Fatal("expected inference config to include max tokens")
	}
	if *runtime.lastConverseInput.InferenceConfig.MaxTokens != 64 {
		t.Fatalf("expected max tokens to map, got %d", *runtime.lastConverseInput.InferenceConfig.MaxTokens)
	}
	if runtime.lastConverseInput.InferenceConfig.Temperature == nil || *runtime.lastConverseInput.InferenceConfig.Temperature != float32(temperature) {
		t.Fatalf("expected temperature to map, got %v", runtime.lastConverseInput.InferenceConfig.Temperature)
	}
	if resp.Object != "response" {
		t.Fatalf("expected response object, got %q", resp.Object)
	}
	if resp.ID == "resp_" || resp.ID == "" {
		t.Fatalf("expected non-empty response id, got %q", resp.ID)
	}
	if len(resp.Output) != 1 || resp.Output[0].Content[0].Text != "hello back" {
		t.Fatalf("expected translated text output, got %+v", resp.Output)
	}
}

type fakeRuntime struct {
	lastConverseInput *bedrockruntime.ConverseInput
	converseOutput    *bedrockruntime.ConverseOutput
	converseErr       error
}

func (f *fakeRuntime) Converse(_ context.Context, input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	f.lastConverseInput = input
	return f.converseOutput, f.converseErr
}

func (f *fakeRuntime) ConverseStream(_ context.Context, _ *bedrockruntime.ConverseStreamInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error) {
	return nil, errors.New("not implemented in test")
}
