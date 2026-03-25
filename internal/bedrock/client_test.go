package bedrock

import (
	"context"
	"errors"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	bedrocksvc "github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrockcatalogtypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/moolen/openai-bedrock-proxy/internal/conversation"
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

func TestClientRespondConversationBuildsConverseInputAndParsesText(t *testing.T) {
	maxTokens := 32
	temperature := 0.9
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberText{Value: "normalized reply"},
					},
				},
			},
		},
	}
	client := &Client{runtime: runtime}

	resp, err := client.RespondConversation(context.Background(), "model-id", conversation.Request{
		System:   []string{"be precise"},
		Messages: []conversation.Message{{Role: "user", Text: "hello"}},
	}, &maxTokens, &temperature)
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
	if *runtime.lastConverseInput.InferenceConfig.MaxTokens != int32(maxTokens) {
		t.Fatalf("expected max tokens to map, got %d", *runtime.lastConverseInput.InferenceConfig.MaxTokens)
	}
	if runtime.lastConverseInput.InferenceConfig.Temperature == nil || *runtime.lastConverseInput.InferenceConfig.Temperature != float32(temperature) {
		t.Fatalf("expected temperature to map, got %v", runtime.lastConverseInput.InferenceConfig.Temperature)
	}
	if resp.Text != "normalized reply" {
		t.Fatalf("expected response text, got %q", resp.Text)
	}
	if resp.ResponseID == "" {
		t.Fatalf("expected response id, got %q", resp.ResponseID)
	}
}

func TestClientStreamConversationRejectsNilResponse(t *testing.T) {
	client := &Client{runtime: &fakeRuntime{}}
	_, err := client.StreamConversation(context.Background(), "model-id", conversation.Request{
		Messages: []conversation.Message{{Role: "user", Text: "hello"}},
	}, nil, nil, httptest.NewRecorder())
	if err == nil {
		t.Fatal("expected error when bedrock stream response is nil")
	}
}

func TestClientStreamConversationAccumulatesTextAndStopReason(t *testing.T) {
	streamErr := errors.New("stream failure")
	events := make(chan bedrocktypes.ConverseStreamOutput, 3)
	events <- &bedrocktypes.ConverseStreamOutputMemberContentBlockDelta{
		Value: bedrocktypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(0),
			Delta:             &bedrocktypes.ContentBlockDeltaMemberText{Value: "hello"},
		},
	}
	events <- &bedrocktypes.ConverseStreamOutputMemberContentBlockDelta{
		Value: bedrocktypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(0),
			Delta:             &bedrocktypes.ContentBlockDeltaMemberText{Value: " world"},
		},
	}
	events <- &bedrocktypes.ConverseStreamOutputMemberMessageStop{
		Value: bedrocktypes.MessageStopEvent{StopReason: bedrocktypes.StopReasonEndTurn},
	}
	close(events)

	stream := &fakeStream{events: events, err: streamErr}
	client := &Client{
		runtime: &fakeRuntime{converseStreamOutput: &bedrockruntime.ConverseStreamOutput{}},
		streamAdapter: func(*bedrockruntime.ConverseStreamOutput) (streamEvents, error) {
			return stream, nil
		},
	}
	recorder := httptest.NewRecorder()

	resp, err := client.StreamConversation(context.Background(), "model-id", conversation.Request{
		Messages: []conversation.Message{{Role: "user", Text: "hello"}},
	}, nil, nil, recorder)
	if !errors.Is(err, streamErr) {
		t.Fatalf("expected stream error, got %v", err)
	}
	if resp.Text != "hello world" {
		t.Fatalf("expected accumulated text, got %q", resp.Text)
	}
	if resp.StopReason != string(bedrocktypes.StopReasonEndTurn) {
		t.Fatalf("expected stop reason to map, got %q", resp.StopReason)
	}
	if resp.ResponseID == "" {
		t.Fatalf("expected response id to be set, got %q", resp.ResponseID)
	}
	expected := "event: response.output_text.delta\ndata: {\"delta\":\"hello\"}\n\n" +
		"event: response.output_text.delta\ndata: {\"delta\":\" world\"}\n\n" +
		"event: response.completed\ndata: {\"status\":\"end_turn\"}\n\n"
	if got := recorder.Body.String(); got != expected {
		t.Fatalf("expected SSE body %q, got %q", expected, got)
	}
}

func TestClientStreamWrapperNormalizesAndStreams(t *testing.T) {
	events := make(chan bedrocktypes.ConverseStreamOutput, 2)
	events <- &bedrocktypes.ConverseStreamOutputMemberContentBlockDelta{
		Value: bedrocktypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(0),
			Delta:             &bedrocktypes.ContentBlockDeltaMemberText{Value: "hi"},
		},
	}
	events <- &bedrocktypes.ConverseStreamOutputMemberMessageStop{
		Value: bedrocktypes.MessageStopEvent{StopReason: bedrocktypes.StopReasonEndTurn},
	}
	close(events)

	stream := &fakeStream{events: events, err: nil}
	runtime := &fakeRuntime{converseStreamOutput: &bedrockruntime.ConverseStreamOutput{}}
	client := &Client{
		runtime: runtime,
		streamAdapter: func(*bedrockruntime.ConverseStreamOutput) (streamEvents, error) {
			return stream, nil
		},
	}
	recorder := httptest.NewRecorder()

	err := client.Stream(context.Background(), openai.ResponsesRequest{
		Model:        "model-id",
		Instructions: "be terse",
		Input:        "hello",
	}, recorder)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if runtime.lastConverseStreamInput == nil {
		t.Fatal("expected converse stream input to be captured")
	}
	if got := aws.ToString(runtime.lastConverseStreamInput.ModelId); got != "model-id" {
		t.Fatalf("expected model id passthrough, got %q", got)
	}
	if len(runtime.lastConverseStreamInput.System) != 1 {
		t.Fatalf("expected system to normalize, got %d", len(runtime.lastConverseStreamInput.System))
	}
	if len(runtime.lastConverseStreamInput.Messages) != 1 {
		t.Fatalf("expected one user message, got %d", len(runtime.lastConverseStreamInput.Messages))
	}
	expected := "event: response.output_text.delta\ndata: {\"delta\":\"hi\"}\n\n" +
		"event: response.completed\ndata: {\"status\":\"end_turn\"}\n\n"
	if got := recorder.Body.String(); got != expected {
		t.Fatalf("expected SSE body %q, got %q", expected, got)
	}
}

func TestClientListModelsUsesBedrockCatalog(t *testing.T) {
	catalog := &fakeCatalog{
		output: &bedrocksvc.ListFoundationModelsOutput{
			ModelSummaries: []bedrockcatalogtypes.FoundationModelSummary{
				{
					ModelId:      aws.String("anthropic.claude-3-7-sonnet-20250219-v1:0"),
					ProviderName: aws.String("Anthropic"),
					ModelName:    aws.String("Claude 3.7 Sonnet"),
				},
				{
					ModelId:      aws.String("amazon.nova-pro-v1:0"),
					ProviderName: aws.String("Amazon"),
					ModelName:    aws.String("Nova Pro"),
				},
			},
		},
	}
	client := &Client{catalog: catalog}

	got, err := client.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if catalog.lastInput == nil {
		t.Fatal("expected list foundation models input to be captured")
	}
	if catalog.lastInput.ByOutputModality != bedrockcatalogtypes.ModelModalityText {
		t.Fatalf("expected output modality filter to be text, got %q", catalog.lastInput.ByOutputModality)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 models, got %d", len(got))
	}
	if got[0].ID != "anthropic.claude-3-7-sonnet-20250219-v1:0" {
		t.Fatalf("expected first model id to map, got %q", got[0].ID)
	}
	if got[0].Provider != "Anthropic" {
		t.Fatalf("expected provider to map, got %q", got[0].Provider)
	}
	if got[0].Name != "Claude 3.7 Sonnet" {
		t.Fatalf("expected name to map, got %q", got[0].Name)
	}
}

type fakeRuntime struct {
	lastConverseInput       *bedrockruntime.ConverseInput
	converseOutput          *bedrockruntime.ConverseOutput
	converseErr             error
	lastConverseStreamInput *bedrockruntime.ConverseStreamInput
	converseStreamOutput    *bedrockruntime.ConverseStreamOutput
	converseStreamErr       error
}

func (f *fakeRuntime) Converse(_ context.Context, input *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	f.lastConverseInput = input
	return f.converseOutput, f.converseErr
}

func (f *fakeRuntime) ConverseStream(_ context.Context, input *bedrockruntime.ConverseStreamInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error) {
	f.lastConverseStreamInput = input
	if f.converseStreamOutput == nil && f.converseStreamErr == nil {
		return nil, nil
	}
	return f.converseStreamOutput, f.converseStreamErr
}

type fakeStream struct {
	events chan bedrocktypes.ConverseStreamOutput
	err    error
}

func (f *fakeStream) Events() <-chan bedrocktypes.ConverseStreamOutput {
	return f.events
}

func (f *fakeStream) Close() error {
	return nil
}

func (f *fakeStream) Err() error {
	return f.err
}

type fakeCatalog struct {
	lastInput *bedrocksvc.ListFoundationModelsInput
	output    *bedrocksvc.ListFoundationModelsOutput
	err       error
}

func (f *fakeCatalog) ListFoundationModels(_ context.Context, input *bedrocksvc.ListFoundationModelsInput, _ ...func(*bedrocksvc.Options)) (*bedrocksvc.ListFoundationModelsOutput, error) {
	f.lastInput = input
	return f.output, f.err
}
