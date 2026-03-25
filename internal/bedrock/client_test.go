package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	bedrocksvc "github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrockcatalogtypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	bedrockdocument "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
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
		System: []string{"be precise"},
		Messages: []conversation.Message{{
			Role: "user",
			Blocks: []conversation.Block{
				{Type: conversation.BlockTypeText, Text: "hello"},
			},
		}},
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
	if len(resp.Output) != 1 || resp.Output[0].Type != OutputBlockTypeText || resp.Output[0].Text != "normalized reply" {
		t.Fatalf("expected text output block, got %#v", resp.Output)
	}
	if resp.ResponseID == "" {
		t.Fatalf("expected response id, got %q", resp.ResponseID)
	}
}

func TestClientRespondConversationBuildsToolAwareConverseInput(t *testing.T) {
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberText{Value: "done"},
					},
				},
			},
		},
	}
	client := &Client{runtime: runtime}

	resp, err := client.RespondConversation(context.Background(), "model-id", conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "assistant",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "Checking"},
					{
						Type: conversation.BlockTypeToolCall,
						ToolCall: &conversation.ToolCall{
							ID:        "call_123",
							Name:      "lookup",
							Arguments: `{"q":"weather"}`,
						},
					},
				},
			},
			{
				Role: "user",
				Blocks: []conversation.Block{
					{
						Type: conversation.BlockTypeToolResult,
						ToolResult: &conversation.ToolResult{
							CallID: "call_123",
							Output: map[string]any{"answer": "sunny"},
						},
					},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type:        "function",
				Name:        "lookup",
				Description: "Look up weather",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"q": map[string]any{"type": "string"},
					},
				},
			},
			{
				Type:    "web_search_preview",
				Name:    "__builtin_web_search_preview",
				BuiltIn: true,
				Config: map[string]json.RawMessage{
					"user_location": json.RawMessage(`{"type":"approximate","country":"DE"}`),
				},
			},
		},
		ToolChoice: conversation.ToolChoice{Type: "auto"},
	}, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Output) != 1 || resp.Output[0].Type != OutputBlockTypeText || resp.Output[0].Text != "done" {
		t.Fatalf("expected translated response output, got %#v", resp.Output)
	}
	if runtime.lastConverseInput == nil {
		t.Fatal("expected converse input to be captured")
	}
	if runtime.lastConverseInput.ToolConfig == nil {
		t.Fatal("expected tool config to be emitted")
	}
	if len(runtime.lastConverseInput.ToolConfig.Tools) != 2 {
		t.Fatalf("expected 2 tool specs, got %#v", runtime.lastConverseInput.ToolConfig.Tools)
	}
	if _, ok := runtime.lastConverseInput.ToolConfig.ToolChoice.(*bedrocktypes.ToolChoiceMemberAuto); !ok {
		t.Fatalf("expected auto tool choice, got %T", runtime.lastConverseInput.ToolConfig.ToolChoice)
	}
	if len(runtime.lastConverseInput.Messages) != 2 {
		t.Fatalf("expected tool-capable messages to map, got %#v", runtime.lastConverseInput.Messages)
	}

	assistant := runtime.lastConverseInput.Messages[0]
	if len(assistant.Content) != 2 {
		t.Fatalf("expected assistant content blocks, got %#v", assistant.Content)
	}
	toolUse, ok := assistant.Content[1].(*bedrocktypes.ContentBlockMemberToolUse)
	if !ok {
		t.Fatalf("expected assistant tool use block, got %T", assistant.Content[1])
	}
	if aws.ToString(toolUse.Value.ToolUseId) != "call_123" || aws.ToString(toolUse.Value.Name) != "lookup" {
		t.Fatalf("unexpected SDK tool use block: %#v", toolUse.Value)
	}
	toolUseInput, ok := decodeDocument(t, toolUse.Value.Input).(map[string]any)
	if !ok {
		t.Fatalf("expected decoded tool use input object, got %#v", toolUse.Value.Input)
	}
	if toolUseInput["q"] != "weather" {
		t.Fatalf("expected tool use input JSON to survive, got %#v", toolUseInput)
	}

	user := runtime.lastConverseInput.Messages[1]
	if len(user.Content) != 1 {
		t.Fatalf("expected user tool result block, got %#v", user.Content)
	}
	toolResult, ok := user.Content[0].(*bedrocktypes.ContentBlockMemberToolResult)
	if !ok {
		t.Fatalf("expected user tool result block, got %T", user.Content[0])
	}
	if aws.ToString(toolResult.Value.ToolUseId) != "call_123" {
		t.Fatalf("unexpected SDK tool result id: %#v", toolResult.Value)
	}
	if len(toolResult.Value.Content) != 1 {
		t.Fatalf("expected one SDK tool result content block, got %#v", toolResult.Value.Content)
	}
	jsonResult, ok := toolResult.Value.Content[0].(*bedrocktypes.ToolResultContentBlockMemberJson)
	if !ok {
		t.Fatalf("expected structured SDK tool result content, got %T", toolResult.Value.Content[0])
	}
	toolResultOutput, ok := decodeDocument(t, jsonResult.Value).(map[string]any)
	if !ok {
		t.Fatalf("expected decoded tool result object, got %#v", jsonResult.Value)
	}
	if toolResultOutput["answer"] != "sunny" {
		t.Fatalf("expected tool result output JSON to survive, got %#v", toolResultOutput)
	}

	functionTool, ok := runtime.lastConverseInput.ToolConfig.Tools[0].(*bedrocktypes.ToolMemberToolSpec)
	if !ok {
		t.Fatalf("expected first SDK tool spec, got %T", runtime.lastConverseInput.ToolConfig.Tools[0])
	}
	if aws.ToString(functionTool.Value.Name) != "lookup" {
		t.Fatalf("unexpected function tool name: %#v", functionTool.Value)
	}
	functionSchema, ok := decodeDocument(t, functionTool.Value.InputSchema.(*bedrocktypes.ToolInputSchemaMemberJson).Value).(map[string]any)
	if !ok {
		t.Fatalf("expected decoded function schema object, got %#v", functionTool.Value.InputSchema)
	}
	if functionSchema["type"] != "object" {
		t.Fatalf("expected function schema to survive, got %#v", functionSchema)
	}

	builtInTool, ok := runtime.lastConverseInput.ToolConfig.Tools[1].(*bedrocktypes.ToolMemberToolSpec)
	if !ok {
		t.Fatalf("expected second SDK tool spec, got %T", runtime.lastConverseInput.ToolConfig.Tools[1])
	}
	if aws.ToString(builtInTool.Value.Name) != "__builtin_web_search_preview" {
		t.Fatalf("unexpected synthetic tool name: %#v", builtInTool.Value)
	}
	builtInSchema, ok := decodeDocument(t, builtInTool.Value.InputSchema.(*bedrocktypes.ToolInputSchemaMemberJson).Value).(map[string]any)
	if !ok {
		t.Fatalf("expected decoded built-in schema object, got %#v", builtInTool.Value.InputSchema)
	}
	if builtInSchema["x-openai-tool-type"] != "web_search_preview" {
		t.Fatalf("expected built-in schema metadata, got %#v", builtInSchema)
	}
}

func TestClientChatBuildsConverseInputWithImageBlocks(t *testing.T) {
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberText{Value: "done"},
					},
				},
			},
		},
	}
	client := &Client{runtime: runtime}

	_, err := client.Chat(context.Background(), ConverseRequest{
		ModelID: "model-id",
		Messages: []Message{{
			Role: "user",
			Content: []ContentBlock{
				{Text: "describe this"},
				{Image: &ImageBlock{Format: "png", Bytes: []byte("hello")}},
			},
		}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if runtime.lastConverseInput == nil {
		t.Fatal("expected converse input to be captured")
	}
	if len(runtime.lastConverseInput.Messages) != 1 || len(runtime.lastConverseInput.Messages[0].Content) != 2 {
		t.Fatalf("expected mixed content blocks in SDK request, got %#v", runtime.lastConverseInput.Messages)
	}

	imageBlock, ok := runtime.lastConverseInput.Messages[0].Content[1].(*bedrocktypes.ContentBlockMemberImage)
	if !ok {
		t.Fatalf("expected SDK image content block, got %T", runtime.lastConverseInput.Messages[0].Content[1])
	}
	if imageBlock.Value.Format != bedrocktypes.ImageFormatPng {
		t.Fatalf("expected png SDK image format, got %#v", imageBlock.Value)
	}
	source, ok := imageBlock.Value.Source.(*bedrocktypes.ImageSourceMemberBytes)
	if !ok {
		t.Fatalf("expected byte-backed image source, got %T", imageBlock.Value.Source)
	}
	if !bytes.Equal(source.Value, []byte("hello")) {
		t.Fatalf("expected SDK image bytes to survive, got %#v", source.Value)
	}
}

func TestClientChatMapsAdvancedInferenceFieldsAndCachePoints(t *testing.T) {
	topP := float32(0.9)
	temperature := float32(0.2)
	maxTokens := int32(512)
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberText{Value: "done"},
					},
				},
			},
		},
	}
	client := &Client{runtime: runtime}

	_, err := client.Chat(context.Background(), ConverseRequest{
		ModelID:          "model-id",
		System:           []string{"cached system"},
		SystemCachePoint: true,
		Messages: []Message{{
			Role: "user",
			Content: []ContentBlock{
				{Text: "cached user"},
				{CachePoint: &CachePointBlock{Type: "default"}},
			},
		}},
		MaxTokens:   &maxTokens,
		Temperature: &temperature,
		TopP:        &topP,
		StopSequences: []string{
			"END",
		},
		AdditionalModelRequestFields: map[string]any{
			"thinking": map[string]any{"type": "enabled"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if runtime.lastConverseInput == nil || runtime.lastConverseInput.InferenceConfig == nil {
		t.Fatalf("expected converse input with inference config, got %#v", runtime.lastConverseInput)
	}
	if runtime.lastConverseInput.InferenceConfig.TopP == nil || *runtime.lastConverseInput.InferenceConfig.TopP != topP {
		t.Fatalf("expected top_p to map, got %#v", runtime.lastConverseInput.InferenceConfig)
	}
	if len(runtime.lastConverseInput.InferenceConfig.StopSequences) != 1 || runtime.lastConverseInput.InferenceConfig.StopSequences[0] != "END" {
		t.Fatalf("expected stop sequences to map, got %#v", runtime.lastConverseInput.InferenceConfig)
	}
	if len(runtime.lastConverseInput.System) != 2 {
		t.Fatalf("expected system text plus cache point, got %#v", runtime.lastConverseInput.System)
	}
	if _, ok := runtime.lastConverseInput.System[1].(*bedrocktypes.SystemContentBlockMemberCachePoint); !ok {
		t.Fatalf("expected system cache point block, got %T", runtime.lastConverseInput.System[1])
	}
	if len(runtime.lastConverseInput.Messages) != 1 || len(runtime.lastConverseInput.Messages[0].Content) != 2 {
		t.Fatalf("expected message cache point block, got %#v", runtime.lastConverseInput.Messages)
	}
	if _, ok := runtime.lastConverseInput.Messages[0].Content[1].(*bedrocktypes.ContentBlockMemberCachePoint); !ok {
		t.Fatalf("expected user cache point block, got %T", runtime.lastConverseInput.Messages[0].Content[1])
	}
	if runtime.lastConverseInput.AdditionalModelRequestFields == nil {
		t.Fatal("expected additional model request fields to map")
	}
	additional, ok := decodeDocument(t, runtime.lastConverseInput.AdditionalModelRequestFields).(map[string]any)
	if !ok || additional["thinking"] == nil {
		t.Fatalf("expected additional request fields to survive, got %#v", runtime.lastConverseInput.AdditionalModelRequestFields)
	}
}

func TestClientRespondConversationParsesMixedTextAndToolUseOutput(t *testing.T) {
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberText{Value: "Checking"},
						&bedrocktypes.ContentBlockMemberToolUse{
							Value: bedrocktypes.ToolUseBlock{
								ToolUseId: aws.String("call_123"),
								Name:      aws.String("lookup"),
								Input:     bedrockdocument.NewLazyDocument(map[string]any{"q": "weather"}),
							},
						},
						&bedrocktypes.ContentBlockMemberText{Value: "Waiting"},
					},
				},
			},
		},
	}
	client := &Client{runtime: runtime}

	resp, err := client.RespondConversation(context.Background(), "model-id", conversation.Request{
		Messages: []conversation.Message{{
			Role: "user",
			Blocks: []conversation.Block{
				{Type: conversation.BlockTypeText, Text: "hello"},
			},
		}},
	}, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Output) != 3 {
		t.Fatalf("expected 3 output blocks, got %#v", resp.Output)
	}
	if resp.Output[0].Type != OutputBlockTypeText || resp.Output[0].Text != "Checking" {
		t.Fatalf("expected first text block, got %#v", resp.Output[0])
	}
	if resp.Output[1].Type != OutputBlockTypeToolCall || resp.Output[1].ToolCall == nil {
		t.Fatalf("expected tool call block, got %#v", resp.Output[1])
	}
	if resp.Output[1].ToolCall.ID != "call_123" || resp.Output[1].ToolCall.Name != "lookup" {
		t.Fatalf("expected tool call metadata, got %#v", resp.Output[1].ToolCall)
	}
	if resp.Output[1].ToolCall.Arguments != `{"q":"weather"}` {
		t.Fatalf("expected tool call arguments JSON, got %q", resp.Output[1].ToolCall.Arguments)
	}
	if resp.Output[2].Type != OutputBlockTypeText || resp.Output[2].Text != "Waiting" {
		t.Fatalf("expected trailing text block, got %#v", resp.Output[2])
	}
}

func TestClientRespondConversationIgnoresReasoningBlocks(t *testing.T) {
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberReasoningContent{
							Value: &bedrocktypes.ReasoningContentBlockMemberReasoningText{
								Value: bedrocktypes.ReasoningTextBlock{
									Text:      aws.String("private reasoning"),
									Signature: aws.String("sig"),
								},
							},
						},
						&bedrocktypes.ContentBlockMemberText{Value: "visible answer"},
					},
				},
			},
		},
	}
	client := &Client{runtime: runtime}

	resp, err := client.RespondConversation(context.Background(), "model-id", conversation.Request{
		Messages: []conversation.Message{{
			Role: "user",
			Blocks: []conversation.Block{
				{Type: conversation.BlockTypeText, Text: "hello"},
			},
		}},
	}, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Output) != 1 {
		t.Fatalf("expected only visible assistant output, got %#v", resp.Output)
	}
	if resp.Output[0].Type != OutputBlockTypeText || resp.Output[0].Text != "visible answer" {
		t.Fatalf("expected visible text output only, got %#v", resp.Output[0])
	}
}

func TestClientRespondConversationPreservesEmptyStringToolResultAsText(t *testing.T) {
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberText{Value: "done"},
					},
				},
			},
		},
	}
	client := &Client{runtime: runtime}

	_, err := client.RespondConversation(context.Background(), "model-id", conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{
						Type: conversation.BlockTypeToolResult,
						ToolResult: &conversation.ToolResult{
							CallID: "call_123",
							Output: "",
						},
					},
				},
			},
		},
	}, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	toolResult, ok := runtime.lastConverseInput.Messages[0].Content[0].(*bedrocktypes.ContentBlockMemberToolResult)
	if !ok {
		t.Fatalf("expected tool result block, got %T", runtime.lastConverseInput.Messages[0].Content[0])
	}
	if len(toolResult.Value.Content) != 1 {
		t.Fatalf("expected one tool result content block, got %#v", toolResult.Value.Content)
	}
	textResult, ok := toolResult.Value.Content[0].(*bedrocktypes.ToolResultContentBlockMemberText)
	if !ok {
		t.Fatalf("expected empty string to remain text content, got %T", toolResult.Value.Content[0])
	}
	if textResult.Value != "" {
		t.Fatalf("expected empty text result, got %q", textResult.Value)
	}
}

func TestClientRespondConversationOmitsEmptyToolDescription(t *testing.T) {
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberText{Value: "done"},
					},
				},
			},
		},
	}
	client := &Client{runtime: runtime}

	_, err := client.RespondConversation(context.Background(), "model-id", conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "user",
				Blocks: []conversation.Block{
					{Type: conversation.BlockTypeText, Text: "hello"},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type: "function",
				Name: "lookup",
				Parameters: map[string]any{
					"type": "object",
				},
			},
		},
	}, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	functionTool, ok := runtime.lastConverseInput.ToolConfig.Tools[0].(*bedrocktypes.ToolMemberToolSpec)
	if !ok {
		t.Fatalf("expected tool spec, got %T", runtime.lastConverseInput.ToolConfig.Tools[0])
	}
	if functionTool.Value.Description != nil {
		t.Fatalf("expected empty description to be omitted, got %#v", functionTool.Value.Description)
	}
}

func TestToSDKToolResultContentPanicsOnUnknownInternalType(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for unknown internal tool result content type")
		}
	}()

	toSDKToolResultContent([]ToolResultContentBlock{
		{
			Type: "unknown",
			JSON: map[string]any{"ignored": true},
		},
	})
}

func TestClientStreamConversationRejectsNilResponse(t *testing.T) {
	client := &Client{runtime: &fakeRuntime{}}
	_, err := client.StreamConversation(context.Background(), "model-id", conversation.Request{
		Messages: []conversation.Message{{
			Role: "user",
			Blocks: []conversation.Block{
				{Type: conversation.BlockTypeText, Text: "hello"},
			},
		}},
	}, nil, nil, httptest.NewRecorder())
	if err == nil {
		t.Fatal("expected error when bedrock stream response is nil")
	}
}

func TestClientChatStreamRejectsNilResponse(t *testing.T) {
	client := &Client{runtime: &fakeRuntime{}}
	_, err := client.ChatStream(context.Background(), ConverseRequest{
		ModelID: "model-id",
		Messages: []Message{
			{
				Role: "user",
				Content: []ContentBlock{
					{Text: "hello"},
				},
			},
		},
	})
	if err == nil {
		t.Fatal("expected error when bedrock stream response is nil")
	}
}

func TestClientChatStreamBuildsConverseStreamInput(t *testing.T) {
	events := make(chan bedrocktypes.ConverseStreamOutput, 1)
	events <- &bedrocktypes.ConverseStreamOutputMemberMessageStop{
		Value: bedrocktypes.MessageStopEvent{StopReason: bedrocktypes.StopReasonEndTurn},
	}
	close(events)

	stream := &fakeStream{events: events}
	runtime := &fakeRuntime{converseStreamOutput: &bedrockruntime.ConverseStreamOutput{}}
	client := &Client{
		runtime: runtime,
		streamAdapter: func(*bedrockruntime.ConverseStreamOutput) (streamEvents, error) {
			return stream, nil
		},
	}

	resp, err := client.ChatStream(context.Background(), ConverseRequest{
		ModelID: "model-id",
		System:  []string{"be terse"},
		Messages: []Message{
			{
				Role: "user",
				Content: []ContentBlock{
					{Text: "hello"},
				},
			},
		},
	})
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
		t.Fatalf("expected one system block, got %d", len(runtime.lastConverseStreamInput.System))
	}
	if resp.Stream != stream {
		t.Fatalf("expected stream to pass through, got %#v", resp.Stream)
	}
	if resp.ResponseID == "" {
		t.Fatalf("expected response id to be set, got %q", resp.ResponseID)
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
		Messages: []conversation.Message{{
			Role: "user",
			Blocks: []conversation.Block{
				{Type: conversation.BlockTypeText, Text: "hello"},
			},
		}},
	}, nil, nil, recorder)
	if !errors.Is(err, streamErr) {
		t.Fatalf("expected stream error, got %v", err)
	}
	if len(resp.Output) != 1 || resp.Output[0].Type != OutputBlockTypeText || resp.Output[0].Text != "hello world" {
		t.Fatalf("expected accumulated text output, got %#v", resp.Output)
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

func TestClientStreamConversationBuildsToolAwareConverseStreamInput(t *testing.T) {
	events := make(chan bedrocktypes.ConverseStreamOutput, 1)
	events <- &bedrocktypes.ConverseStreamOutputMemberMessageStop{
		Value: bedrocktypes.MessageStopEvent{StopReason: bedrocktypes.StopReasonEndTurn},
	}
	close(events)

	runtime := &fakeRuntime{converseStreamOutput: &bedrockruntime.ConverseStreamOutput{}}
	client := &Client{
		runtime: runtime,
		streamAdapter: func(*bedrockruntime.ConverseStreamOutput) (streamEvents, error) {
			return &fakeStream{events: events}, nil
		},
	}

	_, err := client.StreamConversation(context.Background(), "model-id", conversation.Request{
		Messages: []conversation.Message{
			{
				Role: "assistant",
				Blocks: []conversation.Block{
					{
						Type: conversation.BlockTypeToolCall,
						ToolCall: &conversation.ToolCall{
							ID:        "call_123",
							Name:      "lookup",
							Arguments: `{"q":"weather"}`,
						},
					},
				},
			},
		},
		Tools: []conversation.ToolDefinition{
			{
				Type: "function",
				Name: "lookup",
				Parameters: map[string]any{
					"type": "object",
				},
			},
		},
		ToolChoice: conversation.ToolChoice{Type: "auto"},
	}, nil, nil, httptest.NewRecorder())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if runtime.lastConverseStreamInput == nil {
		t.Fatal("expected converse stream input to be captured")
	}
	if runtime.lastConverseStreamInput.ToolConfig == nil {
		t.Fatal("expected tool config on stream input")
	}
	if _, ok := runtime.lastConverseStreamInput.ToolConfig.ToolChoice.(*bedrocktypes.ToolChoiceMemberAuto); !ok {
		t.Fatalf("expected auto tool choice on stream input, got %T", runtime.lastConverseStreamInput.ToolConfig.ToolChoice)
	}
	toolUse, ok := runtime.lastConverseStreamInput.Messages[0].Content[0].(*bedrocktypes.ContentBlockMemberToolUse)
	if !ok {
		t.Fatalf("expected tool use block on stream input, got %T", runtime.lastConverseStreamInput.Messages[0].Content[0])
	}
	toolUseInput, ok := decodeDocument(t, toolUse.Value.Input).(map[string]any)
	if !ok {
		t.Fatalf("expected decoded stream tool input object, got %#v", toolUse.Value.Input)
	}
	if toolUseInput["q"] != "weather" {
		t.Fatalf("expected stream tool input JSON to survive, got %#v", toolUseInput)
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
		systemProfilesOutput: &bedrocksvc.ListInferenceProfilesOutput{
			InferenceProfileSummaries: []bedrockcatalogtypes.InferenceProfileSummary{
				{
					InferenceProfileId: aws.String("us.anthropic.claude-3-7-sonnet-20250219-v1:0"),
					Models: []bedrockcatalogtypes.InferenceProfileModel{
						{ModelArn: aws.String("arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-7-sonnet-20250219-v1:0")},
					},
				},
			},
		},
		applicationProfilesOutput: &bedrocksvc.ListInferenceProfilesOutput{
			InferenceProfileSummaries: []bedrockcatalogtypes.InferenceProfileSummary{
				{
					InferenceProfileId: aws.String("arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/app-profile"),
					Models: []bedrockcatalogtypes.InferenceProfileModel{
						{ModelArn: aws.String("arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0")},
					},
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
	if len(catalog.inferenceInputs) != 2 {
		t.Fatalf("expected two inference profile list calls, got %d", len(catalog.inferenceInputs))
	}
	typesRequested := map[bedrockcatalogtypes.InferenceProfileType]bool{}
	for _, input := range catalog.inferenceInputs {
		typesRequested[input.TypeEquals] = true
	}
	if !typesRequested[bedrockcatalogtypes.InferenceProfileTypeSystemDefined] {
		t.Fatalf("expected system-defined inference profile listing request, got %#v", catalog.inferenceInputs)
	}
	if !typesRequested[bedrockcatalogtypes.InferenceProfileTypeApplication] {
		t.Fatalf("expected application inference profile listing request, got %#v", catalog.inferenceInputs)
	}
	if len(got) != 4 {
		t.Fatalf("expected merged foundation and profile models, got %d", len(got))
	}
	seen := map[string]bool{}
	for _, model := range got {
		seen[model.ID] = true
	}
	if !seen["anthropic.claude-3-7-sonnet-20250219-v1:0"] {
		t.Fatalf("expected foundation model id in output, got %#v", got)
	}
	if !seen["amazon.nova-pro-v1:0"] {
		t.Fatalf("expected second foundation model id in output, got %#v", got)
	}
	if !seen["us.anthropic.claude-3-7-sonnet-20250219-v1:0"] {
		t.Fatalf("expected system profile id in output, got %#v", got)
	}
	if !seen["arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/app-profile"] {
		t.Fatalf("expected application profile id in output, got %#v", got)
	}
}

func TestClientLookupModelResolvesCatalogRecord(t *testing.T) {
	client := &Client{catalog: &fakeCatalog{
		output: &bedrocksvc.ListFoundationModelsOutput{
			ModelSummaries: []bedrockcatalogtypes.FoundationModelSummary{
				{
					ModelId:      aws.String("anthropic.claude-3-7-sonnet-20250219-v1:0"),
					ProviderName: aws.String("Anthropic"),
					ModelName:    aws.String("Claude 3.7 Sonnet"),
				},
			},
		},
		systemProfilesOutput: &bedrocksvc.ListInferenceProfilesOutput{
			InferenceProfileSummaries: []bedrockcatalogtypes.InferenceProfileSummary{
				{
					InferenceProfileId: aws.String("us.anthropic.claude-3-7-sonnet-20250219-v1:0"),
					Models: []bedrockcatalogtypes.InferenceProfileModel{
						{ModelArn: aws.String("arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-7-sonnet-20250219-v1:0")},
					},
				},
			},
		},
		applicationProfilesOutput: &bedrocksvc.ListInferenceProfilesOutput{},
	}}

	got, err := client.LookupModel(context.Background(), "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ID != "us.anthropic.claude-3-7-sonnet-20250219-v1:0" {
		t.Fatalf("expected profile id to resolve, got %#v", got)
	}
	if got.ResolvedFoundationModelID != "anthropic.claude-3-7-sonnet-20250219-v1:0" {
		t.Fatalf("expected resolved foundation model id, got %#v", got)
	}
}

func TestClientLookupModelRejectsUnknownModel(t *testing.T) {
	client := &Client{catalog: &fakeCatalog{
		output:                    &bedrocksvc.ListFoundationModelsOutput{},
		systemProfilesOutput:      &bedrocksvc.ListInferenceProfilesOutput{},
		applicationProfilesOutput: &bedrocksvc.ListInferenceProfilesOutput{},
	}}

	_, err := client.LookupModel(context.Background(), "missing-model")
	if err == nil {
		t.Fatal("expected unknown model error")
	}
	var invalidRequest openai.InvalidRequestError
	if !errors.As(err, &invalidRequest) {
		t.Fatalf("expected invalid request error, got %T", err)
	}
}

func TestClientChatBuildsConverseInputAndParsesReasoningBlocks(t *testing.T) {
	runtime := &fakeRuntime{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &bedrocktypes.ConverseOutputMemberMessage{
				Value: bedrocktypes.Message{
					Content: []bedrocktypes.ContentBlock{
						&bedrocktypes.ContentBlockMemberReasoningContent{
							Value: &bedrocktypes.ReasoningContentBlockMemberReasoningText{
								Value: bedrocktypes.ReasoningTextBlock{
									Text:      aws.String("draft reasoning"),
									Signature: aws.String("sig"),
								},
							},
						},
						&bedrocktypes.ContentBlockMemberText{Value: "final answer"},
					},
				},
			},
			StopReason: bedrocktypes.StopReasonEndTurn,
		},
	}
	client := &Client{runtime: runtime}

	resp, err := client.Chat(context.Background(), ConverseRequest{
		ModelID: "model-id",
		System:  []string{"be terse"},
		Messages: []Message{
			{
				Role: "user",
				Content: []ContentBlock{
					{Text: "hello"},
				},
			},
		},
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
	if len(resp.Output) != 2 {
		t.Fatalf("expected reasoning and text output blocks, got %#v", resp.Output)
	}
	if resp.Output[0].Type != OutputBlockTypeReasoning || resp.Output[0].Text != "draft reasoning" {
		t.Fatalf("expected reasoning output block, got %#v", resp.Output[0])
	}
	if resp.Output[1].Type != OutputBlockTypeText || resp.Output[1].Text != "final answer" {
		t.Fatalf("expected text output block, got %#v", resp.Output[1])
	}
	if resp.StopReason != string(bedrocktypes.StopReasonEndTurn) {
		t.Fatalf("expected stop reason to map, got %#v", resp)
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
	lastInput                 *bedrocksvc.ListFoundationModelsInput
	inferenceInputs           []*bedrocksvc.ListInferenceProfilesInput
	output                    *bedrocksvc.ListFoundationModelsOutput
	systemProfilesOutput      *bedrocksvc.ListInferenceProfilesOutput
	applicationProfilesOutput *bedrocksvc.ListInferenceProfilesOutput
	err                       error
}

func (f *fakeCatalog) ListFoundationModels(_ context.Context, input *bedrocksvc.ListFoundationModelsInput, _ ...func(*bedrocksvc.Options)) (*bedrocksvc.ListFoundationModelsOutput, error) {
	f.lastInput = input
	return f.output, f.err
}

func (f *fakeCatalog) ListInferenceProfiles(_ context.Context, input *bedrocksvc.ListInferenceProfilesInput, _ ...func(*bedrocksvc.Options)) (*bedrocksvc.ListInferenceProfilesOutput, error) {
	f.inferenceInputs = append(f.inferenceInputs, input)
	switch input.TypeEquals {
	case bedrockcatalogtypes.InferenceProfileTypeSystemDefined:
		return f.systemProfilesOutput, f.err
	case bedrockcatalogtypes.InferenceProfileTypeApplication:
		return f.applicationProfilesOutput, f.err
	default:
		return &bedrocksvc.ListInferenceProfilesOutput{}, f.err
	}
}

func decodeDocument(t *testing.T, doc bedrockdocument.Interface) any {
	t.Helper()
	raw, err := doc.MarshalSmithyDocument()
	if err != nil {
		t.Fatalf("failed to marshal Smithy document: %v", err)
	}
	var decoded any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("failed to decode Smithy document JSON: %v", err)
	}
	return decoded
}
