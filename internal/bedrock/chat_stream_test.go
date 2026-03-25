package bedrock

import (
	"bytes"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func TestWriteChatStreamEmitsTextChunksAndDone(t *testing.T) {
	stream := newFakeChatStream(
		textDelta("hel"),
		textDelta("lo"),
		messageStop("end_turn"),
		metadataUsage(10, 2),
	)
	var buf bytes.Buffer

	usage, err := WriteChatCompletionsStream(stream, "chatcmpl_123", "model", true, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if usage == nil || usage.TotalTokens != 12 {
		t.Fatalf("expected returned usage, got %#v", usage)
	}

	got := buf.String()
	if !strings.Contains(got, "\"object\":\"chat.completion.chunk\"") {
		t.Fatalf("expected chunk output, got %s", got)
	}
	if !strings.Contains(got, "\"content\":\"hel\"") || !strings.Contains(got, "\"content\":\"lo\"") {
		t.Fatalf("expected text deltas, got %s", got)
	}
	if !strings.Contains(got, "\"finish_reason\":\"stop\"") {
		t.Fatalf("expected stop finish reason, got %s", got)
	}
	if !strings.HasSuffix(got, "data: [DONE]\n\n") {
		t.Fatalf("expected terminal done marker, got %s", got)
	}
}

func TestWriteChatStreamEmitsReasoningDeltaSeparately(t *testing.T) {
	stream := newFakeChatStream(
		reasoningDelta("think"),
		messageStop("end_turn"),
	)
	var buf bytes.Buffer

	_, err := WriteChatCompletionsStream(stream, "chatcmpl_123", "model", false, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "\"reasoning_content\":\"think\"") {
		t.Fatalf("expected reasoning delta, got %s", buf.String())
	}
}

func TestWriteChatStreamEmitsToolCallAndUsageChunks(t *testing.T) {
	stream := newFakeChatStream(
		toolUseStart(0, "call_123", "lookup"),
		toolUseDelta(0, `{"q":"weather"}`),
		messageStop("tool_use"),
		metadataUsage(10, 2),
	)
	var buf bytes.Buffer

	usage, err := WriteChatCompletionsStream(stream, "chatcmpl_123", "model", true, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if usage == nil || usage.TotalTokens != 12 {
		t.Fatalf("expected returned usage, got %#v", usage)
	}

	got := buf.String()
	if !strings.Contains(got, "\"tool_calls\":[{\"index\":0") {
		t.Fatalf("expected indexed tool call delta, got %s", got)
	}
	if !strings.Contains(got, "\"id\":\"call_123\"") || !strings.Contains(got, "\"name\":\"lookup\"") {
		t.Fatalf("expected tool call identity delta, got %s", got)
	}
	if !strings.Contains(got, "\"arguments\":\"{\\\"q\\\":\\\"weather\\\"}\"") {
		t.Fatalf("expected tool call arguments delta, got %s", got)
	}
	if strings.Contains(got, "\"name\":\"\"") {
		t.Fatalf("expected argument-only tool-call deltas to omit empty function.name, got %s", got)
	}
	if !strings.Contains(got, "\"finish_reason\":\"tool_calls\"") {
		t.Fatalf("expected tool finish reason, got %s", got)
	}
	if !strings.Contains(got, "\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}") {
		t.Fatalf("expected usage chunk with empty choices, got %s", got)
	}
}

func TestWriteChatStreamSkipsUsageChunkWhenDisabled(t *testing.T) {
	stream := newFakeChatStream(
		textDelta("hello"),
		messageStop("end_turn"),
		metadataUsage(10, 2),
	)
	var buf bytes.Buffer

	usage, err := WriteChatCompletionsStream(stream, "chatcmpl_123", "model", false, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if usage != nil {
		t.Fatalf("expected usage to be omitted when disabled, got %#v", usage)
	}
	if strings.Contains(buf.String(), "\"usage\":") {
		t.Fatalf("expected usage to be omitted when disabled, got %s", buf.String())
	}
}

func TestWriteChatStreamEmitsAssistantRoleBeforeStopWithoutContent(t *testing.T) {
	stream := newFakeChatStream(
		messageStop("end_turn"),
	)
	var buf bytes.Buffer

	_, err := WriteChatCompletionsStream(stream, "chatcmpl_123", "model", false, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := buf.String()
	rolePos := strings.Index(got, "\"role\":\"assistant\"")
	finishPos := strings.Index(got, "\"finish_reason\":\"stop\"")
	if rolePos == -1 {
		t.Fatalf("expected assistant role chunk before stop-only stream completion, got %s", got)
	}
	if finishPos == -1 {
		t.Fatalf("expected stop finish reason, got %s", got)
	}
	if rolePos > finishPos {
		t.Fatalf("expected assistant role chunk before finish chunk, got %s", got)
	}
}

type fakeChatStream struct {
	events chan bedrocktypes.ConverseStreamOutput
	err    error
}

func newFakeChatStream(events ...bedrocktypes.ConverseStreamOutput) *fakeChatStream {
	ch := make(chan bedrocktypes.ConverseStreamOutput, len(events))
	for _, event := range events {
		ch <- event
	}
	close(ch)
	return &fakeChatStream{events: ch}
}

func (f *fakeChatStream) Events() <-chan bedrocktypes.ConverseStreamOutput {
	return f.events
}

func (f *fakeChatStream) Close() error {
	return nil
}

func (f *fakeChatStream) Err() error {
	return f.err
}

func textDelta(value string) bedrocktypes.ConverseStreamOutput {
	return &bedrocktypes.ConverseStreamOutputMemberContentBlockDelta{
		Value: bedrocktypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(0),
			Delta:             &bedrocktypes.ContentBlockDeltaMemberText{Value: value},
		},
	}
}

func reasoningDelta(value string) bedrocktypes.ConverseStreamOutput {
	return &bedrocktypes.ConverseStreamOutputMemberContentBlockDelta{
		Value: bedrocktypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(0),
			Delta: &bedrocktypes.ContentBlockDeltaMemberReasoningContent{
				Value: &bedrocktypes.ReasoningContentBlockDeltaMemberText{Value: value},
			},
		},
	}
}

func toolUseStart(index int32, id string, name string) bedrocktypes.ConverseStreamOutput {
	return &bedrocktypes.ConverseStreamOutputMemberContentBlockStart{
		Value: bedrocktypes.ContentBlockStartEvent{
			ContentBlockIndex: aws.Int32(index),
			Start: &bedrocktypes.ContentBlockStartMemberToolUse{
				Value: bedrocktypes.ToolUseBlockStart{
					ToolUseId: aws.String(id),
					Name:      aws.String(name),
				},
			},
		},
	}
}

func toolUseDelta(index int32, arguments string) bedrocktypes.ConverseStreamOutput {
	return &bedrocktypes.ConverseStreamOutputMemberContentBlockDelta{
		Value: bedrocktypes.ContentBlockDeltaEvent{
			ContentBlockIndex: aws.Int32(index),
			Delta: &bedrocktypes.ContentBlockDeltaMemberToolUse{
				Value: bedrocktypes.ToolUseBlockDelta{
					Input: aws.String(arguments),
				},
			},
		},
	}
}

func messageStop(reason string) bedrocktypes.ConverseStreamOutput {
	return &bedrocktypes.ConverseStreamOutputMemberMessageStop{
		Value: bedrocktypes.MessageStopEvent{
			StopReason: bedrocktypes.StopReason(reason),
		},
	}
}

func metadataUsage(prompt int32, completion int32) bedrocktypes.ConverseStreamOutput {
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
