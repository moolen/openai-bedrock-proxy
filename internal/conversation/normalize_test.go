package conversation

import (
	"errors"
	"testing"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

func TestNormalizeRequestBuildsSystemAndMessages(t *testing.T) {
	req := openai.ResponsesRequest{
		Model:        "model",
		Instructions: "follow",
		Input: []map[string]any{
			{"role": "system", "content": "sys"},
			{"role": "developer", "content": []map[string]any{{"type": "input_text", "text": "dev"}}},
			{"role": "user", "content": "hi"},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.System) != 3 {
		t.Fatalf("expected 3 system entries, got %d", len(normalized.System))
	}
	if normalized.System[0] != "sys" || normalized.System[1] != "dev" || normalized.System[2] != "follow" {
		t.Fatalf("unexpected system ordering: %#v", normalized.System)
	}

	if len(normalized.Messages) != 1 {
		t.Fatalf("expected 1 persisted message, got %d", len(normalized.Messages))
	}
	if normalized.Messages[0].Role != "user" || normalized.Messages[0].Text != "hi" {
		t.Fatalf("unexpected persisted message: %#v", normalized.Messages[0])
	}
}

func TestNormalizeRequestPersistsAssistantTurnsButNotSystemTurns(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{"role": "system", "content": "sys"},
			{"role": "assistant", "content": "assistant reply"},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.System) != 1 || normalized.System[0] != "sys" {
		t.Fatalf("unexpected system content: %#v", normalized.System)
	}
	if len(normalized.Messages) != 1 {
		t.Fatalf("expected 1 persisted message, got %d", len(normalized.Messages))
	}
	if normalized.Messages[0].Role != "assistant" || normalized.Messages[0].Text != "assistant reply" {
		t.Fatalf("unexpected persisted message: %#v", normalized.Messages[0])
	}
}

func TestNormalizeRequestConcatenatesTextBlocksWithoutSeparators(t *testing.T) {
	req := openai.ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{"role": "user", "content": []map[string]any{
				{"type": "input_text", "text": "hello"},
				{"type": "input_text", "text": "world"},
			}},
		},
	}

	normalized, err := NormalizeRequest(req)
	if err != nil {
		t.Fatalf("NormalizeRequest returned error: %v", err)
	}

	if len(normalized.Messages) != 1 {
		t.Fatalf("expected 1 persisted message, got %d", len(normalized.Messages))
	}
	if normalized.Messages[0].Text != "helloworld" {
		t.Fatalf("expected concatenated text, got %q", normalized.Messages[0].Text)
	}
}

func TestMergeUsesCurrentSystemAndAppendsMessages(t *testing.T) {
	base := Request{
		System: []string{"old"},
		Messages: []Message{
			{Role: "user", Text: "previous"},
		},
	}
	current := Request{
		System: []string{"current"},
		Messages: []Message{
			{Role: "assistant", Text: "now"},
		},
	}

	merged := Merge(base, current)

	if len(merged.System) != 1 || merged.System[0] != "current" {
		t.Fatalf("unexpected merged system: %#v", merged.System)
	}
	if len(merged.Messages) != 2 {
		t.Fatalf("expected 2 merged messages, got %d", len(merged.Messages))
	}
	if merged.Messages[0].Text != "previous" || merged.Messages[1].Text != "now" {
		t.Fatalf("unexpected merged messages: %#v", merged.Messages)
	}
}

func TestAppendAssistantReplyAppendsToMessages(t *testing.T) {
	req := Request{
		System: []string{"sys"},
		Messages: []Message{
			{Role: "user", Text: "hi"},
		},
	}

	updated := AppendAssistantReply(req, "ok")

	if len(updated.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(updated.Messages))
	}
	if updated.Messages[1].Role != "assistant" || updated.Messages[1].Text != "ok" {
		t.Fatalf("unexpected assistant append: %#v", updated.Messages[1])
	}
	if len(updated.System) != 1 || updated.System[0] != "sys" {
		t.Fatalf("unexpected system after append: %#v", updated.System)
	}
}

func TestNormalizeRequestRejectsMalformedInputs(t *testing.T) {
	cases := []struct {
		name  string
		input any
	}{
		{
			name:  "missing role",
			input: map[string]any{"content": "hi"},
		},
		{
			name:  "wrong role type",
			input: map[string]any{"role": 7, "content": "hi"},
		},
		{
			name:  "wrong content type",
			input: map[string]any{"role": "user", "content": 7},
		},
		{
			name:  "non-object item",
			input: []any{"oops"},
		},
		{
			name: "wrong block type",
			input: map[string]any{"role": "user", "content": []map[string]any{
				{"type": "output_text", "text": "hi"},
			}},
		},
		{
			name: "missing text",
			input: map[string]any{"role": "user", "content": []map[string]any{
				{"type": "input_text"},
			}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := openai.ResponsesRequest{
				Model: "model",
				Input: tc.input,
			}

			_, err := NormalizeRequest(req)
			if err == nil {
				t.Fatalf("expected error for malformed input")
			}
			var invalid openai.InvalidRequestError
			if !errors.As(err, &invalid) {
				t.Fatalf("expected invalid request error, got %T", err)
			}
		})
	}
}

func TestNormalizeRequestRejectsStructuredInputWithoutPersistedMessages(t *testing.T) {
	cases := []struct {
		name  string
		input any
	}{
		{
			name:  "system only",
			input: []map[string]any{{"role": "system", "content": "sys"}},
		},
		{
			name:  "developer only",
			input: []map[string]any{{"role": "developer", "content": "dev"}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := openai.ResponsesRequest{
				Model: "model",
				Input: tc.input,
			}

			_, err := NormalizeRequest(req)
			if err == nil {
				t.Fatalf("expected error for structured input without persisted messages")
			}

			var invalid openai.InvalidRequestError
			if !errors.As(err, &invalid) {
				t.Fatalf("expected invalid request error, got %T", err)
			}
		})
	}
}
