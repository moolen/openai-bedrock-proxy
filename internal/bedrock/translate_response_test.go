package bedrock

import "testing"

func TestTranslateConverseResponseBuildsOutputText(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Text:       "hello back",
		StopReason: "end_turn",
	}
	got := TranslateResponse(resp, "model")
	if got.Object != "response" {
		t.Fatalf("expected response object, got %q", got.Object)
	}
	if got.ID != "resp_bedrock-1" {
		t.Fatalf("expected translated response id, got %q", got.ID)
	}
	if got.Model != "model" {
		t.Fatalf("expected model to pass through, got %q", got.Model)
	}
	if len(got.Output) != 1 {
		t.Fatalf("expected one output item, got %d", len(got.Output))
	}
	if got.Output[0].Type != "message" {
		t.Fatalf("expected message output item, got %q", got.Output[0].Type)
	}
	if got.Output[0].Role != "assistant" {
		t.Fatalf("expected assistant role, got %q", got.Output[0].Role)
	}
	if len(got.Output[0].Content) != 1 {
		t.Fatalf("expected one content item, got %d", len(got.Output[0].Content))
	}
	if got.Output[0].Content[0].Type != "output_text" {
		t.Fatalf("expected output_text content type, got %q", got.Output[0].Content[0].Type)
	}
	if got.Output[0].Content[0].Text != "hello back" {
		t.Fatalf("expected text content to pass through, got %q", got.Output[0].Content[0].Text)
	}
}
