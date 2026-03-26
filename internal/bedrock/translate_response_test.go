package bedrock

import (
	"encoding/json"
	"testing"
)

func TestTranslateConverseResponseBuildsOutputText(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Output: []OutputBlock{
			{Type: OutputBlockTypeText, Text: "hello back"},
		},
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

func TestTranslateConverseResponseBuildsFunctionCallOutput(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Output: []OutputBlock{
			{
				Type: OutputBlockTypeToolCall,
				ToolCall: &ToolCall{
					ID:        "call_123",
					Name:      "lookup",
					Arguments: `{"q":"hi"}`,
				},
			},
		},
	}

	got := TranslateResponse(resp, "model")
	if len(got.Output) != 1 {
		t.Fatalf("expected one output item, got %d", len(got.Output))
	}
	if got.Output[0].Type != "function_call" {
		t.Fatalf("expected function_call output, got %#v", got.Output[0])
	}
	if got.Output[0].CallID != "call_123" {
		t.Fatalf("expected call id to pass through, got %#v", got.Output[0])
	}
	if got.Output[0].Name != "lookup" {
		t.Fatalf("expected tool name to pass through, got %#v", got.Output[0])
	}
	if got.Output[0].Arguments != `{"q":"hi"}` {
		t.Fatalf("expected arguments JSON to pass through, got %#v", got.Output[0])
	}
}

func TestTranslateConverseResponseMapsSyntheticBuiltInToolCall(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Output: []OutputBlock{
			{
				Type: OutputBlockTypeToolCall,
				ToolCall: &ToolCall{
					ID:        "call_web",
					Name:      "__builtin_web_search_preview",
					Arguments: `{"query":"golang"}`,
				},
			},
		},
	}

	got := TranslateResponse(resp, "model")
	if len(got.Output) != 1 {
		t.Fatalf("expected one output item, got %d", len(got.Output))
	}
	if got.Output[0].Type != "web_search_call" {
		t.Fatalf("expected built-in output type, got %#v", got.Output[0])
	}
	if got.Output[0].CallID != "call_web" {
		t.Fatalf("expected built-in call id, got %#v", got.Output[0])
	}
	if got.Output[0].Action["query"] != "golang" {
		t.Fatalf("expected built-in action payload, got %#v", got.Output[0].Action)
	}
}

func TestTranslateConverseResponseMapsSyntheticCustomToolCall(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Output: []OutputBlock{
			{
				Type: OutputBlockTypeToolCall,
				ToolCall: &ToolCall{
					ID:        "call_patch",
					Name:      "__custom_apply_patch",
					Arguments: `{"input":"*** Begin Patch"}`,
				},
			},
		},
	}

	got := TranslateResponse(resp, "model")
	if len(got.Output) != 1 {
		t.Fatalf("expected one output item, got %d", len(got.Output))
	}
	if got.Output[0].Type != "custom_tool_call" {
		t.Fatalf("expected custom_tool_call output, got %#v", got.Output[0])
	}
	if got.Output[0].CallID != "call_patch" {
		t.Fatalf("expected custom call id, got %#v", got.Output[0])
	}
	if got.Output[0].Name != "apply_patch" {
		t.Fatalf("expected custom tool name to be restored, got %#v", got.Output[0])
	}
	if got.Output[0].Input != "*** Begin Patch" {
		t.Fatalf("expected custom input to be extracted, got %#v", got.Output[0])
	}
}

func TestTranslateConverseResponseMapsToolSearchCall(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Output: []OutputBlock{
			{
				Type: OutputBlockTypeToolCall,
				ToolCall: &ToolCall{
					ID:        "search_1",
					Name:      "__builtin_tool_search",
					Arguments: `{"query":"calendar create","limit":1}`,
				},
			},
		},
	}

	got := TranslateResponse(resp, "model")
	if len(got.Output) != 1 {
		t.Fatalf("expected one output item, got %d", len(got.Output))
	}
	if got.Output[0].Type != "tool_search_call" {
		t.Fatalf("expected tool_search_call output, got %#v", got.Output[0])
	}
	encoded, err := json.Marshal(got.Output[0])
	if err != nil {
		t.Fatalf("expected tool_search output item to marshal, got %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("expected marshaled tool_search output to decode, got %v", err)
	}
	if decoded["execution"] != "client" {
		t.Fatalf("expected tool_search execution metadata, got %#v", decoded)
	}
	arguments, ok := decoded["arguments"].(map[string]any)
	if !ok || arguments["query"] != "calendar create" {
		t.Fatalf("expected tool_search arguments payload, got %#v", decoded)
	}
}

func TestTranslateConverseResponseMapsLocalShellCall(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Output: []OutputBlock{
			{
				Type: OutputBlockTypeToolCall,
				ToolCall: &ToolCall{
					ID:        "shell_1",
					Name:      "__builtin_local_shell",
					Arguments: `{"command":["pwd"],"working_directory":"/tmp"}`,
				},
			},
		},
	}

	got := TranslateResponse(resp, "model")
	if len(got.Output) != 1 {
		t.Fatalf("expected one output item, got %d", len(got.Output))
	}
	if got.Output[0].Type != "local_shell_call" {
		t.Fatalf("expected local_shell_call output, got %#v", got.Output[0])
	}
	encoded, err := json.Marshal(got.Output[0])
	if err != nil {
		t.Fatalf("expected local_shell output item to marshal, got %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("expected marshaled local_shell output to decode, got %v", err)
	}
	action, ok := decoded["action"].(map[string]any)
	if !ok || action["type"] != "exec" {
		t.Fatalf("expected local_shell action payload, got %#v", decoded)
	}
}

func TestTranslateConverseResponseMapsImageGenerationCall(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Output: []OutputBlock{
			{
				Type: OutputBlockTypeToolCall,
				ToolCall: &ToolCall{
					ID:        "img_1",
					Name:      "__builtin_image_generation",
					Arguments: `{"status":"completed","revised_prompt":"A blue square","result":"Zm9v"}`,
				},
			},
		},
	}

	got := TranslateResponse(resp, "model")
	if len(got.Output) != 1 {
		t.Fatalf("expected one output item, got %d", len(got.Output))
	}
	if got.Output[0].Type != "image_generation_call" {
		t.Fatalf("expected image_generation_call output, got %#v", got.Output[0])
	}
	encoded, err := json.Marshal(got.Output[0])
	if err != nil {
		t.Fatalf("expected image_generation output item to marshal, got %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("expected marshaled image generation output to decode, got %v", err)
	}
	if decoded["status"] != "completed" || decoded["result"] != "Zm9v" || decoded["revised_prompt"] != "A blue square" {
		t.Fatalf("expected image generation payload to be preserved, got %#v", decoded)
	}
}

func TestTranslateConverseResponsePreservesMixedTextAndToolOrder(t *testing.T) {
	resp := ConverseResponse{
		ResponseID: "bedrock-1",
		Output: []OutputBlock{
			{Type: OutputBlockTypeText, Text: "Checking"},
			{
				Type: OutputBlockTypeToolCall,
				ToolCall: &ToolCall{
					ID:        "call_123",
					Name:      "lookup",
					Arguments: `{"q":"weather"}`,
				},
			},
			{Type: OutputBlockTypeText, Text: "Waiting"},
		},
	}

	got := TranslateResponse(resp, "model")
	if len(got.Output) != 3 {
		t.Fatalf("expected 3 ordered output items, got %#v", got.Output)
	}
	if got.Output[0].Type != "message" || got.Output[0].Content[0].Text != "Checking" {
		t.Fatalf("expected leading text message, got %#v", got.Output[0])
	}
	if got.Output[1].Type != "function_call" || got.Output[1].CallID != "call_123" {
		t.Fatalf("expected middle tool call, got %#v", got.Output[1])
	}
	if got.Output[2].Type != "message" || got.Output[2].Content[0].Text != "Waiting" {
		t.Fatalf("expected trailing text message, got %#v", got.Output[2])
	}
}

func TestTextAccumulatorJoinsAllDeltas(t *testing.T) {
	accumulator := TextAccumulator{}
	accumulator.Add("hello")
	accumulator.Add(" ")
	accumulator.Add("world")

	if got := accumulator.Text(); got != "hello world" {
		t.Fatalf("expected joined text, got %q", got)
	}
}
