package openai

import (
	"encoding/json"
	"testing"
)

const invalidResponsesInputMessage = "input must be a non-empty string or supported message object/array"

func TestValidateResponsesRequestAcceptsSimpleTextInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "anthropic.claude-3-7-sonnet-20250219-v1:0",
		Input: "write a haiku",
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected valid request, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsEasyMessageInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role":    "user",
			"content": "hello",
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected easy message input to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsExplicitMessageInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"type": "message",
			"role": "user",
			"content": []map[string]any{
				{"type": "input_text", "text": "hello"},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected explicit message input to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsPlainStringRegression(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hello",
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected plain string input to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsArrayOfMessages(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: []map[string]any{
			{
				"role":    "user",
				"content": "hello",
			},
			{
				"type": "message",
				"role": "developer",
				"content": []map[string]any{
					{"type": "input_text", "text": "be brief"},
				},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected array of messages input to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsFunctionCallOutputsInUserMessages(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "user",
			"content": []map[string]any{
				{"type": "function_call_output", "call_id": "call_1", "output": "ok"},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected function_call_output blocks to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsCustomToolCallOutputsInUserMessages(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "user",
			"content": []map[string]any{
				{"type": "custom_tool_call_output", "call_id": "call_1", "output": "ok"},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected custom_tool_call_output blocks to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsAssistantEasyMessage(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": "hi"},
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected assistant easy message input to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestRejectsUnsupportedContentBlock(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "user",
			"content": []map[string]any{
				{"type": "input_image", "image_url": "https://example.com/cat.png"},
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsNonMessageItem(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: []any{
			map[string]any{
				"role":    "user",
				"content": "hello",
			},
			"not-a-message",
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsAssistantInputTextBlocks(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "assistant",
			"content": []map[string]any{
				{"type": "input_text", "text": "hi"},
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsUserOutputTextBlocks(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "user",
			"content": []map[string]any{
				{"type": "output_text", "text": "hi"},
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsEmptyEasyMessageContent(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role":    "user",
			"content": "",
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsEmptyTextBlockContent(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: map[string]any{
			"role": "assistant",
			"content": []map[string]any{
				{"type": "output_text", "text": ""},
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), invalidResponsesInputMessage)
}

func TestValidateResponsesRequestRejectsMissingModel(t *testing.T) {
	req := ResponsesRequest{Input: "hi"}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected missing-model validation error")
	}
}

func TestValidateResponsesRequestRejectsMissingInput(t *testing.T) {
	req := ResponsesRequest{Model: "model"}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected missing-input validation error")
	}
}

func TestValidateResponsesRequestRejectsEmptyStringInput(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "",
	}
	if err := ValidateResponsesRequest(req); err == nil {
		t.Fatal("expected empty-input validation error")
	}
}

func TestValidateResponsesRequestRejectsUnsupportedFields(t *testing.T) {
	req := ResponsesRequest{
		Model:             "model",
		Input:             "hi",
		ParallelToolCalls: ptr(true),
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected parallel_tool_calls to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsFunctionTools(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{{Type: "function", Function: &ToolFunction{Name: "lookup"}}},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected tools to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsResponsesAPIFunctionTools(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{{
			Type: "function",
			Name: "lookup",
		}},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected Responses API tool shape to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsCodexCustomTools(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"input":"hi",
		"tools":[
			{
				"type":"custom",
				"name":"apply_patch",
				"description":"Apply a patch",
				"format":{"type":"grammar","syntax":"lark","definition":"start: /[\\s\\S]+/"}
			}
		]
	}`)

	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected request to unmarshal, got %v", err)
	}
	if len(req.Tools) != 1 {
		t.Fatalf("expected one tool, got %d", len(req.Tools))
	}
	if req.Tools[0].Type != "custom" || req.Tools[0].Name != "apply_patch" {
		t.Fatalf("expected custom tool metadata to be preserved, got %#v", req.Tools[0])
	}
	if req.Tools[0].Config == nil {
		t.Fatalf("expected custom tool config to be preserved, got %#v", req.Tools[0])
	}
	if _, ok := req.Tools[0].Config["format"]; !ok {
		t.Fatalf("expected custom tool format to be preserved, got %#v", req.Tools[0].Config)
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected custom tool to validate, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsWebSearchTools(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"input":"hi",
		"tools":[
			{
				"type":"web_search",
				"external_web_access":true,
				"user_location":{"type":"approximate","country":"DE"}
			}
		],
		"tool_choice":{"type":"web_search"}
	}`)

	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected request to unmarshal, got %v", err)
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected web_search tool to validate, got %v", err)
	}
}

func TestValidateResponsesRequestRejectsDuplicateFunctionToolNames(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{Type: "function", Function: &ToolFunction{Name: "lookup"}},
			{Type: "function", Function: &ToolFunction{Name: "lookup"}},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tools[1].function.name duplicates a previous tool")
}

func TestValidateResponsesRequestRejectsFunctionToolWithMismatchedTopLevelName(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{
				Type: "function",
				Name: "top_level_name",
				Function: &ToolFunction{
					Name: "function_name",
				},
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tools[0].name must match tools[0].function.name")
}

func TestValidateResponsesRequestAcceptsBuiltInTools(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{Type: "web_search_preview"},
			{Type: "file_search"},
			{Type: "computer_use_preview"},
			{Type: "code_interpreter"},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected built-in tools to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsToolChoiceAuto(t *testing.T) {
	req := ResponsesRequest{
		Model:      "model",
		Input:      "hi",
		ToolChoice: &ToolChoice{Type: "auto", Mode: "string"},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected tool_choice auto to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsToolChoiceAutoWithoutMode(t *testing.T) {
	req := ResponsesRequest{
		Model:      "model",
		Input:      "hi",
		ToolChoice: &ToolChoice{Type: "auto"},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected programmatic tool_choice auto to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestRejectsToolChoiceAutoWithName(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		ToolChoice: &ToolChoice{
			Type: "auto",
			Name: "lookup",
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice.name is not allowed for this tool_choice type")
}

func TestValidateResponsesRequestRejectsToolChoiceAutoWithFunction(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		ToolChoice: &ToolChoice{
			Type:     "auto",
			Function: &ToolChoiceFunction{Name: "lookup"},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice.function is not allowed for this tool_choice type")
}

func TestValidateResponsesRequestAcceptsNamedFunctionToolChoice(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{Type: "function", Function: &ToolFunction{Name: "lookup"}},
		},
		ToolChoice: &ToolChoice{
			Mode: "object",
			Type: "function",
			Function: &ToolChoiceFunction{
				Name: "lookup",
			},
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected named function tool_choice to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestAcceptsNamedFunctionToolChoiceViaTopLevelName(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{Type: "function", Function: &ToolFunction{Name: "lookup"}},
		},
		ToolChoice: &ToolChoice{
			Type: "function",
			Name: "lookup",
		},
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected top-level name function tool_choice to be accepted, got %v", err)
	}
}

func TestValidateResponsesRequestRejectsNamedFunctionToolChoiceWithMissingTool(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{Type: "function", Function: &ToolFunction{Name: "lookup"}},
		},
		ToolChoice: &ToolChoice{
			Mode: "object",
			Type: "function",
			Function: &ToolChoiceFunction{
				Name: "different_tool",
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice.function.name is not present in tools")
}

func TestValidateResponsesRequestRejectsMalformedFunctionTool(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{{Type: "function"}},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tools[0].function.name is required")
}

func TestResponsesRequestUnmarshalAcceptsResponsesAPIToolShape(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"input":"hi",
		"tools":[
			{
				"type":"function",
				"name":"lookup",
				"description":"A demo tool",
				"strict":false,
				"parameters":{"type":"object","properties":{"q":{"type":"string"}}}
			}
		]
	}`)

	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected request to unmarshal, got %v", err)
	}
	if len(req.Tools) != 1 {
		t.Fatalf("expected one tool, got %d", len(req.Tools))
	}
	if req.Tools[0].Function == nil {
		t.Fatalf("expected top-level Responses API tool shape to populate function metadata, got %#v", req.Tools[0])
	}
	if req.Tools[0].Function.Name != "lookup" {
		t.Fatalf("expected tool name to be populated from top-level field, got %#v", req.Tools[0].Function)
	}
	if req.Tools[0].Function.Description != "A demo tool" {
		t.Fatalf("expected tool description to be populated from top-level field, got %#v", req.Tools[0].Function)
	}
	if req.Tools[0].Function.Parameters["type"] != "object" {
		t.Fatalf("expected tool parameters to be populated from top-level field, got %#v", req.Tools[0].Function.Parameters)
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected top-level Responses API tool shape to validate, got %v", err)
	}
}

func TestValidateResponsesRequestRejectsMalformedToolChoice(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		ToolChoice: &ToolChoice{
			Mode:     "object",
			Type:     "function",
			Function: &ToolChoiceFunction{},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice.function.name is required")
}

func TestResponsesRequestUnmarshalRejectsFunctionToolChoiceWithNameAndNullFunction(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"input":"hi",
		"tools":[{"type":"function","function":{"name":"lookup"}}],
		"tool_choice":{"type":"function","name":"lookup","function":null}
	}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice.function.name is required")
}

func TestValidateResponsesRequestRejectsUnmappableToolChoice(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		ToolChoice: &ToolChoice{
			Mode: "invalid",
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice is invalid")
}

func TestResponsesRequestUnmarshalAcceptsToolChoiceAutoString(t *testing.T) {
	raw := []byte(`{"model":"model","input":"hi","tool_choice":"auto"}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	if req.ToolChoice == nil {
		t.Fatal("expected tool_choice to be present")
	}
	if req.ToolChoice.Mode != "string" || req.ToolChoice.Type != "auto" {
		t.Fatalf("expected string auto tool_choice, got mode=%q type=%q", req.ToolChoice.Mode, req.ToolChoice.Type)
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected decoded tool_choice auto to validate, got %v", err)
	}
}

func TestResponsesRequestUnmarshalAcceptsNullToolChoice(t *testing.T) {
	raw := []byte(`{"model":"model","input":"hi","tool_choice":null}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected null tool_choice to validate, got %v", err)
	}
}

func TestResponsesRequestUnmarshalAcceptsNamedFunctionToolChoiceObject(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"input":"hi",
		"tools":[{"type":"function","function":{"name":"lookup"}}],
		"tool_choice":{"type":"function","function":{"name":"lookup"}}
	}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	if req.ToolChoice == nil {
		t.Fatal("expected tool_choice to be present")
	}
	if req.ToolChoice.Mode != "object" || req.ToolChoice.Type != "function" || req.ToolChoice.Function == nil || req.ToolChoice.Function.Name != "lookup" {
		t.Fatalf("unexpected tool_choice decode: %+v", req.ToolChoice)
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected decoded named function tool_choice to validate, got %v", err)
	}
}

func TestResponsesRequestUnmarshalAcceptsFunctionToolChoiceObjectViaTopLevelName(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"input":"hi",
		"tools":[{"type":"function","function":{"name":"lookup"}}],
		"tool_choice":{"type":"function","name":"lookup"}
	}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	if req.ToolChoice == nil {
		t.Fatal("expected tool_choice to be present")
	}
	if req.ToolChoice.Type != "function" || req.ToolChoice.Name != "lookup" {
		t.Fatalf("unexpected tool_choice decode: %+v", req.ToolChoice)
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected decoded named function tool_choice to validate, got %v", err)
	}
}

func TestResponsesRequestUnmarshalAcceptsBuiltInToolChoiceObject(t *testing.T) {
	raw := []byte(`{"model":"model","input":"hi","tool_choice":{"type":"web_search_preview"}}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	if req.ToolChoice == nil {
		t.Fatal("expected tool_choice to be present")
	}
	if req.ToolChoice.Mode != "object" || req.ToolChoice.Type != "web_search_preview" {
		t.Fatalf("expected built-in object tool_choice, got mode=%q type=%q", req.ToolChoice.Mode, req.ToolChoice.Type)
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected decoded built-in tool_choice to validate, got %v", err)
	}
}

func TestResponsesRequestUnmarshalRejectsAutoToolChoiceObjectWithFunction(t *testing.T) {
	raw := []byte(`{"model":"model","input":"hi","tool_choice":{"type":"auto","function":{"name":"x"}}}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice.function is not allowed for this tool_choice type")
}

func TestResponsesRequestUnmarshalRejectsToolChoiceAutoStringWithCompanionFields(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		ToolChoice: &ToolChoice{
			Mode:     "string",
			Type:     "auto",
			Name:     "lookup",
			Function: &ToolChoiceFunction{Name: "lookup"},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice.function is not allowed for this tool_choice type")
}

func TestResponsesRequestUnmarshalRejectsBuiltInToolChoiceObjectWithFunction(t *testing.T) {
	raw := []byte(`{"model":"model","input":"hi","tool_choice":{"type":"web_search_preview","function":{"name":"x"}}}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice.function is not allowed for this tool_choice type")
}

func TestResponsesRequestUnmarshalRejectsBuiltInToolChoiceObjectWithName(t *testing.T) {
	raw := []byte(`{"model":"model","input":"hi","tool_choice":{"type":"web_search_preview","name":"x"}}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tool_choice.name is not allowed for this tool_choice type")
}

func TestResponsesRequestUnmarshalPreservesBuiltInToolConfig(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"input":"hi",
		"tools":[
			{
				"type":"web_search_preview",
				"user_location":{"type":"approximate","country":"DE"}
			}
		]
	}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	if len(req.Tools) != 1 {
		t.Fatalf("expected one tool, got %d", len(req.Tools))
	}
	if req.Tools[0].Config == nil {
		t.Fatal("expected built-in tool config to be preserved")
	}
	if _, ok := req.Tools[0].Config["user_location"]; !ok {
		t.Fatalf("expected user_location config key, got %#v", req.Tools[0].Config)
	}
	if err := ValidateResponsesRequest(req); err != nil {
		t.Fatalf("expected decoded built-in tool with config to validate, got %v", err)
	}
}

func TestResponsesRequestUnmarshalRejectsBuiltInToolWithNullFunctionField(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"input":"hi",
		"tools":[
			{
				"type":"web_search_preview",
				"function":null
			}
		]
	}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tools[0].function is only allowed for function tools")
}

func TestValidateResponsesRequestRejectsUnsupportedBuiltInTool(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{Type: "image_generation"},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tools[0].type is not supported")
}

func TestValidateResponsesRequestRejectsBuiltInToolWithFunction(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{
				Type:     "web_search_preview",
				Function: &ToolFunction{Name: "lookup"},
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tools[0].function is only allowed for function tools")
}

func TestValidateResponsesRequestRejectsBuiltInToolWithName(t *testing.T) {
	req := ResponsesRequest{
		Model: "model",
		Input: "hi",
		Tools: []Tool{
			{
				Type: "web_search_preview",
				Name: "named",
			},
		},
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tools[0].name is only allowed for function tools")
}

func TestResponsesRequestUnmarshalRejectsBuiltInToolWithNameField(t *testing.T) {
	raw := []byte(`{
		"model":"model",
		"input":"hi",
		"tools":[
			{
				"type":"web_search_preview",
				"name":"named"
			}
		]
	}`)
	var req ResponsesRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("expected json unmarshal to succeed, got %v", err)
	}
	assertInvalidRequestMessage(t, ValidateResponsesRequest(req), "tools[0].name is only allowed for function tools")
}

func TestErrorResponseFromClassifiesWrappedInvalidRequestErrors(t *testing.T) {
	err := NewInvalidRequestError("bad request")
	resp := ErrorResponseFrom(wrapError{err: err})
	if resp.Error.Type != "invalid_request_error" {
		t.Fatalf("expected invalid request error type, got %q", resp.Error.Type)
	}
	if resp.Error.Message != "bad request" {
		t.Fatalf("expected wrapped error message, got %q", resp.Error.Message)
	}
}

func assertInvalidRequestMessage(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatal("expected validation error")
	}
	if err.Error() != want {
		t.Fatalf("expected error %q, got %q", want, err.Error())
	}
}

func ptr[T any](value T) *T {
	return &value
}

type wrapError struct {
	err error
}

func (e wrapError) Error() string {
	return "wrapped: " + e.err.Error()
}

func (e wrapError) Unwrap() error {
	return e.err
}
