package openai

import (
	"bytes"
	"encoding/json"
)

type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type Tool struct {
	Type     string        `json:"type"`
	Name     string        `json:"name,omitempty"`
	Function *ToolFunction `json:"function,omitempty"`
}

type ToolChoiceFunction struct {
	Name string `json:"name"`
}

type ToolChoice struct {
	Mode     string              `json:"-"`
	Type     string              `json:"type"`
	Name     string              `json:"name,omitempty"`
	Function *ToolChoiceFunction `json:"function,omitempty"`
}

func (tc *ToolChoice) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if bytes.Equal(trimmed, []byte("null")) {
		*tc = ToolChoice{}
		return nil
	}
	if len(trimmed) == 0 {
		*tc = ToolChoice{Mode: "invalid"}
		return nil
	}

	if trimmed[0] == '"' {
		var value string
		if err := json.Unmarshal(trimmed, &value); err != nil {
			return err
		}
		*tc = ToolChoice{
			Mode: "string",
			Type: value,
		}
		return nil
	}

	if trimmed[0] == '{' {
		type toolChoiceAlias ToolChoice
		var decoded toolChoiceAlias
		if err := json.Unmarshal(trimmed, &decoded); err != nil {
			return err
		}
		*tc = ToolChoice(decoded)
		tc.Mode = "object"
		return nil
	}

	*tc = ToolChoice{Mode: "invalid"}
	return nil
}

type ResponsesRequest struct {
	Model              string      `json:"model"`
	Input              any         `json:"input"`
	PreviousResponseID string      `json:"previous_response_id,omitempty"`
	Instructions       string      `json:"instructions,omitempty"`
	Stream             bool        `json:"stream,omitempty"`
	MaxOutputTokens    *int        `json:"max_output_tokens,omitempty"`
	Temperature        *float64    `json:"temperature,omitempty"`
	Tools              []Tool      `json:"tools,omitempty"`
	ToolChoice         *ToolChoice `json:"tool_choice,omitempty"`
	ParallelToolCalls  *bool       `json:"parallel_tool_calls,omitempty"`
}

type ContentItem struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type OutputItem struct {
	Type    string        `json:"type"`
	Role    string        `json:"role,omitempty"`
	Content []ContentItem `json:"content,omitempty"`
}

type Response struct {
	ID     string       `json:"id"`
	Object string       `json:"object"`
	Model  string       `json:"model"`
	Output []OutputItem `json:"output"`
}

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
	Name    string `json:"name,omitempty"`
}

type ModelsList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}
