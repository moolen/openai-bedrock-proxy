package openai

type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type Tool struct {
	Type     string       `json:"type"`
	Name     string       `json:"name,omitempty"`
	Function ToolFunction `json:"function"`
}

type ResponsesRequest struct {
	Model             string    `json:"model"`
	Input             any       `json:"input"`
	Instructions      string    `json:"instructions,omitempty"`
	Stream            bool      `json:"stream,omitempty"`
	MaxOutputTokens   *int      `json:"max_output_tokens,omitempty"`
	Temperature       *float64  `json:"temperature,omitempty"`
	Tools             []Tool    `json:"tools,omitempty"`
	ToolChoice        any       `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool     `json:"parallel_tool_calls,omitempty"`
}
