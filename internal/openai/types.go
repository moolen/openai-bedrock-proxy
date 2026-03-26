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
	Type             string                     `json:"type"`
	Name             string                     `json:"name,omitempty"`
	Description      string                     `json:"description,omitempty"`
	Function         *ToolFunction              `json:"function,omitempty"`
	Config           map[string]json.RawMessage `json:"-"`
	hasFunctionField bool                       `json:"-"`
	hasNameField     bool                       `json:"-"`
}

func (t *Tool) UnmarshalJSON(data []byte) error {
	type toolPayload struct {
		Type        string         `json:"type"`
		Name        string         `json:"name,omitempty"`
		Function    *ToolFunction  `json:"function,omitempty"`
		Description string         `json:"description,omitempty"`
		Parameters  map[string]any `json:"parameters,omitempty"`
	}
	var decoded toolPayload
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	_, hasFunctionField := raw["function"]
	_, hasNameField := raw["name"]

	function := decoded.Function
	if decoded.Type == "function" {
		if function == nil {
			function = &ToolFunction{}
		}
		if function.Name == "" {
			function.Name = decoded.Name
		}
		if function.Description == "" {
			function.Description = decoded.Description
		}
		if len(function.Parameters) == 0 && len(decoded.Parameters) > 0 {
			function.Parameters = decoded.Parameters
		}
	}

	delete(raw, "type")
	delete(raw, "name")
	delete(raw, "function")
	delete(raw, "description")
	if decoded.Type == "function" {
		delete(raw, "parameters")
	}
	delete(raw, "strict")
	delete(raw, "defer_loading")

	*t = Tool{
		Type:        decoded.Type,
		Name:        decoded.Name,
		Description: decoded.Description,
		Function:    function,
	}
	t.hasFunctionField = hasFunctionField
	t.hasNameField = hasNameField
	if len(raw) > 0 {
		t.Config = raw
	}
	return nil
}

type ToolChoiceFunction struct {
	Name string `json:"name"`
}

type ToolChoice struct {
	Mode             string              `json:"-"`
	Type             string              `json:"type"`
	Name             string              `json:"name,omitempty"`
	Function         *ToolChoiceFunction `json:"function,omitempty"`
	hasFunctionField bool                `json:"-"`
	hasNameField     bool                `json:"-"`
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

		var raw map[string]json.RawMessage
		if err := json.Unmarshal(trimmed, &raw); err != nil {
			return err
		}
		*tc = ToolChoice(decoded)
		tc.Mode = "object"
		_, tc.hasFunctionField = raw["function"]
		_, tc.hasNameField = raw["name"]
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
	Type          string         `json:"type"`
	ID            string         `json:"id,omitempty"`
	Role          string         `json:"role,omitempty"`
	Content       []ContentItem  `json:"content,omitempty"`
	CallID        string         `json:"call_id,omitempty"`
	Status        string         `json:"status,omitempty"`
	Execution     string         `json:"execution,omitempty"`
	Name          string         `json:"name,omitempty"`
	Input         string         `json:"input,omitempty"`
	Arguments     string         `json:"arguments,omitempty"`
	Action        map[string]any `json:"action,omitempty"`
	Result        string         `json:"result,omitempty"`
	RevisedPrompt string         `json:"revised_prompt,omitempty"`
	Raw           map[string]any `json:"-"`
}

func (o OutputItem) MarshalJSON() ([]byte, error) {
	if len(o.Raw) > 0 {
		raw := make(map[string]any, len(o.Raw)+1)
		for key, value := range o.Raw {
			raw[key] = value
		}
		if _, ok := raw["type"]; !ok && o.Type != "" {
			raw["type"] = o.Type
		}
		return json.Marshal(raw)
	}

	type outputItemAlias OutputItem
	alias := outputItemAlias(o)
	alias.Raw = nil
	return json.Marshal(alias)
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

type CodexReasoningLevel struct {
	Effort      string `json:"effort"`
	Description string `json:"description"`
}

type CodexTruncationPolicy struct {
	Mode  string `json:"mode"`
	Limit int64  `json:"limit"`
}

type CodexModelInfo struct {
	Slug                          string                `json:"slug"`
	DisplayName                   string                `json:"display_name"`
	Description                   string                `json:"description,omitempty"`
	DefaultReasoningLevel         string                `json:"default_reasoning_level,omitempty"`
	SupportedReasoningLevels      []CodexReasoningLevel `json:"supported_reasoning_levels"`
	ShellType                     string                `json:"shell_type"`
	Visibility                    string                `json:"visibility"`
	SupportedInAPI                bool                  `json:"supported_in_api"`
	Priority                      int                   `json:"priority"`
	BaseInstructions              string                `json:"base_instructions"`
	SupportsReasoningSummaries    bool                  `json:"supports_reasoning_summaries"`
	DefaultReasoningSummary       string                `json:"default_reasoning_summary"`
	SupportVerbosity              bool                  `json:"support_verbosity"`
	WebSearchToolType             string                `json:"web_search_tool_type"`
	TruncationPolicy              CodexTruncationPolicy `json:"truncation_policy"`
	SupportsParallelToolCalls     bool                  `json:"supports_parallel_tool_calls"`
	SupportsImageDetailOriginal   bool                  `json:"supports_image_detail_original"`
	ContextWindow                 *int64                `json:"context_window,omitempty"`
	EffectiveContextWindowPercent int                   `json:"effective_context_window_percent"`
	ExperimentalSupportedTools    []string              `json:"experimental_supported_tools"`
	InputModalities               []string              `json:"input_modalities"`
	SupportsSearchTool            bool                  `json:"supports_search_tool"`
}

type ModelsList struct {
	Object string           `json:"object"`
	Data   []Model          `json:"data"`
	Models []CodexModelInfo `json:"models"`
}
