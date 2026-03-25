package openai

import (
	"bytes"
	"encoding/json"
)

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type ChatToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments,omitempty"`
}

type ChatToolCall struct {
	ID       string               `json:"id,omitempty"`
	Type     string               `json:"type,omitempty"`
	Function ChatToolCallFunction `json:"function,omitempty"`
}

type ChatMessageContentKind string

const (
	ChatMessageContentKindUnset   ChatMessageContentKind = ""
	ChatMessageContentKindText    ChatMessageContentKind = "text"
	ChatMessageContentKindParts   ChatMessageContentKind = "parts"
	ChatMessageContentKindInvalid ChatMessageContentKind = "invalid"
)

type ChatMessageContentPart struct {
	Type     string                     `json:"type"`
	Text     string                     `json:"text,omitempty"`
	ImageURL map[string]any             `json:"image_url,omitempty"`
	raw      map[string]json.RawMessage `json:"-"`
}

func (p *ChatMessageContentPart) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	p.raw = raw

	if rawType, ok := raw["type"]; ok {
		_ = json.Unmarshal(rawType, &p.Type)
	}
	if rawText, ok := raw["text"]; ok {
		_ = json.Unmarshal(rawText, &p.Text)
	}
	if rawImageURL, ok := raw["image_url"]; ok {
		var imageURL map[string]any
		if err := json.Unmarshal(rawImageURL, &imageURL); err == nil {
			p.ImageURL = imageURL
		}
	}
	return nil
}

func (p ChatMessageContentPart) MarshalJSON() ([]byte, error) {
	if p.raw != nil {
		return json.Marshal(p.raw)
	}

	type alias ChatMessageContentPart
	return json.Marshal(alias{
		Type:     p.Type,
		Text:     p.Text,
		ImageURL: p.ImageURL,
	})
}

type ChatMessageContent struct {
	Kind  ChatMessageContentKind   `json:"-"`
	Text  string                   `json:"-"`
	Parts []ChatMessageContentPart `json:"-"`
}

func ChatMessageText(value string) ChatMessageContent {
	return ChatMessageContent{
		Kind: ChatMessageContentKindText,
		Text: value,
	}
}

func (c *ChatMessageContent) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if bytes.Equal(trimmed, []byte("null")) {
		*c = ChatMessageContent{}
		return nil
	}
	if len(trimmed) == 0 {
		*c = ChatMessageContent{Kind: ChatMessageContentKindInvalid}
		return nil
	}

	if trimmed[0] == '"' {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return err
		}
		*c = ChatMessageText(text)
		return nil
	}

	if trimmed[0] == '[' {
		var parts []ChatMessageContentPart
		if err := json.Unmarshal(trimmed, &parts); err != nil {
			return err
		}
		*c = ChatMessageContent{
			Kind:  ChatMessageContentKindParts,
			Parts: parts,
		}
		return nil
	}

	*c = ChatMessageContent{Kind: ChatMessageContentKindInvalid}
	return nil
}

func (c ChatMessageContent) MarshalJSON() ([]byte, error) {
	switch c.Kind {
	case ChatMessageContentKindUnset:
		return []byte("null"), nil
	case ChatMessageContentKindText:
		return json.Marshal(c.Text)
	case ChatMessageContentKindParts:
		return json.Marshal(c.Parts)
	default:
		return []byte("null"), nil
	}
}

func (c ChatMessageContent) IsZero() bool {
	return c.Kind == ChatMessageContentKindUnset
}

type ChatMessage struct {
	Role             string             `json:"role"`
	Content          ChatMessageContent `json:"content,omitempty"`
	Name             string             `json:"name,omitempty"`
	ToolCallID       string             `json:"tool_call_id,omitempty"`
	ToolCalls        []ChatToolCall     `json:"tool_calls,omitempty"`
	ReasoningContent string             `json:"reasoning_content,omitempty"`
}

func (m ChatMessage) MarshalJSON() ([]byte, error) {
	type chatMessageJSON struct {
		Role             string              `json:"role"`
		Content          *ChatMessageContent `json:"content,omitempty"`
		Name             string              `json:"name,omitempty"`
		ToolCallID       string              `json:"tool_call_id,omitempty"`
		ToolCalls        []ChatToolCall      `json:"tool_calls,omitempty"`
		ReasoningContent string              `json:"reasoning_content,omitempty"`
	}

	var content *ChatMessageContent
	if !m.Content.IsZero() {
		content = &m.Content
	}

	return json.Marshal(chatMessageJSON{
		Role:             m.Role,
		Content:          content,
		Name:             m.Name,
		ToolCallID:       m.ToolCallID,
		ToolCalls:        m.ToolCalls,
		ReasoningContent: m.ReasoningContent,
	})
}

type ChatStopKind string

const (
	ChatStopKindUnset   ChatStopKind = ""
	ChatStopKindString  ChatStopKind = "string"
	ChatStopKindStrings ChatStopKind = "strings"
	ChatStopKindInvalid ChatStopKind = "invalid"
)

type ChatStop struct {
	Kind   ChatStopKind `json:"-"`
	Value  string       `json:"-"`
	Values []string     `json:"-"`
}

func (s *ChatStop) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if bytes.Equal(trimmed, []byte("null")) {
		*s = ChatStop{}
		return nil
	}
	if len(trimmed) == 0 {
		*s = ChatStop{Kind: ChatStopKindInvalid}
		return nil
	}

	if trimmed[0] == '"' {
		var value string
		if err := json.Unmarshal(trimmed, &value); err != nil {
			return err
		}
		*s = ChatStop{
			Kind:  ChatStopKindString,
			Value: value,
		}
		return nil
	}

	if trimmed[0] == '[' {
		var values []string
		if err := json.Unmarshal(trimmed, &values); err != nil {
			return err
		}
		*s = ChatStop{
			Kind:   ChatStopKindStrings,
			Values: values,
		}
		return nil
	}

	*s = ChatStop{Kind: ChatStopKindInvalid}
	return nil
}

func (s ChatStop) MarshalJSON() ([]byte, error) {
	switch s.Kind {
	case ChatStopKindUnset:
		return []byte("null"), nil
	case ChatStopKindString:
		return json.Marshal(s.Value)
	case ChatStopKindStrings:
		return json.Marshal(s.Values)
	default:
		return []byte("null"), nil
	}
}

func (s ChatStop) IsZero() bool {
	return s.Kind == ChatStopKindUnset
}

type ChatToolChoiceKind string

const (
	ChatToolChoiceKindUnset    ChatToolChoiceKind = ""
	ChatToolChoiceKindString   ChatToolChoiceKind = "string"
	ChatToolChoiceKindFunction ChatToolChoiceKind = "function"
	ChatToolChoiceKindInvalid  ChatToolChoiceKind = "invalid"
)

type ChatToolChoiceFunction struct {
	Type     string             `json:"type"`
	Function ToolChoiceFunction `json:"function"`
}

type ChatToolChoice struct {
	Kind          ChatToolChoiceKind      `json:"-"`
	StringValue   string                  `json:"-"`
	FunctionValue *ChatToolChoiceFunction `json:"-"`
}

func ChatToolChoiceString(value string) ChatToolChoice {
	return ChatToolChoice{
		Kind:        ChatToolChoiceKindString,
		StringValue: value,
	}
}

func ChatToolChoiceFunctionName(name string) ChatToolChoice {
	return ChatToolChoice{
		Kind: ChatToolChoiceKindFunction,
		FunctionValue: &ChatToolChoiceFunction{
			Type: "function",
			Function: ToolChoiceFunction{
				Name: name,
			},
		},
	}
}

func (c *ChatToolChoice) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if bytes.Equal(trimmed, []byte("null")) {
		*c = ChatToolChoice{}
		return nil
	}
	if len(trimmed) == 0 {
		*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
		return nil
	}

	if trimmed[0] == '"' {
		var value string
		if err := json.Unmarshal(trimmed, &value); err != nil {
			return err
		}
		*c = ChatToolChoiceString(value)
		return nil
	}

	if trimmed[0] == '{' {
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(trimmed, &raw); err != nil {
			return err
		}

		if len(raw) != 2 {
			*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
			return nil
		}

		rawType, ok := raw["type"]
		if !ok {
			*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
			return nil
		}
		rawFunction, ok := raw["function"]
		if !ok {
			*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
			return nil
		}

		var typeValue string
		if err := json.Unmarshal(rawType, &typeValue); err != nil {
			*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
			return nil
		}

		var rawFunctionObject map[string]json.RawMessage
		if err := json.Unmarshal(rawFunction, &rawFunctionObject); err != nil {
			*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
			return nil
		}
		if len(rawFunctionObject) != 1 {
			*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
			return nil
		}
		rawName, ok := rawFunctionObject["name"]
		if !ok {
			*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
			return nil
		}

		var functionName string
		if err := json.Unmarshal(rawName, &functionName); err != nil {
			*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
			return nil
		}

		*c = ChatToolChoice{
			Kind: ChatToolChoiceKindFunction,
			FunctionValue: &ChatToolChoiceFunction{
				Type: typeValue,
				Function: ToolChoiceFunction{
					Name: functionName,
				},
			},
		}
		return nil
	}

	*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
	return nil
}

func (c ChatToolChoice) MarshalJSON() ([]byte, error) {
	switch c.Kind {
	case ChatToolChoiceKindUnset:
		return []byte("null"), nil
	case ChatToolChoiceKindString:
		return json.Marshal(c.StringValue)
	case ChatToolChoiceKindFunction:
		if c.FunctionValue == nil {
			return []byte("null"), nil
		}
		return json.Marshal(c.FunctionValue)
	default:
		return []byte("null"), nil
	}
}

func (c ChatToolChoice) IsZero() bool {
	return c.Kind == ChatToolChoiceKindUnset
}

type ChatCompletionRequest struct {
	Model               string         `json:"model"`
	Messages            []ChatMessage  `json:"messages"`
	Stream              bool           `json:"stream,omitempty"`
	StreamOptions       *StreamOptions `json:"stream_options,omitempty"`
	Temperature         *float64       `json:"temperature,omitempty"`
	TopP                *float64       `json:"top_p,omitempty"`
	MaxTokens           *int           `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int           `json:"max_completion_tokens,omitempty"`
	Stop                ChatStop       `json:"stop,omitempty"`
	Tools               []Tool         `json:"tools,omitempty"`
	ToolChoice          ChatToolChoice `json:"tool_choice,omitempty"`
	ReasoningEffort     string         `json:"reasoning_effort,omitempty"`
	ExtraBody           map[string]any `json:"extra_body,omitempty"`
}

func (r ChatCompletionRequest) MarshalJSON() ([]byte, error) {
	type chatCompletionRequestJSON struct {
		Model               string          `json:"model"`
		Messages            []ChatMessage   `json:"messages"`
		Stream              bool            `json:"stream,omitempty"`
		StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
		Temperature         *float64        `json:"temperature,omitempty"`
		TopP                *float64        `json:"top_p,omitempty"`
		MaxTokens           *int            `json:"max_tokens,omitempty"`
		MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
		Stop                *ChatStop       `json:"stop,omitempty"`
		Tools               []Tool          `json:"tools,omitempty"`
		ToolChoice          *ChatToolChoice `json:"tool_choice,omitempty"`
		ReasoningEffort     string          `json:"reasoning_effort,omitempty"`
		ExtraBody           map[string]any  `json:"extra_body,omitempty"`
	}

	var stop *ChatStop
	if !r.Stop.IsZero() {
		stop = &r.Stop
	}

	var toolChoice *ChatToolChoice
	if !r.ToolChoice.IsZero() {
		toolChoice = &r.ToolChoice
	}

	return json.Marshal(chatCompletionRequestJSON{
		Model:               r.Model,
		Messages:            r.Messages,
		Stream:              r.Stream,
		StreamOptions:       r.StreamOptions,
		Temperature:         r.Temperature,
		TopP:                r.TopP,
		MaxTokens:           r.MaxTokens,
		MaxCompletionTokens: r.MaxCompletionTokens,
		Stop:                stop,
		Tools:               r.Tools,
		ToolChoice:          toolChoice,
		ReasoningEffort:     r.ReasoningEffort,
		ExtraBody:           r.ExtraBody,
	})
}

type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

type ChatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   *ChatCompletionUsage   `json:"usage,omitempty"`
}

type ChatCompletionChunkDelta struct {
	Role      string         `json:"role,omitempty"`
	Content   string         `json:"content,omitempty"`
	ToolCalls []ChatToolCall `json:"tool_calls,omitempty"`
}

type ChatCompletionChunkChoice struct {
	Index        int                      `json:"index"`
	Delta        ChatCompletionChunkDelta `json:"delta"`
	FinishReason *string                  `json:"finish_reason"`
}

type ChatCompletionChunk struct {
	ID      string                      `json:"id"`
	Object  string                      `json:"object"`
	Created int64                       `json:"created"`
	Model   string                      `json:"model"`
	Choices []ChatCompletionChunkChoice `json:"choices"`
	Usage   *ChatCompletionUsage        `json:"usage,omitempty"`
}
