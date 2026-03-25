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
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
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

type ChatMessage struct {
	Role       string             `json:"role"`
	Content    ChatMessageContent `json:"content,omitempty"`
	Name       string             `json:"name,omitempty"`
	ToolCallID string             `json:"tool_call_id,omitempty"`
	ToolCalls  []ChatToolCall     `json:"tool_calls,omitempty"`
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
		var functionChoice ChatToolChoiceFunction
		if err := json.Unmarshal(trimmed, &functionChoice); err != nil {
			return err
		}
		*c = ChatToolChoice{
			Kind:          ChatToolChoiceKindFunction,
			FunctionValue: &functionChoice,
		}
		return nil
	}

	*c = ChatToolChoice{Kind: ChatToolChoiceKindInvalid}
	return nil
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
