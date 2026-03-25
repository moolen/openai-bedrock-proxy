package conversation

import (
	"encoding/json"
	"time"
)

const (
	BlockTypeText       = "text"
	BlockTypeToolCall   = "tool_call"
	BlockTypeToolResult = "tool_result"
)

type Block struct {
	Type       string
	Text       string
	ToolCall   *ToolCall
	ToolResult *ToolResult
}

type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

type ToolResult struct {
	CallID string
	Output any
}

type ToolDefinition struct {
	Type        string
	Name        string
	Description string
	Parameters  map[string]any
	Config      map[string]json.RawMessage
	BuiltIn     bool
}

type ToolChoice struct {
	Type string
	Name string
}

type Message struct {
	Role string
	// Text is a legacy mirror of concatenated text blocks kept for package-local compatibility.
	Text   string
	Blocks []Block
}

type Request struct {
	System     []string
	Messages   []Message
	Tools      []ToolDefinition
	ToolChoice ToolChoice
}

type Record struct {
	ResponseID string
	ModelID    string
	Messages   []Message
	CreatedAt  time.Time
}

type Store interface {
	Get(string) (Record, bool)
	Save(Record)
}
