package bedrock

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type chatToolUseState struct {
	index int
	id    string
	name  string
}

func WriteChatCompletionsStream(stream streamEvents, responseID string, model string, includeUsage bool, w io.Writer) error {
	if stream == nil {
		return errors.New("bedrock chat stream was nil")
	}
	if w == nil {
		return errors.New("chat stream writer was nil")
	}
	defer stream.Close()

	responseID = normalizeChatCompletionID(responseID)
	created := time.Now().Unix()
	roleEmitted := false
	toolStates := map[int32]chatToolUseState{}
	nextToolIndex := 0
	var usage *openai.ChatCompletionUsage

	emitRoleChunk := func() error {
		if roleEmitted {
			return nil
		}
		roleEmitted = true
		return writeChatChunk(w, openai.ChatCompletionChunk{
			ID:      responseID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []openai.ChatCompletionChunkChoice{{
				Index: 0,
				Delta: openai.ChatCompletionChunkDelta{
					Role: "assistant",
				},
				FinishReason: nil,
			}},
		})
	}

	for event := range stream.Events() {
		switch typed := event.(type) {
		case *bedrocktypes.ConverseStreamOutputMemberContentBlockStart:
			start, ok := typed.Value.Start.(*bedrocktypes.ContentBlockStartMemberToolUse)
			if !ok {
				continue
			}
			if typed.Value.ContentBlockIndex == nil {
				continue
			}

			if err := emitRoleChunk(); err != nil {
				return err
			}

			blockIndex := aws.ToInt32(typed.Value.ContentBlockIndex)
			state, exists := toolStates[blockIndex]
			if !exists {
				state.index = nextToolIndex
				nextToolIndex++
			}
			state.id = aws.ToString(start.Value.ToolUseId)
			state.name = aws.ToString(start.Value.Name)
			toolStates[blockIndex] = state

			index := state.index
			if err := writeChatChunk(w, openai.ChatCompletionChunk{
				ID:      responseID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []openai.ChatCompletionChunkChoice{{
					Index: 0,
					Delta: openai.ChatCompletionChunkDelta{
						ToolCalls: []openai.ChatToolCall{{
							Index: &index,
							ID:    state.id,
							Type:  "function",
							Function: openai.ChatToolCallFunction{
								Name: state.name,
							},
						}},
					},
					FinishReason: nil,
				}},
			}); err != nil {
				return err
			}
		case *bedrocktypes.ConverseStreamOutputMemberContentBlockDelta:
			if err := emitRoleChunk(); err != nil {
				return err
			}

			switch delta := typed.Value.Delta.(type) {
			case *bedrocktypes.ContentBlockDeltaMemberText:
				if err := writeChatChunk(w, openai.ChatCompletionChunk{
					ID:      responseID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []openai.ChatCompletionChunkChoice{{
						Index: 0,
						Delta: openai.ChatCompletionChunkDelta{
							Content: delta.Value,
						},
						FinishReason: nil,
					}},
				}); err != nil {
					return err
				}
			case *bedrocktypes.ContentBlockDeltaMemberReasoningContent:
				reasoning, ok := reasoningDeltaText(delta.Value)
				if !ok || reasoning == "" {
					continue
				}
				if err := writeChatChunk(w, openai.ChatCompletionChunk{
					ID:      responseID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []openai.ChatCompletionChunkChoice{{
						Index: 0,
						Delta: openai.ChatCompletionChunkDelta{
							ReasoningContent: reasoning,
						},
						FinishReason: nil,
					}},
				}); err != nil {
					return err
				}
			case *bedrocktypes.ContentBlockDeltaMemberToolUse:
				if typed.Value.ContentBlockIndex == nil {
					continue
				}
				blockIndex := aws.ToInt32(typed.Value.ContentBlockIndex)
				state, exists := toolStates[blockIndex]
				if !exists {
					state = chatToolUseState{index: nextToolIndex}
					nextToolIndex++
					toolStates[blockIndex] = state
				}
				arguments := aws.ToString(delta.Value.Input)
				index := state.index
				if err := writeChatChunk(w, openai.ChatCompletionChunk{
					ID:      responseID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []openai.ChatCompletionChunkChoice{{
						Index: 0,
						Delta: openai.ChatCompletionChunkDelta{
							ToolCalls: []openai.ChatToolCall{{
								Index: &index,
								Function: openai.ChatToolCallFunction{
									Arguments: arguments,
								},
							}},
						},
						FinishReason: nil,
					}},
				}); err != nil {
					return err
				}
			}
		case *bedrocktypes.ConverseStreamOutputMemberMessageStop:
			if err := emitRoleChunk(); err != nil {
				return err
			}
			finishReason := mapChatFinishReason(string(typed.Value.StopReason))
			if err := writeChatChunk(w, openai.ChatCompletionChunk{
				ID:      responseID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []openai.ChatCompletionChunkChoice{{
					Index:        0,
					Delta:        openai.ChatCompletionChunkDelta{},
					FinishReason: &finishReason,
				}},
			}); err != nil {
				return err
			}
		case *bedrocktypes.ConverseStreamOutputMemberMetadata:
			if !includeUsage {
				continue
			}
			usage = chatUsageFromMetadata(typed.Value.Usage)
		}
	}

	if err := stream.Err(); err != nil {
		return err
	}

	if includeUsage && usage != nil {
		if err := writeChatChunk(w, openai.ChatCompletionChunk{
			ID:      responseID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []openai.ChatCompletionChunkChoice{},
			Usage:   usage,
		}); err != nil {
			return err
		}
	}

	return writeChatDone(w)
}

func reasoningDeltaText(delta bedrocktypes.ReasoningContentBlockDelta) (string, bool) {
	switch typed := delta.(type) {
	case *bedrocktypes.ReasoningContentBlockDeltaMemberText:
		return typed.Value, true
	default:
		return "", false
	}
}

func chatUsageFromMetadata(usage *bedrocktypes.TokenUsage) *openai.ChatCompletionUsage {
	if usage == nil {
		return nil
	}
	prompt := int(aws.ToInt32(usage.InputTokens))
	completion := int(aws.ToInt32(usage.OutputTokens))
	total := int(aws.ToInt32(usage.TotalTokens))
	if total == 0 {
		total = prompt + completion
	}
	return &openai.ChatCompletionUsage{
		PromptTokens:     prompt,
		CompletionTokens: completion,
		TotalTokens:      total,
	}
}

func normalizeChatCompletionID(id string) string {
	normalized := strings.TrimSpace(id)
	if normalized == "" {
		return "chatcmpl_" + fallbackResponseID()
	}
	if strings.HasPrefix(normalized, "chatcmpl_") {
		return normalized
	}
	return "chatcmpl_" + normalized
}

func writeChatChunk(w io.Writer, payload openai.ChatCompletionChunk) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}
	flushChatWriter(w)
	return nil
}

func writeChatDone(w io.Writer) error {
	if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	flushChatWriter(w)
	return nil
}

func flushChatWriter(w io.Writer) {
	flusher, ok := w.(interface{ Flush() })
	if !ok {
		return
	}
	flusher.Flush()
}
