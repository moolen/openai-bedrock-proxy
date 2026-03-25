package bedrock

import (
	"github.com/aws/aws-sdk-go-v2/aws"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

func usageFromMetadata(usage *bedrocktypes.TokenUsage) *Usage {
	if usage == nil {
		return nil
	}
	prompt := int(aws.ToInt32(usage.InputTokens))
	completion := int(aws.ToInt32(usage.OutputTokens))
	total := int(aws.ToInt32(usage.TotalTokens))
	if total == 0 {
		total = prompt + completion
	}
	return &Usage{
		PromptTokens:     prompt,
		CompletionTokens: completion,
		TotalTokens:      total,
	}
}

func chatCompletionUsage(usage *Usage) *openai.ChatCompletionUsage {
	if usage == nil {
		return nil
	}
	return &openai.ChatCompletionUsage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
	}
}
