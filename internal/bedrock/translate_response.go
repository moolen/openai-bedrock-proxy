package bedrock

import (
	"strings"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

type ConverseResponse struct {
	ResponseID string
	Text       string
	StopReason string
}

type TextAccumulator struct {
	builder strings.Builder
}

func (t *TextAccumulator) Add(delta string) {
	t.builder.WriteString(delta)
}

func (t *TextAccumulator) Text() string {
	return t.builder.String()
}

func TranslateResponse(resp ConverseResponse, model string) openai.Response {
	return openai.Response{
		ID:     "resp_" + resp.ResponseID,
		Object: "response",
		Model:  model,
		Output: []openai.OutputItem{
			{
				Type: "message",
				Role: "assistant",
				Content: []openai.ContentItem{
					{
						Type: "output_text",
						Text: resp.Text,
					},
				},
			},
		},
	}
}
