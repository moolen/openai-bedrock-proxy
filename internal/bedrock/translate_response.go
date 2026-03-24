package bedrock

import "github.com/moolen/openai-bedrock-proxy/internal/openai"

type ConverseResponse struct {
	ResponseID string
	Text       string
	StopReason string
}

type TextAccumulator struct {
	text string
}

func (t *TextAccumulator) Add(delta string) {
	t.text += delta
}

func (t *TextAccumulator) Text() string {
	return t.text
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
