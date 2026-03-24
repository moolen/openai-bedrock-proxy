package bedrock

import "github.com/moolen/openai-bedrock-proxy/internal/openai"

type ConverseResponse struct {
	ResponseID string
	Text       string
	StopReason string
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
