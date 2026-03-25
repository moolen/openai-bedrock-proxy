package openai

type UsageTotals struct {
	Requests         int `json:"requests"`
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type UsageSummary struct {
	Object string      `json:"object"`
	Totals UsageTotals `json:"totals"`
}

type UsageSession struct {
	Object    string      `json:"object"`
	SessionID string      `json:"session_id"`
	Totals    UsageTotals `json:"totals"`
}
