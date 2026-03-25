package openai

import "strings"

func ValidateEmbeddingsRequest(req EmbeddingsRequest) error {
	if strings.TrimSpace(req.Model) == "" {
		return NewInvalidRequestError("model is required")
	}
	if _, err := NormalizeEmbeddingsInput(req.Input); err != nil {
		return err
	}

	switch strings.TrimSpace(req.EncodingFormat) {
	case "", "float", "base64":
	default:
		return NewInvalidRequestError("encoding_format is invalid")
	}

	if req.Dimensions != nil {
		if *req.Dimensions <= 0 {
			return NewInvalidRequestError("dimensions must be greater than 0")
		}
		if knownEmbeddingModelFamily(req.Model) == "non-nova" {
			return NewInvalidRequestError("dimensions is only supported for Nova embedding models")
		}
	}

	return nil
}

func NormalizeEmbeddingsInput(input any) ([]string, error) {
	switch value := input.(type) {
	case nil:
		return nil, NewInvalidRequestError("input is required")
	case string:
		if value == "" {
			return nil, NewInvalidRequestError("input is required")
		}
		return []string{value}, nil
	case []string:
		return validateEmbeddingsInputSlice(value)
	case []any:
		texts := make([]string, 0, len(value))
		for _, item := range value {
			text, ok := item.(string)
			if !ok {
				return nil, NewInvalidRequestError("input is invalid")
			}
			texts = append(texts, text)
		}
		return validateEmbeddingsInputSlice(texts)
	default:
		return nil, NewInvalidRequestError("input is invalid")
	}
}

func validateEmbeddingsInputSlice(items []string) ([]string, error) {
	if len(items) == 0 {
		return nil, NewInvalidRequestError("input is required")
	}
	texts := make([]string, 0, len(items))
	for _, item := range items {
		if item == "" {
			return nil, NewInvalidRequestError("input is invalid")
		}
		texts = append(texts, item)
	}
	return texts, nil
}

func knownEmbeddingModelFamily(model string) string {
	switch {
	case strings.Contains(model, "amazon.nova-2-multimodal-embeddings-v1:0"):
		return "nova"
	case strings.Contains(model, "cohere.embed-english-v3"),
		strings.Contains(model, "cohere.embed-multilingual-v3"),
		strings.Contains(model, "amazon.titan-embed-text-v1"),
		strings.Contains(model, "amazon.titan-embed-text-v2:0"):
		return "non-nova"
	default:
		return ""
	}
}
