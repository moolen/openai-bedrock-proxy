package openai

import (
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/tiktoken-go/tokenizer"
)

var (
	cl100kCodecOnce sync.Once
	cl100kCodec     tokenizer.Codec
	cl100kCodecErr  error
)

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
		if texts, ok, err := decodeEmbeddingsTokenBatch(value); ok || err != nil {
			return texts, err
		}
		if text, ok, err := decodeEmbeddingsTokenSequence(value); ok || err != nil {
			if err != nil {
				return nil, err
			}
			return []string{text}, nil
		}
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
		if texts, ok, err := decodeEmbeddingsTokenBatch(input); ok || err != nil {
			return texts, err
		}
		if text, ok, err := decodeEmbeddingsTokenSequence(input); ok || err != nil {
			if err != nil {
				return nil, err
			}
			return []string{text}, nil
		}
		return nil, NewInvalidRequestError("input is invalid")
	}
}

func NormalizeSingleStringEmbeddingsInput(input any) (string, error) {
	switch value := input.(type) {
	case string:
		if value == "" {
			return "", NewInvalidRequestError("input is required")
		}
		return value, nil
	case []string:
		if len(value) != 1 || value[0] == "" {
			return "", NewInvalidRequestError("Amazon Titan Embeddings models support only single strings as input")
		}
		return value[0], nil
	case []any:
		if len(value) != 1 {
			return "", NewInvalidRequestError("Amazon Titan Embeddings models support only single strings as input")
		}
		text, ok := value[0].(string)
		if !ok || text == "" {
			return "", NewInvalidRequestError("Amazon Titan Embeddings models support only single strings as input")
		}
		return text, nil
	default:
		return "", NewInvalidRequestError("Amazon Titan Embeddings models support only single strings as input")
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

func decodeEmbeddingsTokenSequence(input any) (string, bool, error) {
	tokens, ok, err := coerceTokenSequence(input)
	if !ok || err != nil {
		return "", ok, err
	}
	text, err := decodeCl100kTokens(tokens)
	if err != nil {
		return "", true, err
	}
	if text == "" {
		return "", true, NewInvalidRequestError("input is invalid")
	}
	return text, true, nil
}

func decodeEmbeddingsTokenBatch(input any) ([]string, bool, error) {
	sequences, ok, err := coerceTokenBatch(input)
	if !ok || err != nil {
		return nil, ok, err
	}
	texts := make([]string, 0, len(sequences))
	for _, sequence := range sequences {
		text, err := decodeCl100kTokens(sequence)
		if err != nil {
			return nil, true, err
		}
		if text == "" {
			return nil, true, NewInvalidRequestError("input is invalid")
		}
		texts = append(texts, text)
	}
	return texts, true, nil
}

func coerceTokenBatch(input any) ([][]uint, bool, error) {
	switch value := input.(type) {
	case [][]int:
		if len(value) == 0 {
			return nil, true, NewInvalidRequestError("input is required")
		}
		out := make([][]uint, 0, len(value))
		for _, item := range value {
			tokens, _, err := coerceTokenSequence(item)
			if err != nil {
				return nil, true, err
			}
			out = append(out, tokens)
		}
		return out, true, nil
	case []any:
		if len(value) == 0 {
			return nil, true, NewInvalidRequestError("input is required")
		}
		out := make([][]uint, 0, len(value))
		for _, item := range value {
			tokens, ok, err := coerceTokenSequence(item)
			if !ok {
				return nil, false, nil
			}
			if err != nil {
				return nil, true, err
			}
			out = append(out, tokens)
		}
		return out, true, nil
	default:
		return nil, false, nil
	}
}

func coerceTokenSequence(input any) ([]uint, bool, error) {
	switch value := input.(type) {
	case []int:
		if len(value) == 0 {
			return nil, true, NewInvalidRequestError("input is required")
		}
		out := make([]uint, 0, len(value))
		for _, item := range value {
			if item < 0 {
				return nil, true, NewInvalidRequestError("input is invalid")
			}
			out = append(out, uint(item))
		}
		return out, true, nil
	case []any:
		if len(value) == 0 {
			return nil, true, NewInvalidRequestError("input is required")
		}
		out := make([]uint, 0, len(value))
		for _, item := range value {
			token, ok := integerLikeToken(item)
			if !ok {
				return nil, false, nil
			}
			out = append(out, token)
		}
		return out, true, nil
	default:
		return nil, false, nil
	}
}

func integerLikeToken(value any) (uint, bool) {
	switch token := value.(type) {
	case int:
		if token < 0 {
			return 0, false
		}
		return uint(token), true
	case int32:
		if token < 0 {
			return 0, false
		}
		return uint(token), true
	case int64:
		if token < 0 {
			return 0, false
		}
		return uint(token), true
	case float64:
		if token < 0 || math.Trunc(token) != token {
			return 0, false
		}
		return uint(token), true
	default:
		return 0, false
	}
}

func decodeCl100kTokens(tokens []uint) (string, error) {
	codec, err := cl100kTokenizer()
	if err != nil {
		return "", NewInvalidRequestError("input is invalid")
	}
	text, err := codec.Decode(tokens)
	if err != nil {
		return "", NewInvalidRequestError(fmt.Sprintf("input is invalid: %v", err))
	}
	return text, nil
}

func cl100kTokenizer() (tokenizer.Codec, error) {
	cl100kCodecOnce.Do(func() {
		cl100kCodec, cl100kCodecErr = tokenizer.Get(tokenizer.Cl100kBase)
	})
	return cl100kCodec, cl100kCodecErr
}
