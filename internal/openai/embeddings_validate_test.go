package openai

import "testing"

func TestValidateEmbeddingsRequestAcceptsFloatAndBase64(t *testing.T) {
	for _, format := range []string{"float", "base64"} {
		req := EmbeddingsRequest{
			Model:          "cohere.embed-english-v3",
			Input:          "hello",
			EncodingFormat: format,
		}
		if err := ValidateEmbeddingsRequest(req); err != nil {
			t.Fatalf("expected %s to validate, got %v", format, err)
		}
	}
}

func TestValidateEmbeddingsRequestRejectsDimensionsForNonNovaModel(t *testing.T) {
	req := EmbeddingsRequest{
		Model:      "cohere.embed-english-v3",
		Input:      "hello",
		Dimensions: ptr(256),
	}
	assertInvalidRequestMessage(t, ValidateEmbeddingsRequest(req), "dimensions is only supported for Nova embedding models")
}

func TestValidateEmbeddingsRequestAcceptsStringArrayInput(t *testing.T) {
	req := EmbeddingsRequest{
		Model: "cohere.embed-english-v3",
		Input: []string{"hello", "world"},
	}
	if err := ValidateEmbeddingsRequest(req); err != nil {
		t.Fatalf("expected string array input to validate, got %v", err)
	}
}

func TestValidateEmbeddingsRequestAcceptsJSONDecodedStringArrayInput(t *testing.T) {
	req := EmbeddingsRequest{
		Model: "cohere.embed-english-v3",
		Input: []any{"hello", "world"},
	}
	if err := ValidateEmbeddingsRequest(req); err != nil {
		t.Fatalf("expected JSON-decoded string array input to validate, got %v", err)
	}
}

func TestValidateEmbeddingsRequestAcceptsTokenArrayInput(t *testing.T) {
	req := EmbeddingsRequest{
		Model: "cohere.embed-english-v3",
		Input: []int{15339},
	}
	if err := ValidateEmbeddingsRequest(req); err != nil {
		t.Fatalf("expected token array input to validate, got %v", err)
	}
}

func TestValidateEmbeddingsRequestAcceptsTokenArrayBatchInput(t *testing.T) {
	req := EmbeddingsRequest{
		Model: "cohere.embed-english-v3",
		Input: [][]int{{15339}, {14957}},
	}
	if err := ValidateEmbeddingsRequest(req); err != nil {
		t.Fatalf("expected token array batch input to validate, got %v", err)
	}
}

func TestValidateEmbeddingsRequestAcceptsJSONDecodedTokenArrayInput(t *testing.T) {
	req := EmbeddingsRequest{
		Model: "cohere.embed-english-v3",
		Input: []any{float64(15339)},
	}
	if err := ValidateEmbeddingsRequest(req); err != nil {
		t.Fatalf("expected JSON-decoded token array input to validate, got %v", err)
	}
}

func TestValidateEmbeddingsRequestAcceptsJSONDecodedTokenArrayBatchInput(t *testing.T) {
	req := EmbeddingsRequest{
		Model: "cohere.embed-english-v3",
		Input: []any{
			[]any{float64(15339)},
			[]any{float64(14957)},
		},
	}
	if err := ValidateEmbeddingsRequest(req); err != nil {
		t.Fatalf("expected JSON-decoded token array batch input to validate, got %v", err)
	}
}
