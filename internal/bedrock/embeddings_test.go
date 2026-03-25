package bedrock

import (
	"context"
	"testing"

	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

func TestSelectEmbeddingAdapterByModelFamily(t *testing.T) {
	for _, tc := range []struct {
		model string
		want  string
	}{
		{model: "cohere.embed-english-v3", want: "cohere"},
		{model: "amazon.titan-embed-text-v2:0", want: "titan"},
		{model: "amazon.nova-2-multimodal-embeddings-v1:0", want: "nova"},
	} {
		got, err := selectEmbeddingAdapter(tc.model)
		if err != nil {
			t.Fatalf("model %q returned unexpected error: %v", tc.model, err)
		}
		if got.Name() != tc.want {
			t.Fatalf("model %q expected adapter %q, got %q", tc.model, tc.want, got.Name())
		}
	}
}

func TestSelectEmbeddingAdapterRejectsUnsupportedModel(t *testing.T) {
	_, err := selectEmbeddingAdapter("unsupported-model")
	assertInvalidRequestError(t, err)
}

func TestFormatEmbeddingEncodesBase64Float32Bytes(t *testing.T) {
	got := formatEmbedding([]float64{1, 2}, "base64")
	encoded, ok := got.(string)
	if !ok {
		t.Fatalf("expected base64 string embedding, got %T", got)
	}
	if encoded != "AACAPwAAAEA=" {
		t.Fatalf("expected float32 base64 encoding, got %q", encoded)
	}
}

func TestTitanEmbeddingAdapterRejectsBatchInput(t *testing.T) {
	_, err := titanEmbeddingAdapter{}.Embed(context.Background(), &fakeRuntime{}, "amazon.titan-embed-text-v2:0", openai.EmbeddingsRequest{
		Model: "amazon.titan-embed-text-v2:0",
		Input: []string{"hello", "world"},
	})
	assertInvalidRequestError(t, err)
}

func TestNovaEmbeddingAdapterRejectsUnsupportedDimensions(t *testing.T) {
	dimensions := 512
	_, err := novaEmbeddingAdapter{}.Embed(context.Background(), &fakeRuntime{}, "amazon.nova-2-multimodal-embeddings-v1:0", openai.EmbeddingsRequest{
		Model:      "amazon.nova-2-multimodal-embeddings-v1:0",
		Input:      "hello",
		Dimensions: &dimensions,
	})
	assertInvalidRequestError(t, err)
}
