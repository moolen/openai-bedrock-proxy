package bedrock

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
	"github.com/tiktoken-go/tokenizer"
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

func TestCohereEmbeddingAdapterDecodesTokenArrayInput(t *testing.T) {
	runtime := &captureInvokeRuntime{
		outputs: []*bedrockruntime.InvokeModelOutput{{
			ContentType: aws.String("application/json"),
			Body:        []byte(`{"embeddings":[[0.1,0.2]]}`),
		}},
	}
	input := mustCl100kEncode(t, "hello")

	_, err := cohereEmbeddingAdapter{}.Embed(context.Background(), runtime, "cohere.embed-english-v3", openai.EmbeddingsRequest{
		Model: "cohere.embed-english-v3",
		Input: input,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(runtime.inputs) != 1 {
		t.Fatalf("expected one invoke model call, got %d", len(runtime.inputs))
	}

	var body map[string]any
	if err := json.Unmarshal(runtime.inputs[0].Body, &body); err != nil {
		t.Fatalf("unexpected request body JSON error: %v", err)
	}
	texts, ok := body["texts"].([]any)
	if !ok || len(texts) != 1 || texts[0] != "hello" {
		t.Fatalf("expected decoded text payload, got %#v", body)
	}
}

func TestNovaEmbeddingAdapterDecodesTokenArrayBatchInput(t *testing.T) {
	runtime := &captureInvokeRuntime{
		outputs: []*bedrockruntime.InvokeModelOutput{
			{ContentType: aws.String("application/json"), Body: []byte(`{"embeddings":[{"embedding":[0.1,0.2]}]}`)},
			{ContentType: aws.String("application/json"), Body: []byte(`{"embeddings":[{"embedding":[0.3,0.4]}]}`)},
		},
	}
	input := [][]int{
		mustCl100kEncode(t, "hello"),
		mustCl100kEncode(t, "world"),
	}

	_, err := novaEmbeddingAdapter{}.Embed(context.Background(), runtime, "amazon.nova-2-multimodal-embeddings-v1:0", openai.EmbeddingsRequest{
		Model: "amazon.nova-2-multimodal-embeddings-v1:0",
		Input: input,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(runtime.inputs) != 2 {
		t.Fatalf("expected two invoke model calls, got %d", len(runtime.inputs))
	}

	assertNovaTextValue(t, runtime.inputs[0].Body, "hello")
	assertNovaTextValue(t, runtime.inputs[1].Body, "world")
}

func TestNovaEmbeddingAdapterDecodesJSONDecodedTokenArrayBatchInput(t *testing.T) {
	runtime := &captureInvokeRuntime{
		outputs: []*bedrockruntime.InvokeModelOutput{
			{ContentType: aws.String("application/json"), Body: []byte(`{"embeddings":[{"embedding":[0.1,0.2]}]}`)},
			{ContentType: aws.String("application/json"), Body: []byte(`{"embeddings":[{"embedding":[0.3,0.4]}]}`)},
		},
	}
	hello := mustCl100kEncode(t, "hello")
	world := mustCl100kEncode(t, "world")
	input := []any{
		toJSONNumbers(hello),
		toJSONNumbers(world),
	}

	_, err := novaEmbeddingAdapter{}.Embed(context.Background(), runtime, "amazon.nova-2-multimodal-embeddings-v1:0", openai.EmbeddingsRequest{
		Model: "amazon.nova-2-multimodal-embeddings-v1:0",
		Input: input,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(runtime.inputs) != 2 {
		t.Fatalf("expected two invoke model calls, got %d", len(runtime.inputs))
	}

	assertNovaTextValue(t, runtime.inputs[0].Body, "hello")
	assertNovaTextValue(t, runtime.inputs[1].Body, "world")
}

func TestTitanEmbeddingAdapterRejectsTokenArrayInput(t *testing.T) {
	input := mustCl100kEncode(t, "hello")

	_, err := titanEmbeddingAdapter{}.Embed(context.Background(), &fakeRuntime{}, "amazon.titan-embed-text-v2:0", openai.EmbeddingsRequest{
		Model: "amazon.titan-embed-text-v2:0",
		Input: input,
	})
	assertInvalidRequestError(t, err)
}

type captureInvokeRuntime struct {
	inputs  []*bedrockruntime.InvokeModelInput
	outputs []*bedrockruntime.InvokeModelOutput
}

func (r *captureInvokeRuntime) Converse(context.Context, *bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	return nil, nil
}

func (r *captureInvokeRuntime) ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error) {
	return nil, nil
}

func (r *captureInvokeRuntime) InvokeModel(_ context.Context, input *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	r.inputs = append(r.inputs, input)
	if len(r.outputs) == 0 {
		return nil, nil
	}
	out := r.outputs[0]
	r.outputs = r.outputs[1:]
	return out, nil
}

func mustCl100kEncode(t *testing.T, text string) []int {
	t.Helper()

	codec, err := tokenizer.Get(tokenizer.Cl100kBase)
	if err != nil {
		t.Fatalf("unexpected tokenizer error: %v", err)
	}
	ids, _, err := codec.Encode(text)
	if err != nil {
		t.Fatalf("unexpected encode error: %v", err)
	}

	out := make([]int, 0, len(ids))
	for _, id := range ids {
		out = append(out, int(id))
	}
	return out
}

func assertNovaTextValue(t *testing.T, body []byte, want string) {
	t.Helper()

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unexpected request body JSON error: %v", err)
	}
	params, ok := payload["singleEmbeddingParams"].(map[string]any)
	if !ok {
		t.Fatalf("expected singleEmbeddingParams object, got %#v", payload)
	}
	text, ok := params["text"].(map[string]any)
	if !ok || text["value"] != want {
		t.Fatalf("expected decoded nova text %q, got %#v", want, payload)
	}
}

func toJSONNumbers(tokens []int) []any {
	out := make([]any, 0, len(tokens))
	for _, token := range tokens {
		out = append(out, float64(token))
	}
	return out
}
