package bedrock

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/moolen/openai-bedrock-proxy/internal/openai"
)

const (
	embeddingsContentType = "application/json"
	novaDefaultDimension  = 3072
)

var validNovaEmbeddingDimensions = map[int]struct{}{
	256:  {},
	384:  {},
	1024: {},
	3072: {},
}

type EmbeddingAdapter interface {
	Name() string
	Embed(context.Context, RuntimeAPI, string, openai.EmbeddingsRequest) (openai.EmbeddingsResponse, error)
}

type cohereEmbeddingAdapter struct{}
type titanEmbeddingAdapter struct{}
type novaEmbeddingAdapter struct{}

func selectEmbeddingAdapter(modelID string) (EmbeddingAdapter, error) {
	switch {
	case strings.Contains(modelID, "cohere.embed-english-v3"),
		strings.Contains(modelID, "cohere.embed-multilingual-v3"):
		return cohereEmbeddingAdapter{}, nil
	case strings.Contains(modelID, "amazon.titan-embed-text-v1"),
		strings.Contains(modelID, "amazon.titan-embed-text-v2:0"):
		return titanEmbeddingAdapter{}, nil
	case strings.Contains(modelID, "amazon.nova-2-multimodal-embeddings-v1:0"):
		return novaEmbeddingAdapter{}, nil
	default:
		return nil, openai.NewInvalidRequestError("unsupported embedding model id " + modelID)
	}
}

func (cohereEmbeddingAdapter) Name() string {
	return "cohere"
}

func (cohereEmbeddingAdapter) Embed(ctx context.Context, runtime RuntimeAPI, modelID string, req openai.EmbeddingsRequest) (openai.EmbeddingsResponse, error) {
	if req.Dimensions != nil {
		return openai.EmbeddingsResponse{}, openai.NewInvalidRequestError("dimensions is only supported for Nova embedding models")
	}

	texts, err := openai.NormalizeEmbeddingsInput(req.Input)
	if err != nil {
		return openai.EmbeddingsResponse{}, err
	}

	var payload struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := invokeModelJSON(ctx, runtime, modelID, map[string]any{
		"texts":      texts,
		"input_type": "search_document",
		"truncate":   "END",
	}, &payload); err != nil {
		return openai.EmbeddingsResponse{}, err
	}

	return newEmbeddingsResponse(req.Model, payload.Embeddings, 0, req.EncodingFormat), nil
}

func (titanEmbeddingAdapter) Name() string {
	return "titan"
}

func (titanEmbeddingAdapter) Embed(ctx context.Context, runtime RuntimeAPI, modelID string, req openai.EmbeddingsRequest) (openai.EmbeddingsResponse, error) {
	if req.Dimensions != nil {
		return openai.EmbeddingsResponse{}, openai.NewInvalidRequestError("dimensions is only supported for Nova embedding models")
	}

	text, err := openai.NormalizeSingleStringEmbeddingsInput(req.Input)
	if err != nil {
		return openai.EmbeddingsResponse{}, err
	}

	var payload struct {
		Embedding           []float64 `json:"embedding"`
		InputTextTokenCount int       `json:"inputTextTokenCount"`
	}
	if err := invokeModelJSON(ctx, runtime, modelID, map[string]any{
		"inputText": text,
	}, &payload); err != nil {
		return openai.EmbeddingsResponse{}, err
	}

	return newEmbeddingsResponse(req.Model, [][]float64{payload.Embedding}, payload.InputTextTokenCount, req.EncodingFormat), nil
}

func (novaEmbeddingAdapter) Name() string {
	return "nova"
}

func (novaEmbeddingAdapter) Embed(ctx context.Context, runtime RuntimeAPI, modelID string, req openai.EmbeddingsRequest) (openai.EmbeddingsResponse, error) {
	texts, err := openai.NormalizeEmbeddingsInput(req.Input)
	if err != nil {
		return openai.EmbeddingsResponse{}, err
	}

	dimension := novaDefaultDimension
	if req.Dimensions != nil {
		dimension = *req.Dimensions
	}
	if _, ok := validNovaEmbeddingDimensions[dimension]; !ok {
		return openai.EmbeddingsResponse{}, openai.NewInvalidRequestError("dimensions must be one of 256, 384, 1024, 3072")
	}

	embeddings := make([][]float64, 0, len(texts))
	for _, text := range texts {
		var payload struct {
			Embeddings []struct {
				Embedding []float64 `json:"embedding"`
			} `json:"embeddings"`
		}
		if err := invokeModelJSON(ctx, runtime, modelID, map[string]any{
			"taskType": "SINGLE_EMBEDDING",
			"singleEmbeddingParams": map[string]any{
				"embeddingPurpose":   "GENERIC_INDEX",
				"embeddingDimension": dimension,
				"text": map[string]any{
					"truncationMode": "END",
					"value":          text,
				},
			},
		}, &payload); err != nil {
			return openai.EmbeddingsResponse{}, err
		}
		if len(payload.Embeddings) == 0 {
			return openai.EmbeddingsResponse{}, errors.New("no embeddings returned from nova model")
		}
		embeddings = append(embeddings, payload.Embeddings[0].Embedding)
	}

	return newEmbeddingsResponse(req.Model, embeddings, 0, req.EncodingFormat), nil
}

func invokeModelJSON(ctx context.Context, runtime RuntimeAPI, modelID string, requestBody any, out any) error {
	body, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("marshal invoke model request: %w", err)
	}

	resp, err := runtime.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		Accept:      aws.String(embeddingsContentType),
		ContentType: aws.String(embeddingsContentType),
		Body:        body,
	})
	if err != nil {
		return err
	}
	if resp == nil {
		return errors.New("bedrock invoke model returned nil response")
	}
	if err := json.Unmarshal(resp.Body, out); err != nil {
		return fmt.Errorf("decode invoke model response: %w", err)
	}
	return nil
}

func newEmbeddingsResponse(model string, vectors [][]float64, promptTokens int, encodingFormat string) openai.EmbeddingsResponse {
	data := make([]openai.Embedding, 0, len(vectors))
	for idx, vector := range vectors {
		data = append(data, openai.Embedding{
			Object:    "embedding",
			Embedding: formatEmbedding(vector, encodingFormat),
			Index:     idx,
		})
	}

	return openai.EmbeddingsResponse{
		Object: "list",
		Data:   data,
		Model:  model,
		Usage: openai.EmbeddingsUsage{
			PromptTokens: promptTokens,
			TotalTokens:  promptTokens,
		},
	}
}

func formatEmbedding(vector []float64, encodingFormat string) any {
	if strings.TrimSpace(encodingFormat) != "base64" {
		return vector
	}

	buf := make([]byte, 4*len(vector))
	for idx, value := range vector {
		binary.LittleEndian.PutUint32(buf[idx*4:], math.Float32bits(float32(value)))
	}
	return base64.StdEncoding.EncodeToString(buf)
}
