package bedrock

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrocksvc "github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
)

func TestCatalogListModelsMergesFoundationAndInferenceProfiles(t *testing.T) {
	catalog := newFakeCatalogAPI()
	catalog.foundationModels = []ModelRecord{
		{ID: "anthropic.claude-3-7-sonnet-20250219-v1:0", InputModalities: []string{"TEXT", "IMAGE"}},
	}
	catalog.systemProfiles = []InferenceProfileRecord{
		{ID: "us.anthropic.claude-3-7-sonnet-20250219-v1:0", SourceModelID: "anthropic.claude-3-7-sonnet-20250219-v1:0"},
	}
	catalog.applicationProfiles = []InferenceProfileRecord{
		{ID: "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/app-profile", SourceModelID: "anthropic.claude-3-7-sonnet-20250219-v1:0"},
	}
	got, err := BuildModelCatalog(context.Background(), catalog)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Models) != 3 {
		t.Fatalf("expected merged catalog, got %#v", got.Models)
	}
}

func TestCatalogResolveModelReturnsUnderlyingFoundationModel(t *testing.T) {
	catalog := Catalog{
		ByID: map[string]ModelRecord{
			"us.anthropic.claude-3-7-sonnet-20250219-v1:0": {
				ID:                        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
				ResolvedFoundationModelID: "anthropic.claude-3-7-sonnet-20250219-v1:0",
			},
		},
	}
	got, ok := catalog.Resolve("us.anthropic.claude-3-7-sonnet-20250219-v1:0")
	if !ok || got.ResolvedFoundationModelID != "anthropic.claude-3-7-sonnet-20250219-v1:0" {
		t.Fatalf("expected profile resolution, got %#v ok=%v", got, ok)
	}
}

func newFakeCatalogAPI() *fakeCatalogAPI {
	return &fakeCatalogAPI{}
}

type fakeCatalogAPI struct {
	foundationModels    []ModelRecord
	systemProfiles      []InferenceProfileRecord
	applicationProfiles []InferenceProfileRecord
}

func (f *fakeCatalogAPI) ListFoundationModels(_ context.Context, _ *bedrocksvc.ListFoundationModelsInput, _ ...func(*bedrocksvc.Options)) (*bedrocksvc.ListFoundationModelsOutput, error) {
	summaries := make([]bedrocktypes.FoundationModelSummary, 0, len(f.foundationModels))
	for _, model := range f.foundationModels {
		inputModalities := make([]bedrocktypes.ModelModality, 0, len(model.InputModalities))
		for _, modality := range model.InputModalities {
			inputModalities = append(inputModalities, bedrocktypes.ModelModality(modality))
		}
		summaries = append(summaries, bedrocktypes.FoundationModelSummary{
			ModelId:         aws.String(model.ID),
			ModelName:       aws.String(model.Name),
			ProviderName:    aws.String(model.Provider),
			InputModalities: inputModalities,
		})
	}
	return &bedrocksvc.ListFoundationModelsOutput{ModelSummaries: summaries}, nil
}

func (f *fakeCatalogAPI) ListInferenceProfiles(_ context.Context, input *bedrocksvc.ListInferenceProfilesInput, _ ...func(*bedrocksvc.Options)) (*bedrocksvc.ListInferenceProfilesOutput, error) {
	var source []InferenceProfileRecord
	switch input.TypeEquals {
	case bedrocktypes.InferenceProfileTypeSystemDefined:
		source = f.systemProfiles
	case bedrocktypes.InferenceProfileTypeApplication:
		source = f.applicationProfiles
	}

	summaries := make([]bedrocktypes.InferenceProfileSummary, 0, len(source))
	for _, profile := range source {
		summaries = append(summaries, bedrocktypes.InferenceProfileSummary{
			InferenceProfileId:   aws.String(profile.ID),
			InferenceProfileName: aws.String(profile.Name),
			Models: []bedrocktypes.InferenceProfileModel{
				{ModelArn: aws.String("arn:aws:bedrock:us-west-2::foundation-model/" + profile.SourceModelID)},
			},
		})
	}
	return &bedrocksvc.ListInferenceProfilesOutput{InferenceProfileSummaries: summaries}, nil
}
