package bedrock

import (
	"context"
	"sort"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	bedrockcatalog "github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrockcatalogtypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
)

const (
	modelKindFoundationModel  = "FOUNDATION_MODEL"
	modelKindInferenceProfile = "INFERENCE_PROFILE"
)

type ModelRecord struct {
	ID                        string
	Name                      string
	Provider                  string
	InputModalities           []string
	ModelKind                 string
	ResolvedFoundationModelID string
}

type InferenceProfileRecord struct {
	ID            string
	Name          string
	SourceModelID string
}

type Catalog struct {
	Models []ModelRecord
	ByID   map[string]ModelRecord
}

func (c Catalog) Resolve(id string) (ModelRecord, bool) {
	record, ok := c.ByID[id]
	return record, ok
}

func BuildModelCatalog(ctx context.Context, api CatalogAPI) (Catalog, error) {
	foundationModels, err := listFoundationModels(ctx, api)
	if err != nil {
		return Catalog{}, err
	}

	foundationByID := make(map[string]ModelRecord, len(foundationModels))
	byID := make(map[string]ModelRecord, len(foundationModels))
	for _, model := range foundationModels {
		foundationByID[model.ID] = model
		byID[model.ID] = model
	}

	systemProfiles, err := listInferenceProfiles(ctx, api, bedrockcatalogtypes.InferenceProfileTypeSystemDefined)
	if err != nil {
		return Catalog{}, err
	}
	applicationProfiles, err := listInferenceProfiles(ctx, api, bedrockcatalogtypes.InferenceProfileTypeApplication)
	if err != nil {
		return Catalog{}, err
	}
	for _, profile := range append(systemProfiles, applicationProfiles...) {
		if profile.ID == "" || profile.SourceModelID == "" {
			continue
		}
		record := ModelRecord{
			ID:                        profile.ID,
			Name:                      profile.Name,
			ModelKind:                 modelKindInferenceProfile,
			ResolvedFoundationModelID: profile.SourceModelID,
		}
		if source, ok := foundationByID[profile.SourceModelID]; ok {
			record.Provider = source.Provider
			record.InputModalities = append([]string(nil), source.InputModalities...)
			if record.Name == "" {
				record.Name = source.Name
			}
		}
		byID[record.ID] = record
	}

	ids := make([]string, 0, len(byID))
	for id := range byID {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	models := make([]ModelRecord, 0, len(ids))
	for _, id := range ids {
		models = append(models, byID[id])
	}

	return Catalog{
		Models: models,
		ByID:   byID,
	}, nil
}

func listFoundationModels(ctx context.Context, api CatalogAPI) ([]ModelRecord, error) {
	resp, err := api.ListFoundationModels(ctx, &bedrockcatalog.ListFoundationModelsInput{
		ByOutputModality: bedrockcatalogtypes.ModelModalityText,
	})
	if err != nil {
		return nil, err
	}

	models := make([]ModelRecord, 0, len(resp.ModelSummaries))
	for _, model := range resp.ModelSummaries {
		modelID := aws.ToString(model.ModelId)
		if modelID == "" {
			continue
		}

		inputModalities := make([]string, 0, len(model.InputModalities))
		for _, modality := range model.InputModalities {
			inputModalities = append(inputModalities, string(modality))
		}

		models = append(models, ModelRecord{
			ID:              modelID,
			Name:            aws.ToString(model.ModelName),
			Provider:        aws.ToString(model.ProviderName),
			InputModalities: inputModalities,
			ModelKind:       modelKindFoundationModel,
		})
	}
	return models, nil
}

func listInferenceProfiles(ctx context.Context, api CatalogAPI, profileType bedrockcatalogtypes.InferenceProfileType) ([]InferenceProfileRecord, error) {
	out := []InferenceProfileRecord{}
	var nextToken *string
	for {
		resp, err := api.ListInferenceProfiles(ctx, &bedrockcatalog.ListInferenceProfilesInput{
			TypeEquals: profileType,
			NextToken:  nextToken,
		})
		if err != nil {
			return nil, err
		}

		for _, profile := range resp.InferenceProfileSummaries {
			out = append(out, InferenceProfileRecord{
				ID:            aws.ToString(profile.InferenceProfileId),
				Name:          aws.ToString(profile.InferenceProfileName),
				SourceModelID: profileSourceModelID(profile),
			})
		}

		if aws.ToString(resp.NextToken) == "" {
			break
		}
		nextToken = resp.NextToken
	}
	return out, nil
}

func profileSourceModelID(profile bedrockcatalogtypes.InferenceProfileSummary) string {
	for _, model := range profile.Models {
		if modelID := modelIDFromARN(aws.ToString(model.ModelArn)); modelID != "" {
			return modelID
		}
	}
	return ""
}

func modelIDFromARN(modelARN string) string {
	if modelARN == "" {
		return ""
	}
	if !strings.HasPrefix(modelARN, "arn:") {
		return modelARN
	}
	parts := strings.SplitN(modelARN, "/", 2)
	if len(parts) != 2 {
		return ""
	}
	return strings.TrimSpace(parts[1])
}
