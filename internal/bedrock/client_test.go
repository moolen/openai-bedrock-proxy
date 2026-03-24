package bedrock

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
)

func TestNewClientUsesDefaultAWSConfig(t *testing.T) {
	called := false
	loader := func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error) {
		called = true
		return aws.Config{Region: "us-west-2"}, nil
	}

	client, err := NewClient(context.Background(), "", loader)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatal("expected AWS config loader to be called")
	}
	if client == nil {
		t.Fatal("expected client to be created")
	}
}

func TestNewClientAppliesRegionOverride(t *testing.T) {
	var captured config.LoadOptions
	loader := func(ctx context.Context, optFns ...func(*config.LoadOptions) error) (aws.Config, error) {
		for _, optFn := range optFns {
			if err := optFn(&captured); err != nil {
				t.Fatalf("unexpected option error: %v", err)
			}
		}
		return aws.Config{Region: "ignored-by-test"}, nil
	}

	_, err := NewClient(context.Background(), "eu-central-1", loader)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if captured.Region != "eu-central-1" {
		t.Fatalf("expected region override, got %q", captured.Region)
	}
}
