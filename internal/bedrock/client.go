package bedrock

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

type RuntimeAPI interface {
	Converse(context.Context, *bedrockruntime.ConverseInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
	ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error)
}

type LoadConfigFunc func(context.Context, ...func(*config.LoadOptions) error) (aws.Config, error)

type Client struct {
	runtime RuntimeAPI
}

func NewClient(ctx context.Context, region string, loadConfig LoadConfigFunc) (*Client, error) {
	opts := []func(*config.LoadOptions) error{}
	if region != "" {
		opts = append(opts, config.WithRegion(region))
	}

	awsCfg, err := loadConfig(ctx, opts...)
	if err != nil {
		return nil, err
	}

	return &Client{
		runtime: bedrockruntime.NewFromConfig(awsCfg),
	}, nil
}
