package bedrock

import (
	"context"
	"errors"

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
	if loadConfig == nil {
		return nil, errors.New("load config function is required")
	}

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

func (c *Client) Converse(ctx context.Context, input *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	return c.runtime.Converse(ctx, input, optFns...)
}

func (c *Client) ConverseStream(ctx context.Context, input *bedrockruntime.ConverseStreamInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error) {
	return c.runtime.ConverseStream(ctx, input, optFns...)
}
