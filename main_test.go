package main

import (
	"os/exec"
	"testing"
)

func TestMainServerBuilds(t *testing.T) {
	cmd := exec.Command("go", "build", "./cmd/openai-bedrock-proxy")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}
}
