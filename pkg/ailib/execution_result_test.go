package ailib

import (
	"encoding/json"
	"testing"
)

func TestExecutionMetadataJSON(t *testing.T) {
	code := "E2001"
	m := ExecutionMetadata{
		ProviderID:           "mock-openai",
		ModelID:              "gpt-test",
		ExecutionLatencyMs:   10,
		TranslationLatencyMs: 2,
		MicroRetryCount:      1,
		ErrorCode:            &code,
		Usage:                &ExecutionUsage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8},
	}
	b, err := json.Marshal(&m)
	if err != nil {
		t.Fatal(err)
	}
	var out ExecutionMetadata
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if out.ProviderID != m.ProviderID || out.ModelID != m.ModelID {
		t.Fatalf("roundtrip: %+v vs %+v", out, m)
	}
	if out.ErrorCode == nil || *out.ErrorCode != code {
		t.Fatalf("error_code: %v", out.ErrorCode)
	}
}

func TestExecutionResultJSON(t *testing.T) {
	er := ExecutionResult[string]{
		Data: "hello",
		Metadata: ExecutionMetadata{
			ProviderID: "p",
			ModelID:    "m",
		},
	}
	b, err := json.Marshal(&er)
	if err != nil {
		t.Fatal(err)
	}
	var out ExecutionResult[string]
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if out.Data != "hello" {
		t.Fatalf("data %q", out.Data)
	}
}
