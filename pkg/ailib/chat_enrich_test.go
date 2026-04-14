package ailib

import (
	"encoding/json"
	"testing"

	"github.com/ailib-official/ai-lib-go/internal/protocol"
)

func TestEnrichNonstream_ResponsePathsContent(t *testing.T) {
	m := &protocol.V2Manifest{
		ID:              "p",
		ProtocolVersion: "2.0",
		Endpoint:        protocol.EndpointConfig{BaseURL: "https://example.com"},
		ResponsePaths: &protocol.ResponsePathsConfig{
			Content: "custom.text",
			Usage:   "u",
		},
	}
	raw := map[string]any{
		"choices": []any{map[string]any{"message": map[string]any{"content": "ignored"}}},
		"custom":  map[string]any{"text": "from_manifest"},
		"u":       map[string]any{"prompt_tokens": 1.0, "completion_tokens": 0.0, "total_tokens": 1.0},
	}
	var out ChatResponse
	_ = json.Unmarshal(mustJSON(t, raw), &out)
	EnrichNonstreamChatResponse(m, raw, &out)
	got, _ := out.Choices[0].Message.Content.(string)
	if got != "from_manifest" {
		t.Fatalf("content: want from_manifest got %q", got)
	}
	if out.Usage == nil || out.Usage.TotalTokens != 1 {
		t.Fatalf("usage: %+v", out.Usage)
	}
}

func TestEnrichNonstream_OpenAIReasoningFallback(t *testing.T) {
	m := &protocol.V2Manifest{
		ID:              "p",
		ProtocolVersion: "2.0",
		Endpoint:        protocol.EndpointConfig{BaseURL: "https://example.com"},
	}
	raw := map[string]any{
		"choices": []any{map[string]any{
			"message": map[string]any{
				"content":           "",
				"reasoning_content": "think",
				"role":              "assistant",
			},
			"finish_reason": "stop",
		}},
	}
	var out ChatResponse
	_ = json.Unmarshal(mustJSON(t, raw), &out)
	EnrichNonstreamChatResponse(m, raw, &out)
	got, _ := out.Choices[0].Message.Content.(string)
	if got != "think" {
		t.Fatalf("content: want think got %q", got)
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
