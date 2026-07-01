// Package integration_test exercises ai-protocol-mock when MOCK_HTTP_URL is set.
// 当设置 MOCK_HTTP_URL 时，对 ai-protocol-mock 做冒烟集成测试。
package integration_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

func mockBaseURL(t *testing.T) string {
	t.Helper()
	base := strings.TrimRight(os.Getenv("MOCK_HTTP_URL"), "/")
	if base == "" {
		t.Skip("MOCK_HTTP_URL not set")
	}
	return base
}

func TestMockHealth(t *testing.T) {
	base := mockBaseURL(t)
	resp, err := http.Get(base + "/health")
	if err != nil {
		t.Fatalf("health request: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("health status: %d", resp.StatusCode)
	}
}

func TestMockOpenAIChat(t *testing.T) {
	base := mockBaseURL(t)
	payload := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`)
	resp, err := http.Post(base+"/v1/chat/completions", "application/json", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("chat request: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("chat status %d: %s", resp.StatusCode, body)
	}
	var data map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	choices, ok := data["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatal("expected non-empty choices array")
	}
}
