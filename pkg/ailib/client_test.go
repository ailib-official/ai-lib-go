package ailib

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestClientChat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"id":"r1","model":"m1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	c, err := NewClientBuilder().
		WithBaseURL(srv.URL).
		WithTimeout(5 * time.Second).
		Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	resp, err := c.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, &ChatOptions{Model: "m1"})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices mismatch: %d", len(resp.Choices))
	}
}

func TestClientChatStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"he\"},\"finish_reason\":\"\"}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c, err := NewClientBuilder().WithBaseURL(srv.URL).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	st, err := c.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, &ChatOptions{Model: "m1"})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer st.Close()

	if !st.Next() {
		t.Fatalf("expected first event")
	}
	if st.Event().Delta != "he" {
		t.Fatalf("delta mismatch: %q", st.Event().Delta)
	}
}

func TestClientChatParsesExtendedUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"r1",
			"model":"m1",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}],
			"usage":{
				"prompt_tokens":10,
				"completion_tokens":5,
				"total_tokens":15,
				"reasoning_tokens":3,
				"cache_read_input_tokens":2,
				"cache_creation_input_tokens":1,
				"completion_tokens_details":{"reasoning_tokens":3}
			}
		}`))
	}))
	defer srv.Close()

	c, err := NewClientBuilder().WithBaseURL(srv.URL).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	resp, err := c.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, &ChatOptions{Model: "m1"})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if resp.Usage == nil {
		t.Fatalf("expected usage")
	}
	if resp.Usage.ReasoningTokens != 3 {
		t.Fatalf("reasoning_tokens mismatch: %d", resp.Usage.ReasoningTokens)
	}
	if resp.Usage.CacheReadTokens != 2 || resp.Usage.CacheCreationTokens != 1 {
		t.Fatalf("cache token mismatch: %+v", resp.Usage)
	}
	if resp.Usage.CompletionDetails == nil || resp.Usage.CompletionDetails.ReasoningTokens != 3 {
		t.Fatalf("completion token details mismatch: %+v", resp.Usage.CompletionDetails)
	}
}

func TestClientChatStreamUsesFullChatPayload(t *testing.T) {
	var payload map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode payload: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"\"}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	temp := 0.7
	maxTokens := 128
	topP := 0.9
	opts := &ChatOptions{
		Model:       "m1",
		Temperature: &temp,
		MaxTokens:   &maxTokens,
		TopP:        &topP,
		Tools: []ToolDefinition{{
			Type: "function",
			Function: FunctionSpec{
				Name:        "weather",
				Description: "Get weather",
				Parameters:  map[string]any{"type": "object"},
			},
		}},
		ToolChoice:     map[string]any{"type": "auto"},
		ResponseFormat: map[string]any{"type": "json_object"},
		User:           "user-a",
		Metadata:       map[string]any{"trace_id": "t-1"},
	}

	c, err := NewClientBuilder().WithBaseURL(srv.URL).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	st, err := c.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, opts)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer st.Close()
	if !st.Next() {
		t.Fatalf("expected streaming event")
	}
	if payload["temperature"] != temp || payload["max_tokens"] != float64(maxTokens) || payload["top_p"] != topP {
		t.Fatalf("missing numeric options in payload: %+v", payload)
	}
	if _, ok := payload["tools"]; !ok {
		t.Fatalf("missing tools in payload: %+v", payload)
	}
	if _, ok := payload["tool_choice"]; !ok {
		t.Fatalf("missing tool_choice in payload: %+v", payload)
	}
	if _, ok := payload["response_format"]; !ok {
		t.Fatalf("missing response_format in payload: %+v", payload)
	}
	if payload["user"] != "user-a" {
		t.Fatalf("missing user in payload: %+v", payload)
	}
	if _, ok := payload["metadata"]; !ok {
		t.Fatalf("missing metadata in payload: %+v", payload)
	}
}

func TestClientEmbeddings(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"model":"m1","data":[{"index":0,"embedding":[0.1,0.2]}]}`))
	}))
	defer srv.Close()

	c, err := NewClientBuilder().WithBaseURL(srv.URL).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	resp, err := c.Embeddings(context.Background(), EmbeddingRequest{Model: "m1", Input: []string{"hello"}})
	if err != nil {
		t.Fatalf("embeddings: %v", err)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("embedding items mismatch: %d", len(resp.Data))
	}
}

func TestMCPCapabilityGuard(t *testing.T) {
	manifest := `
id: "test-provider"
protocol_version: "2.0"
endpoint:
  base_url: "http://127.0.0.1:65535"
`
	c, err := NewClientBuilder().WithProtocolData([]byte(manifest)).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	_, err = c.MCPListTools(context.Background())
	if err == nil {
		t.Fatalf("expected unsupported capability error")
	}
	if !strings.Contains(err.Error(), "capability not declared") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAdvancedCapabilityEndpoints(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/mcp/tools/list":
			_, _ = w.Write([]byte(`{"tools":[{"name":"search"}]}`))
		case "/mcp/tools/call":
			_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}]}`))
		case "/computer-use/actions":
			_, _ = w.Write([]byte(`{"session_id":"s1","results":[{"ok":true}]}`))
		case "/reasoning":
			_, _ = w.Write([]byte(`{"model":"r1","reasoning":"x","final_answer":"done"}`))
		case "/video/generations":
			_, _ = w.Write([]byte(`{"id":"v1","status":"queued"}`))
		case "/video/generations/v1":
			if r.Method != "GET" {
				t.Fatalf("unexpected method for video get: %s", r.Method)
			}
			_, _ = w.Write([]byte(`{"id":"v1","status":"succeeded","url":"https://example.com/v.mp4"}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	manifest := fmt.Sprintf(`
id: "test-provider"
protocol_version: "2.0"
endpoint:
  base_url: "%s"
endpoints:
  mcp_list_tools: { path: "/mcp/tools/list", method: "POST" }
  mcp_call_tool: { path: "/mcp/tools/call", method: "POST" }
  computer_use: { path: "/computer-use/actions", method: "POST" }
  reasoning: { path: "/reasoning", method: "POST" }
  video_generate: { path: "/video/generations", method: "POST" }
  video_get: { path: "/video/generations/{id}", method: "GET" }
capabilities:
  required: [text, mcp, computer_use, reasoning, video]
`, srv.URL)
	c, err := NewClientBuilder().WithProtocolData([]byte(manifest)).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	listResp, err := c.MCPListTools(context.Background())
	if err != nil || len(listResp.Tools) != 1 {
		t.Fatalf("MCPListTools failed: err=%v tools=%v", err, listResp)
	}
	if _, err = c.MCPCallTool(context.Background(), MCPCallToolRequest{Name: "search"}); err != nil {
		t.Fatalf("MCPCallTool failed: %v", err)
	}
	if _, err = c.ComputerUse(context.Background(), ComputerUseRequest{SessionID: "s1", Actions: []ComputerUseAction{{ActionType: "click"}}}); err != nil {
		t.Fatalf("ComputerUse failed: %v", err)
	}
	if _, err = c.Reason(context.Background(), ReasoningRequest{Model: "r1", Prompt: "hi"}); err != nil {
		t.Fatalf("Reason failed: %v", err)
	}
	job, err := c.VideoGenerate(context.Background(), VideoGenerateRequest{Model: "v1", Prompt: "p"})
	if err != nil || job.ID != "v1" {
		t.Fatalf("VideoGenerate failed: err=%v job=%v", err, job)
	}
	job, err = c.VideoGet(context.Background(), "v1")
	if err != nil || job.Status != "succeeded" {
		t.Fatalf("VideoGet failed: err=%v job=%v", err, job)
	}
}

func TestProviderErrorClassificationAndFallbackMatrix(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(429)
		_, _ = io.WriteString(w, `{"error":{"type":"insufficient_quota","message":"quota"}}`)
	}))
	defer srv.Close()
	c, err := NewClientBuilder().WithBaseURL(srv.URL).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	_, err = c.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, &ChatOptions{})
	if err == nil {
		t.Fatalf("expected error")
	}
	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("expected APIError got %T", err)
	}
	if apiErr.Code != ErrQuotaExhausted {
		t.Fatalf("code expected %s got %s", ErrQuotaExhausted, apiErr.Code)
	}
	if IsRetryableCode(apiErr.Code) {
		t.Fatalf("quota exhausted should be non-retryable per compliance baseline")
	}
	if !IsFallbackableCode(apiErr.Code) {
		t.Fatalf("quota exhausted should be fallbackable")
	}
}

func TestRequestPreflightValidation(t *testing.T) {
	manifest := `
id: "test-provider"
protocol_version: "2.0"
endpoint:
  base_url: "http://127.0.0.1:65535"
endpoints:
  chat_completions: { path: "chat/completions", method: "PATCH" }
capabilities:
  required: [text]
`
	c, err := NewClientBuilder().WithProtocolData([]byte(manifest)).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	_, err = c.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, &ChatOptions{})
	if err == nil {
		t.Fatalf("expected preflight validation error")
	}
	if !strings.Contains(err.Error(), "endpoint path must start with /") {
		t.Fatalf("unexpected preflight error: %v", err)
	}

	_, err = c.Chat(context.Background(), nil, &ChatOptions{})
	if err == nil {
		t.Fatalf("expected empty message validation error")
	}
}

func TestManifestDrivenErrorClassification(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		_, _ = io.WriteString(w, `{"error":{"code":"provider_boom","message":"boom"}}`)
	}))
	defer srv.Close()

	manifest := fmt.Sprintf(`
id: "m1"
protocol_version: "2.0"
endpoint:
  base_url: "%s"
error_classification:
  by_error_code:
    provider_boom: overloaded
`, srv.URL)

	c, err := NewClientBuilder().WithProtocolData([]byte(manifest)).Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	_, err = c.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, &ChatOptions{})
	if err == nil {
		t.Fatalf("expected error")
	}
	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("expected APIError got %T", err)
	}
	if apiErr.Code != ErrOverloaded {
		t.Fatalf("expected %s got %s", ErrOverloaded, apiErr.Code)
	}
}
