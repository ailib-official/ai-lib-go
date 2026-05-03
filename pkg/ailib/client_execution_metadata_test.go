package ailib

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestClientChatPopulatesExecutionMetadata(t *testing.T) {
	proto := []byte(`id: testprov
protocol_version: "2.0"
endpoint:
  base_url: "http://placeholder"
`)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"r1","model":"m1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	c, err := NewClientBuilder().
		WithProtocolData(proto).
		WithBaseURL(srv.URL).
		WithMaxRetries(1).
		Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	resp, err := c.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, &ChatOptions{Model: "m1"})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if resp.ExecutionMetadata.ProviderID != "testprov" {
		t.Fatalf("provider_id: got %q", resp.ExecutionMetadata.ProviderID)
	}
	if resp.ExecutionMetadata.ModelID != "m1" {
		t.Fatalf("model_id: got %q", resp.ExecutionMetadata.ModelID)
	}
	if resp.ExecutionMetadata.MicroRetryCount != 0 {
		t.Fatalf("micro_retry_count: got %d", resp.ExecutionMetadata.MicroRetryCount)
	}
	if resp.ExecutionMetadata.ExecutionLatencyMs == 0 {
		t.Fatalf("execution_latency_ms should be non-zero")
	}
}

func TestClientChatMicroRetryCountAfterTransientFailure(t *testing.T) {
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&hits, 1)
		if n == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"type":"overloaded"}}`)
			return
		}
		_, _ = w.Write([]byte(`{"id":"r1","model":"m1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	c, err := NewClientBuilder().
		WithBaseURL(srv.URL).
		WithMaxRetries(3).
		Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	resp, err := c.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, &ChatOptions{Model: "m1"})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if resp.ExecutionMetadata.MicroRetryCount < 1 {
		t.Fatalf("expected at least one micro-retry, got %d (hits=%d)", resp.ExecutionMetadata.MicroRetryCount, hits)
	}
	if atomic.LoadInt32(&hits) < 2 {
		t.Fatalf("expected server hit at least twice, got %d", hits)
	}
}

func TestClientChatStreamExecutionMetadataAfterClose(t *testing.T) {
	proto := []byte(`id: streamprov
protocol_version: "2.0"
endpoint:
  base_url: "http://placeholder"
`)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c, err := NewClientBuilder().
		WithProtocolData(proto).
		WithBaseURL(srv.URL).
		Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer c.Close()

	st, err := c.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, &ChatOptions{Model: "m-stream"})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	for st.Next() {
	}
	if st.Err() != nil {
		t.Fatalf("stream err: %v", st.Err())
	}
	_ = st.Close()

	meta, ok := st.ExecutionMetadata()
	if !ok {
		t.Fatalf("expected execution metadata after Close")
	}
	if meta.ProviderID != "streamprov" || meta.ModelID != "m-stream" {
		t.Fatalf("metadata: %+v", meta)
	}
}

func TestChatResponseJSONIncludesExecutionMetadataShape(t *testing.T) {
	resp := ChatResponse{
		ID:    "r",
		Model: "m",
		Choices: []Choice{
			{Index: 0, Message: Message{Role: RoleAssistant, Content: "hi"}},
		},
		ExecutionMetadata: ExecutionMetadata{
			ProviderID:           "p",
			ModelID:              "m",
			ExecutionLatencyMs:   5,
			TranslationLatencyMs: 1,
			MicroRetryCount:      0,
		},
	}
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatal(err)
	}
	if _, ok := raw["execution_metadata"]; !ok {
		t.Fatalf("missing execution_metadata in %s", string(b))
	}
}
