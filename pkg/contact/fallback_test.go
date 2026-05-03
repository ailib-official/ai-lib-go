package contact_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ailib-official/ai-lib-go/pkg/ailib"
	"github.com/ailib-official/ai-lib-go/pkg/contact"
)

func TestFallbackClientChat(t *testing.T) {
	failSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(429)
		_, _ = io.WriteString(w, `{"error":{"type":"insufficient_quota","message":"quota"}}`)
	}))
	defer failSrv.Close()

	okSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"id":"r2","model":"m2","choices":[{"index":0,"message":{"role":"assistant","content":"ok2"}}]}`)
	}))
	defer okSrv.Close()

	c1, err := ailib.NewClientBuilder().WithBaseURL(failSrv.URL).Build()
	if err != nil {
		t.Fatalf("build c1: %v", err)
	}
	defer c1.Close()
	c2, err := ailib.NewClientBuilder().WithBaseURL(okSrv.URL).Build()
	if err != nil {
		t.Fatalf("build c2: %v", err)
	}
	defer c2.Close()

	fc := contact.NewFallbackClient(c1, c2)
	resp, err := fc.Chat(context.Background(), []ailib.Message{{Role: ailib.RoleUser, Content: "hello"}}, &ailib.ChatOptions{})
	if err != nil {
		t.Fatalf("fallback chat failed: %v", err)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Message.Content != "ok2" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestFallbackClientChatStream(t *testing.T) {
	failSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(503)
		_, _ = io.WriteString(w, `{"error":{"type":"overloaded"}}`)
	}))
	defer failSrv.Close()

	okSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer okSrv.Close()

	c1, err := ailib.NewClientBuilder().WithBaseURL(failSrv.URL).Build()
	if err != nil {
		t.Fatalf("build c1: %v", err)
	}
	defer c1.Close()
	c2, err := ailib.NewClientBuilder().WithBaseURL(okSrv.URL).Build()
	if err != nil {
		t.Fatalf("build c2: %v", err)
	}
	defer c2.Close()

	fc := contact.NewFallbackClient(c1, c2)
	st, err := fc.ChatStream(context.Background(), []ailib.Message{{Role: ailib.RoleUser, Content: "hello"}}, &ailib.ChatOptions{})
	if err != nil {
		t.Fatalf("fallback stream failed: %v", err)
	}
	defer st.Close()
	if !st.Next() {
		t.Fatalf("expected streaming event")
	}
	if st.Event().Delta != "ok" {
		t.Fatalf("delta mismatch: %s", st.Event().Delta)
	}
}

func TestFallbackClientCircuitBreaker(t *testing.T) {
	var primaryHits int32
	var secondaryHits int32

	primarySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&primaryHits, 1)
		w.WriteHeader(503)
		_, _ = io.WriteString(w, `{"error":{"type":"overloaded"}}`)
	}))
	defer primarySrv.Close()

	secondarySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&secondaryHits, 1)
		_, _ = io.WriteString(w, `{"id":"r2","model":"m2","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer secondarySrv.Close()

	c1, err := ailib.NewClientBuilder().WithBaseURL(primarySrv.URL).Build()
	if err != nil {
		t.Fatalf("build c1: %v", err)
	}
	defer c1.Close()
	c2, err := ailib.NewClientBuilder().WithBaseURL(secondarySrv.URL).Build()
	if err != nil {
		t.Fatalf("build c2: %v", err)
	}
	defer c2.Close()

	fc := contact.NewFallbackClientWithPolicy(contact.FallbackPolicy{
		FailureThreshold: 1,
		CircuitOpenFor:   time.Hour,
	}, c1, c2)

	_, err = fc.Chat(context.Background(), []ailib.Message{{Role: ailib.RoleUser, Content: "hello"}}, &ailib.ChatOptions{})
	if err != nil {
		t.Fatalf("first call failed: %v", err)
	}
	primaryHitsAfterFirst := atomic.LoadInt32(&primaryHits)
	_, err = fc.Chat(context.Background(), []ailib.Message{{Role: ailib.RoleUser, Content: "hello2"}}, &ailib.ChatOptions{})
	if err != nil {
		t.Fatalf("second call failed: %v", err)
	}

	if got := atomic.LoadInt32(&primaryHits); got != primaryHitsAfterFirst {
		t.Fatalf("expected primary to be skipped after circuit open, before=%d after=%d", primaryHitsAfterFirst, got)
	}
	if got := atomic.LoadInt32(&secondaryHits); got != 2 {
		t.Fatalf("expected secondary to serve two calls, got hits=%d", got)
	}
	snapshot := fc.HealthSnapshot()
	if len(snapshot) == 0 || !snapshot[0].CircuitOpen {
		t.Fatalf("expected primary circuit open in health snapshot: %+v", snapshot)
	}
}
