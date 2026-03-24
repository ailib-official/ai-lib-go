package compliance

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ailib-official/ai-lib-go/pkg/ailib"
)

func TestAdvancedCapabilitiesContract(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/mcp/tools/list":
			_, _ = w.Write([]byte(`{"tools":[{"name":"search"}]}`))
		case "/reasoning":
			_, _ = w.Write([]byte(`{"model":"reasoner","final_answer":"ok"}`))
		case "/video/generations":
			_, _ = w.Write([]byte(`{"id":"vid-1","status":"queued"}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	manifest := fmt.Sprintf(`
id: "go-contract-provider"
protocol_version: "2.0"
endpoint:
  base_url: "%s"
capabilities:
  required: ["text", "mcp", "reasoning", "video"]
`, srv.URL)

	client, err := ailib.NewClientBuilder().WithProtocolData([]byte(manifest)).Build()
	if err != nil {
		t.Fatalf("build client: %v", err)
	}
	defer client.Close()

	if _, err = client.MCPListTools(context.Background()); err != nil {
		t.Fatalf("MCPListTools contract failed: %v", err)
	}
	if _, err = client.Reason(context.Background(), ailib.ReasoningRequest{Model: "reasoner", Prompt: "test"}); err != nil {
		t.Fatalf("Reason contract failed: %v", err)
	}
	if _, err = client.VideoGenerate(context.Background(), ailib.VideoGenerateRequest{Model: "video", Prompt: "test"}); err != nil {
		t.Fatalf("VideoGenerate contract failed: %v", err)
	}
}
