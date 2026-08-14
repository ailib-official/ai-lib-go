package protocol

import (
	"strings"
	"testing"
)

func TestLoaderParsesCurrentV2Shape(t *testing.T) {
	loader := NewLoader()
	manifestYAML := `
id: openai
protocol_version: "2.0"
endpoint:
  base_url: "https://api.openai.com/v1"
  chat: "/chat/completions"
capabilities:
  required: ["text", "streaming", "tools"]
  optional: ["mcp_client"]
capability_profile:
  phase: "ios_v1"
  inputs:
    modalities: ["text"]
`
	manifest, err := loader.LoadBytes([]byte(manifestYAML), ".yaml")
	if err != nil {
		t.Fatalf("expected valid v2 manifest, got error: %v", err)
	}
	v2, ok := manifest.(*V2Manifest)
	if !ok {
		t.Fatalf("expected *V2Manifest got %T", manifest)
	}
	if v2.ID != "openai" {
		t.Fatalf("unexpected id: %s", v2.ID)
	}
	if v2.Endpoint.BaseURL != "https://api.openai.com/v1" {
		t.Fatalf("unexpected base_url: %s", v2.Endpoint.BaseURL)
	}
	if !HasCapability(v2, "chat") {
		t.Fatalf("chat capability should be available via required text")
	}
	if !HasCapability(v2, "mcp") {
		t.Fatalf("mcp capability should be available via optional mcp_client")
	}
}

func TestLoaderRejectsIOSWithProcessContract(t *testing.T) {
	loader := NewLoader()
	manifestYAML := `
id: google_ios_invalid
protocol_version: "2.0"
endpoint:
  base_url: "https://example.com"
capabilities:
  required: ["text"]
capability_profile:
  phase: "ios_v1"
  inputs:
    modalities: ["text"]
  process:
    mode: "async"
`
	_, err := loader.LoadBytes([]byte(manifestYAML), ".yaml")
	if err == nil {
		t.Fatalf("expected ios_v1 process boundary validation error")
	}
}

func TestLoaderRejectsIOSPCWithoutProcessContract(t *testing.T) {
	loader := NewLoader()
	manifestYAML := `
id: google_iospc_invalid
protocol_version: "2.0"
endpoint:
  base_url: "https://example.com"
capabilities:
  required: ["text"]
capability_profile:
  phase: "iospc_v1"
  inputs:
    modalities: ["text"]
`
	_, err := loader.LoadBytes([]byte(manifestYAML), ".yaml")
	if err == nil {
		t.Fatalf("expected iospc_v1 missing process/contract validation error")
	}
}

func TestIsJSONDetectsPathExtension(t *testing.T) {
	if !isJSON("provider.json", []byte("not-json")) {
		t.Fatalf("expected .json path to be treated as json")
	}
	if isJSON("provider.yaml", []byte("key: value")) {
		t.Fatalf("yaml path should not be treated as json")
	}
}

// TestEndpointForEmbeddingsRerankXREmb asserts XR-EMB contract:
// EndpointFor returns path-only fallbacks, never vendor hosts.
func TestEndpointForEmbeddingsRerankXREmb(t *testing.T) {
	empty := &V2Manifest{ID: "mock", ProtocolVersion: "2.0"}
	embPath, embMethod := EndpointFor(empty, "embeddings", "/embeddings")
	if embPath != "/embeddings" || embMethod != "POST" {
		t.Fatalf("embeddings fallback: got %s %s", embPath, embMethod)
	}
	rerankPath, rerankMethod := EndpointFor(empty, "rerank", "/rerank")
	if rerankPath != "/rerank" || rerankMethod != "POST" {
		t.Fatalf("rerank fallback: got %s %s", rerankPath, rerankMethod)
	}

	withMap := &V2Manifest{
		ID:              "mock",
		ProtocolVersion: "2.0",
		Endpoints: map[string]any{
			"embeddings": map[string]any{"path": "/custom/embed"},
			"rerank":     map[string]any{"path": "/custom/rerank"},
		},
	}
	p, _ := EndpointFor(withMap, "embeddings", "/embeddings")
	if p != "/custom/embed" {
		t.Fatalf("embeddings path from map: got %s", p)
	}
	p, _ = EndpointFor(withMap, "rerank", "/rerank")
	if p != "/custom/rerank" {
		t.Fatalf("rerank path from map: got %s", p)
	}

	for _, path := range []string{embPath, rerankPath, p} {
		if strings.Contains(path, "api.openai.com") || strings.Contains(path, "api.cohere.com") {
			t.Fatalf("vendor host leaked into path: %s", path)
		}
	}
}

func TestSupportsGenerativeForModelOmitFailClosed(t *testing.T) {
	loader := NewLoader()
	manifestYAML := `
id: openai
protocol_version: "2.0"
endpoint:
  base_url: "https://api.openai.com/v1"
capabilities:
  required: ["text"]
  optional: ["image_generation"]
metadata:
  models:
    gpt-image-1:
      model_capabilities:
        image_generation: true
    gpt-4o:
      context_window: 128000
`
	manifest, err := loader.LoadBytes([]byte(manifestYAML), ".yaml")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !SupportsGenerativeForModel(manifest, "gpt-image-1", KeyImageGeneration) {
		t.Fatal("declared image_generation should be true")
	}
	if SupportsGenerativeForModel(manifest, "gpt-4o", KeyImageGeneration) {
		t.Fatal("omitted image_generation must not be true")
	}
	if SupportsGenerativeForModel(manifest, "missing", KeyImageGeneration) {
		t.Fatal("missing model must not be true")
	}
}

func TestRequireGenerativeEndpointFailClosedAndAdapters(t *testing.T) {
	loader := NewLoader()
	openaiYAML := `
id: openai
protocol_version: "2.0"
endpoint:
  base_url: "https://api.openai.com/v1"
endpoints:
  image_generation:
    path: "/images/generations"
    method: POST
    adapter: openai
  speech_to_text:
    path: "/audio/transcriptions"
    adapter: openai
metadata:
  models:
    gpt-image-1:
      model_capabilities:
        image_generation: true
    whisper-1:
      model_capabilities:
        speech_to_text: true
    gpt-4o:
      context_window: 128000
`
	openai, err := loader.LoadBytes([]byte(openaiYAML), ".yaml")
	if err != nil {
		t.Fatalf("load openai: %v", err)
	}
	if _, err := RequireGenerativeEndpoint(openai, "gpt-4o", KeyImageGeneration); err == nil {
		t.Fatal("omit must fail-closed")
	} else if !strings.Contains(err.Error(), "omit") {
		t.Fatalf("expected omit in error, got %v", err)
	}
	ep, err := RequireGenerativeEndpoint(openai, "gpt-image-1", KeyImageGeneration)
	if err != nil {
		t.Fatalf("openai image: %v", err)
	}
	if ep.Path != "/images/generations" || ep.Adapter != "openai" {
		t.Fatalf("unexpected ep: %+v", ep)
	}

	missingYAML := `
id: genprov
protocol_version: "2.0"
endpoint:
  base_url: "https://example.com/v1"
metadata:
  models:
    img-1:
      model_capabilities:
        image_generation: true
`
	missing, err := loader.LoadBytes([]byte(missingYAML), ".yaml")
	if err != nil {
		t.Fatalf("load missing: %v", err)
	}
	if _, err := RequireGenerativeEndpoint(missing, "img-1", KeyImageGeneration); err == nil {
		t.Fatal("missing L-Exec must fail")
	} else if !strings.Contains(err.Error(), "endpoints.image_generation") {
		t.Fatalf("expected endpoints.image_generation, got %v", err)
	}

	qwenYAML := `
id: qwen
protocol_version: "2.0"
endpoint:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
endpoints:
  image_generation:
    path: "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    method: POST
    adapter: dashscope
metadata:
  models:
    qwen-image-plus:
      model_capabilities:
        image_generation: true
`
	qwen, err := loader.LoadBytes([]byte(qwenYAML), ".yaml")
	if err != nil {
		t.Fatalf("load qwen: %v", err)
	}
	qep, err := RequireGenerativeEndpoint(qwen, "qwen-image-plus", KeyImageGeneration)
	if err != nil {
		t.Fatalf("qwen: %v", err)
	}
	if !strings.HasPrefix(qep.Path, "https://") || qep.Adapter != "dashscope" {
		t.Fatalf("unexpected qwen ep: %+v", qep)
	}
}

