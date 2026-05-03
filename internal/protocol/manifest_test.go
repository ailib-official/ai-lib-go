package protocol

import "testing"

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
