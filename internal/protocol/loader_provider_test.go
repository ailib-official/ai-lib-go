package protocol

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCanonicalFromIdentityMapResolvesGoogleToGemini(t *testing.T) {
	value := map[string]any{
		"families": []any{
			map[string]any{"canonical_id": "gemini", "aliases": []any{"google"}},
			map[string]any{"canonical_id": "moonshot", "aliases": []any{"kimi"}},
		},
	}
	if got := canonicalFromIdentityValue(value, "google"); got != "gemini" {
		t.Fatalf("google -> %q, want gemini", got)
	}
	if got := canonicalFromIdentityValue(value, "kimi"); got != "moonshot" {
		t.Fatalf("kimi -> %q, want moonshot", got)
	}
	if got := canonicalFromIdentityValue(value, "openai"); got != "" {
		t.Fatalf("openai should not resolve, got %q", got)
	}

	legacy := map[string]any{"canonical_id": "gemini", "aliases": []any{"google"}}
	if got := canonicalFromIdentityValue(legacy, "google"); got != "gemini" {
		t.Fatalf("legacy google -> %q, want gemini", got)
	}
}

func TestLoadProviderPrefersDistAndResolvesAlias(t *testing.T) {
	root := t.TempDir()
	v2Dir := filepath.Join(root, "dist", "v2", "providers")
	if err := os.MkdirAll(v2Dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "dist"), 0o755); err != nil {
		t.Fatal(err)
	}

	gemini := map[string]any{
		"id":               "gemini",
		"aliases":          []string{"google"},
		"protocol_version": "2.0",
		"endpoint":         map[string]any{"base_url": "https://generativelanguage.googleapis.com"},
	}
	writeJSON(t, filepath.Join(v2Dir, "gemini.json"), gemini)
	writeJSON(t, filepath.Join(root, "dist", "provider-identity.json"), map[string]any{
		"families": []any{
			map[string]any{"canonical_id": "gemini", "aliases": []any{"google"}},
		},
	})

	// Also place a source-only openai to prove dist preference path works separately.
	srcDir := filepath.Join(root, "v2", "providers")
	if err := os.MkdirAll(srcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(srcDir, "openai.yaml"), []byte(
		"id: openai\nprotocol_version: \"2.0\"\nendpoint:\n  base_url: https://yaml.example.com\n",
	), 0o644); err != nil {
		t.Fatal(err)
	}

	loader := &Loader{Root: root}
	manifest, err := loader.LoadProvider("google")
	if err != nil {
		t.Fatalf("LoadProvider(google): %v", err)
	}
	v2, ok := manifest.(*V2Manifest)
	if !ok {
		t.Fatalf("expected *V2Manifest, got %T", manifest)
	}
	if v2.ID != "gemini" {
		t.Fatalf("id=%q want gemini", v2.ID)
	}
	if len(v2.Aliases) == 0 || v2.Aliases[0] != "google" {
		t.Fatalf("aliases=%v want [google]", v2.Aliases)
	}

	openai, err := loader.LoadProvider("openai")
	if err != nil {
		t.Fatalf("LoadProvider(openai) source degrade: %v", err)
	}
	ov2 := openai.(*V2Manifest)
	if ov2.Endpoint.BaseURL != "https://yaml.example.com" {
		t.Fatalf("unexpected openai base_url: %s", ov2.Endpoint.BaseURL)
	}
}

func TestLoadProviderConsumesDistGeminiWhenProtocolRootPresent(t *testing.T) {
	root := os.Getenv("AI_PROTOCOL_DIR")
	if root == "" {
		root = "../ai-protocol"
	}
	if _, err := os.Stat(filepath.Join(root, "dist", "v2", "providers", "gemini.json")); err != nil {
		t.Skipf("published dist gemini missing under %s", root)
	}

	loader := &Loader{Root: root}
	manifest, err := loader.LoadProvider("gemini")
	if err != nil {
		t.Fatalf("LoadProvider(gemini): %v", err)
	}
	v2, ok := manifest.(*V2Manifest)
	if !ok {
		t.Fatalf("expected *V2Manifest, got %T", manifest)
	}
	if v2.ID != "gemini" {
		t.Fatalf("id=%q", v2.ID)
	}
	foundGoogle := false
	for _, a := range v2.Aliases {
		if a == "google" {
			foundGoogle = true
			break
		}
	}
	if !foundGoogle {
		t.Fatalf("expected aliases to include google, got %v", v2.Aliases)
	}

	aliased, err := loader.LoadProvider("google")
	if err != nil {
		t.Fatalf("LoadProvider(google) via identity: %v", err)
	}
	if ManifestProviderID(aliased) != "gemini" {
		t.Fatalf("google resolved to %q", ManifestProviderID(aliased))
	}
}

func writeJSON(t *testing.T, path string, value any) {
	t.Helper()
	b, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}
