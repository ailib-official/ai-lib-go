// Package protocol loads provider manifests.
// 协议加载器，支持本地文件与内存数据；优先消费 dist/，并经 identity map 解析别名。
package protocol

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

type Loader struct {
	// Root is an optional protocol repository root (overrides AI_PROTOCOL_DIR).
	Root string
}

func NewLoader() *Loader {
	return &Loader{}
}

func (l *Loader) LoadFile(path string) (any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return l.LoadBytes(b, path)
}

func (l *Loader) LoadBytes(data []byte, source string) (any, error) {
	// Parse lightweight meta first for version detection.
	meta := map[string]any{}
	if isJSON(source, data) {
		if err := json.Unmarshal(data, &meta); err != nil {
			return nil, fmt.Errorf("decode manifest metadata: %w", err)
		}
	} else {
		if err := yaml.Unmarshal(data, &meta); err != nil {
			return nil, fmt.Errorf("decode manifest metadata: %w", err)
		}
	}

	version, _ := meta["protocol_version"].(string)
	if strings.HasPrefix(version, "2") || hasCore(meta) || hasCapabilityProfile(meta) {
		var out V2Manifest
		if err := unmarshalBySource(data, source, &out); err != nil {
			return nil, err
		}
		baseURL := out.Endpoint.BaseURL
		if baseURL == "" && out.Core != nil {
			baseURL = out.Core.Endpoint.BaseURL
		}
		if out.ID == "" || baseURL == "" {
			return nil, fmt.Errorf("invalid v2 manifest: missing id or base_url")
		}
		if err := ValidateCapabilityProfile(out.CapabilityProfile); err != nil {
			return nil, fmt.Errorf("invalid capability_profile: %w", err)
		}
		return &out, nil
	}

	var out V1Manifest
	if err := unmarshalBySource(data, source, &out); err != nil {
		return nil, err
	}
	baseURL := out.BaseURL
	if baseURL == "" {
		baseURL = out.Endpoint.BaseURL
	}
	if out.ID == "" || baseURL == "" {
		return nil, fmt.Errorf("invalid v1 manifest: missing id or base_url")
	}
	return &out, nil
}

// LoadProvider loads by provider id.
//
// Resolution (PT-ARCH-005 / GO-ID-001):
//  1. Exact match via published dist/ (then source YAML degrade)
//  2. If missing: alias → canonical via dist/provider-identity.json
//  3. Else fail closed
//
// Parse/validation errors are never masked by alias fallthrough.
func (l *Loader) LoadProvider(providerID string) (any, error) {
	manifest, err := l.loadProviderExact(providerID)
	if err == nil {
		return manifest, nil
	}
	if !os.IsNotExist(err) && !isNotFound(err) {
		return nil, err
	}

	canonical := l.resolveCanonicalProviderID(providerID)
	if canonical != "" && canonical != providerID {
		manifest, err = l.loadProviderExact(canonical)
		if err == nil {
			return manifest, nil
		}
		if !os.IsNotExist(err) && !isNotFound(err) {
			return nil, err
		}
	}

	return nil, fmt.Errorf("provider %q not found (checked dist/ then source; alias map if present)", providerID)
}

func isNotFound(err error) bool {
	return err != nil && strings.Contains(err.Error(), "not found")
}

func (l *Loader) loadProviderExact(providerID string) (any, error) {
	path := l.providerPath(providerID)
	if path == "" {
		return nil, fmt.Errorf("provider %q not found", providerID)
	}
	return l.LoadFile(path)
}

func (l *Loader) protocolRoot() string {
	if l.Root != "" {
		return l.Root
	}
	if env := os.Getenv("AI_PROTOCOL_DIR"); env != "" {
		return env
	}
	if env := os.Getenv("AI_PROTOCOL_PATH"); env != "" {
		return env
	}
	for _, candidate := range []string{"ai-protocol", "../ai-protocol", "../../ai-protocol"} {
		if st, err := os.Stat(candidate); err == nil && st.IsDir() {
			abs, err := filepath.Abs(candidate)
			if err == nil {
				return abs
			}
			return candidate
		}
	}
	return ""
}

// providerPath prefers published dist/ JSON; degrades to source YAML/JSON.
func (l *Loader) providerPath(providerID string) string {
	root := l.protocolRoot()
	if root == "" {
		return ""
	}
	candidates := []string{
		filepath.Join(root, "dist", "v2", "providers", providerID+".json"),
		filepath.Join(root, "dist", "v1", "providers", providerID+".json"),
		filepath.Join(root, "v2", "providers", providerID+".json"),
		filepath.Join(root, "v2", "providers", providerID+".yaml"),
		filepath.Join(root, "v1", "providers", providerID+".json"),
		filepath.Join(root, "v1", "providers", providerID+".yaml"),
	}
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

func (l *Loader) identityMapCandidates() []string {
	root := l.protocolRoot()
	if root == "" {
		return nil
	}
	return []string{
		filepath.Join(root, "dist", "provider-identity.json"),
		filepath.Join(root, "v2", "provider-identity.fixture.json"),
	}
}

func (l *Loader) resolveCanonicalProviderID(key string) string {
	for _, mapPath := range l.identityMapCandidates() {
		b, err := os.ReadFile(mapPath)
		if err != nil {
			continue
		}
		var value any
		if err := json.Unmarshal(b, &value); err != nil {
			continue
		}
		if canonical := canonicalFromIdentityValue(value, key); canonical != "" {
			return canonical
		}
	}
	return ""
}

func canonicalFromIdentityValue(value any, key string) string {
	obj, ok := value.(map[string]any)
	if !ok {
		return ""
	}
	if families, ok := obj["families"].([]any); ok {
		for _, family := range families {
			if fm, ok := family.(map[string]any); ok {
				if canonical := canonicalFromFamily(fm, key); canonical != "" {
					return canonical
				}
			}
		}
		return ""
	}
	return canonicalFromFamily(obj, key)
}

func canonicalFromFamily(family map[string]any, key string) string {
	canonical, _ := family["canonical_id"].(string)
	if canonical == "" {
		return ""
	}
	if key == canonical {
		return canonical
	}
	aliases, _ := family["aliases"].([]any)
	for _, alias := range aliases {
		if s, ok := alias.(string); ok && s == key {
			return canonical
		}
	}
	return ""
}

func hasCore(meta map[string]any) bool {
	_, ok := meta["core"]
	return ok
}

func hasCapabilityProfile(meta map[string]any) bool {
	_, ok := meta["capability_profile"]
	return ok
}

func isJSON(source string, data []byte) bool {
	trimmed := strings.TrimSpace(string(data))
	if strings.EqualFold(filepath.Ext(source), ".json") {
		return true
	}
	return strings.HasPrefix(trimmed, "{")
}

func unmarshalBySource(data []byte, source string, out any) error {
	if isJSON(source, data) {
		if err := json.Unmarshal(data, out); err != nil {
			return fmt.Errorf("decode json manifest: %w", err)
		}
		return nil
	}
	if err := yaml.Unmarshal(data, out); err != nil {
		return fmt.Errorf("decode yaml manifest: %w", err)
	}
	return nil
}
