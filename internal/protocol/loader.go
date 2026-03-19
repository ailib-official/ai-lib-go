// Package protocol loads provider manifests.
// 协议加载器，支持本地文件与内存数据。
package protocol

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type Loader struct{}

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
	if strings.HasPrefix(source, ".json") {
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
