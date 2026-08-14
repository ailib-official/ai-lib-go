// Generative L-Exec endpoint resolution (ALG-GEN-002 / PT-GEN-002).
// 生成式 endpoints.<key> 解析：omit≠false 门控 + adapter（缺省 openai）。
package protocol

import (
	"fmt"
	"strings"
)

// GenerativeEndpoint is a resolved L-Exec map entry.
type GenerativeEndpoint struct {
	Path    string
	Method  string
	Adapter string
}

// AdapterName returns L-Exec adapter; empty ⇒ openai (ALR-GEN-002).
func AdapterName(adapter string) string {
	if strings.TrimSpace(adapter) == "" {
		return "openai"
	}
	return strings.TrimSpace(adapter)
}

func endpointsMap(m any) map[string]any {
	switch v := m.(type) {
	case *V1Manifest:
		return v.Endpoints
	case *V2Manifest:
		if len(v.Endpoints) > 0 {
			return v.Endpoints
		}
		if len(v.Endpoint.Endpoints) > 0 {
			return v.Endpoint.Endpoints
		}
		if v.Core != nil && len(v.Core.Endpoint.Endpoints) > 0 {
			return v.Core.Endpoint.Endpoints
		}
		return nil
	default:
		return nil
	}
}

func endpointEntryFromMap(endpoints map[string]any, key string) (GenerativeEndpoint, bool) {
	if len(endpoints) == 0 {
		return GenerativeEndpoint{}, false
	}
	raw, ok := endpoints[key]
	if !ok {
		return GenerativeEndpoint{}, false
	}
	switch v := raw.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return GenerativeEndpoint{}, false
		}
		return GenerativeEndpoint{Path: v, Method: "POST", Adapter: "openai"}, true
	case map[string]any:
		path, _ := v["path"].(string)
		if strings.TrimSpace(path) == "" {
			return GenerativeEndpoint{}, false
		}
		method, _ := v["method"].(string)
		adapter, _ := v["adapter"].(string)
		return GenerativeEndpoint{
			Path:    path,
			Method:  upperOrPOST(method),
			Adapter: AdapterName(adapter),
		}, true
	default:
		return GenerativeEndpoint{}, false
	}
}

// ResolveGenerativeEndpoint returns endpoints.<key> without capability gate.
func ResolveGenerativeEndpoint(m any, key string) (GenerativeEndpoint, error) {
	switch key {
	case KeyImageGeneration, KeySpeechToText, KeyTextToSpeech:
	default:
		return GenerativeEndpoint{}, fmt.Errorf(
			"unknown generative capability %q; expected one of %s, %s, %s",
			key, KeyImageGeneration, KeySpeechToText, KeyTextToSpeech,
		)
	}
	ep, ok := endpointEntryFromMap(endpointsMap(m), key)
	if !ok {
		return GenerativeEndpoint{}, fmt.Errorf(
			"manifest endpoints.%s missing; declare PT-GEN-002 L-Exec map", key,
		)
	}
	return ep, nil
}

// RequireGenerativeEndpoint gates model_capabilities (omit≠false) then resolves L-Exec.
func RequireGenerativeEndpoint(m any, modelID, key string) (GenerativeEndpoint, error) {
	if !SupportsGenerativeForModel(m, modelID, key) {
		return GenerativeEndpoint{}, fmt.Errorf(
			"model %q does not declare model_capabilities.%s=true (omit≠false fail-closed)",
			modelID, key,
		)
	}
	return ResolveGenerativeEndpoint(m, key)
}
