// Package protocol defines protocol manifest contracts.
// 协议清单模型，统一 V1/V2 读取与端点解析。
package protocol

import (
	"fmt"
	"strings"
)

type V1Manifest struct {
	ID              string               `yaml:"id" json:"id"`
	ProtocolVersion string               `yaml:"protocol_version" json:"protocol_version"`
	BaseURL         string               `yaml:"base_url" json:"base_url"`
	APIFormat       string               `yaml:"api_format" json:"api_format"`
	Capabilities    []string             `yaml:"capabilities" json:"capabilities"`
	ErrorClass      ErrorClass           `yaml:"error_classification" json:"error_classification"`
	RetryPolicy     RetryPolicy          `yaml:"retry_policy" json:"retry_policy"`
	Auth            *V1Auth              `yaml:"auth" json:"auth"`
	Endpoint        EndpointConfig       `yaml:"endpoint" json:"endpoint"`
	Endpoints       map[string]any       `yaml:"endpoints" json:"endpoints"`
	Streaming       *StreamingConfig     `yaml:"streaming" json:"streaming"`
	ResponsePaths   *ResponsePathsConfig `yaml:"response_paths" json:"response_paths"`
}

type StreamingConfig struct {
	Decoder *DecoderConfig `yaml:"decoder" json:"decoder"`
}

type DecoderConfig struct {
	Format   string `yaml:"format" json:"format"`
	Strategy string `yaml:"strategy" json:"strategy"`
}

type V1Auth struct {
	Type   string `yaml:"type" json:"type"`
	Header string `yaml:"header" json:"header"`
	Prefix string `yaml:"prefix" json:"prefix"`
}

type EndpointConfig struct {
	BaseURL    string         `yaml:"base_url" json:"base_url"`
	Chat       string         `yaml:"chat" json:"chat"`
	Embeddings string         `yaml:"embeddings" json:"embeddings"`
	STT        string         `yaml:"stt" json:"stt"`
	TTS        string         `yaml:"tts" json:"tts"`
	Auth       *V2Auth        `yaml:"auth" json:"auth"`
	Protocol   string         `yaml:"protocol" json:"protocol"`
	TimeoutMS  int            `yaml:"timeout_ms" json:"timeout_ms"`
	Endpoints  map[string]any `yaml:"endpoints" json:"endpoints"`
}

type V2Manifest struct {
	ID                string               `yaml:"id" json:"id"`
	ProtocolVersion   string               `yaml:"protocol_version" json:"protocol_version"`
	Endpoint          EndpointConfig       `yaml:"endpoint" json:"endpoint"`
	Endpoints         map[string]any       `yaml:"endpoints" json:"endpoints"`
	ErrorClass        ErrorClass           `yaml:"error_classification" json:"error_classification"`
	RetryPolicy       RetryPolicy          `yaml:"retry_policy" json:"retry_policy"`
	Capabilities      V2Caps               `yaml:"capabilities" json:"capabilities"`
	CapabilityProfile *CapabilityProfile   `yaml:"capability_profile" json:"capability_profile"`
	Streaming         *StreamingConfig     `yaml:"streaming" json:"streaming"`
	ResponsePaths     *ResponsePathsConfig `yaml:"response_paths" json:"response_paths"`
	// Backward-compatible alias for old local fixtures.
	Core *V2CoreLegacy `yaml:"core" json:"core"`
}

type V2CoreLegacy struct {
	Endpoint EndpointConfig `yaml:"endpoint" json:"endpoint"`
	Auth     V2Auth         `yaml:"auth" json:"auth"`
}

type V2Auth struct {
	Type     string `yaml:"type" json:"type"`
	Header   string `yaml:"header" json:"header"`
	Key      string `yaml:"key" json:"key"`
	Prefix   string `yaml:"prefix" json:"prefix"`
	TokenEnv string `yaml:"token_env" json:"token_env"`
}

type V2Caps struct {
	Required     []string        `yaml:"required" json:"required"`
	Optional     []string        `yaml:"optional" json:"optional"`
	FeatureFlags map[string]bool `yaml:"feature_flags" json:"feature_flags"`
}

type CapabilityProfile struct {
	Phase    string         `yaml:"phase" json:"phase"`
	Inputs   map[string]any `yaml:"inputs" json:"inputs"`
	Outcomes map[string]any `yaml:"outcomes" json:"outcomes"`
	Systems  map[string]any `yaml:"systems" json:"systems"`
	Process  map[string]any `yaml:"process" json:"process"`
	Contract map[string]any `yaml:"contract" json:"contract"`
}

type ErrorClass struct {
	ByHTTPStatus map[string]string `yaml:"by_http_status" json:"by_http_status"`
	ByErrorCode  map[string]string `yaml:"by_error_code" json:"by_error_code"`
	ByErrorType  map[string]string `yaml:"by_error_type" json:"by_error_type"`
}

type RetryPolicy struct {
	MaxRetries int `yaml:"max_retries" json:"max_retries"`
}

func BaseURL(m any) (string, error) {
	switch v := m.(type) {
	case *V1Manifest:
		if v.Endpoint.BaseURL != "" {
			return v.Endpoint.BaseURL, nil
		}
		return v.BaseURL, nil
	case *V2Manifest:
		if v.Endpoint.BaseURL != "" {
			return v.Endpoint.BaseURL, nil
		}
		if v.Core != nil {
			return v.Core.Endpoint.BaseURL, nil
		}
		return "", nil
	default:
		return "", fmt.Errorf("unsupported manifest type: %T", m)
	}
}

func AuthHeader(m any) (name string, valuePrefix string, err error) {
	switch v := m.(type) {
	case *V1Manifest:
		h := "Authorization"
		p := ""
		if v.Endpoint.Auth != nil {
			if v.Endpoint.Auth.Header != "" {
				h = v.Endpoint.Auth.Header
			} else if v.Endpoint.Auth.Key != "" {
				h = v.Endpoint.Auth.Key
			}
			p = v.Endpoint.Auth.Prefix
			if p == "" && v.Endpoint.Auth.Type == "bearer" {
				p = "Bearer "
			}
		}
		if v.Auth != nil {
			if v.Auth.Header != "" {
				h = v.Auth.Header
			}
			if p == "" {
				p = v.Auth.Prefix
			}
			if p == "" && v.Auth.Type == "bearer" {
				p = "Bearer "
			}
		}
		return h, p, nil
	case *V2Manifest:
		auth := v.Endpoint.Auth
		if auth == nil && v.Core != nil {
			auth = &v.Core.Auth
		}
		h := "Authorization"
		if auth != nil {
			if auth.Header != "" {
				h = auth.Header
			} else if auth.Key != "" {
				h = auth.Key
			}
		}
		p := ""
		if auth != nil {
			p = auth.Prefix
			if p == "" && auth.Type == "bearer" {
				p = "Bearer "
			}
		}
		return h, p, nil
	default:
		return "", "", fmt.Errorf("unsupported manifest type: %T", m)
	}
}

func EndpointFor(m any, key string, fallback string) (path string, method string) {
	switch v := m.(type) {
	case *V1Manifest:
		if p, mth, ok := endpointFromMap(v.Endpoints, key); ok {
			return p, mth
		}
		switch key {
		case "chat_completions":
			if v.Endpoint.Chat != "" {
				return v.Endpoint.Chat, "POST"
			}
		case "embeddings":
			if v.Endpoint.Embeddings != "" {
				return v.Endpoint.Embeddings, "POST"
			}
		case "audio_transcriptions":
			if v.Endpoint.STT != "" {
				return v.Endpoint.STT, "POST"
			}
		case "audio_speech":
			if v.Endpoint.TTS != "" {
				return v.Endpoint.TTS, "POST"
			}
		}
	case *V2Manifest:
		if p, mth, ok := endpointFromMap(v.Endpoints, key); ok {
			return p, mth
		}
		if p, mth, ok := endpointFromMap(v.Endpoint.Endpoints, key); ok {
			return p, mth
		}
		if v.Core != nil {
			if p, mth, ok := endpointFromMap(v.Core.Endpoint.Endpoints, key); ok {
				return p, mth
			}
		}
		switch key {
		case "chat_completions":
			if v.Endpoint.Chat != "" {
				return v.Endpoint.Chat, "POST"
			}
		case "embeddings":
			if v.Endpoint.Embeddings != "" {
				return v.Endpoint.Embeddings, "POST"
			}
		case "audio_transcriptions":
			if v.Endpoint.STT != "" {
				return v.Endpoint.STT, "POST"
			}
		case "audio_speech":
			if v.Endpoint.TTS != "" {
				return v.Endpoint.TTS, "POST"
			}
		}
	}
	return fallback, "POST"
}

func HasCapability(m any, name string) bool {
	switch v := m.(type) {
	case *V1Manifest:
		target := normalizeCapabilityName(name)
		for _, c := range v.Capabilities {
			if normalizeCapabilityName(c) == target {
				return true
			}
		}
		if target == "chat" {
			for _, c := range v.Capabilities {
				if normalizeCapabilityName(c) == "text" {
					return true
				}
			}
		}
		return false
	case *V2Manifest:
		target := normalizeCapabilityName(name)
		has := func(values []string, capName string) bool {
			for _, value := range values {
				if normalizeCapabilityName(value) == capName {
					return true
				}
			}
			return false
		}
		if has(v.Capabilities.Required, target) || has(v.Capabilities.Optional, target) {
			return true
		}
		switch target {
		case "chat":
			return has(v.Capabilities.Required, "text") || has(v.Capabilities.Optional, "text")
		case "mcp":
			return has(v.Capabilities.Required, "mcp_client") || has(v.Capabilities.Optional, "mcp_client")
		default:
			return false
		}
	default:
		return true
	}
}

func endpointFromMap(endpoints map[string]any, key string) (string, string, bool) {
	if len(endpoints) == 0 {
		return "", "", false
	}
	raw, ok := endpoints[key]
	if !ok {
		return "", "", false
	}
	switch v := raw.(type) {
	case string:
		if v == "" {
			return "", "", false
		}
		return v, "POST", true
	case map[string]any:
		path, _ := v["path"].(string)
		method, _ := v["method"].(string)
		if path == "" {
			return "", "", false
		}
		return path, upperOrPOST(method), true
	default:
		return "", "", false
	}
}

func normalizeCapabilityName(name string) string {
	key := strings.ToLower(strings.TrimSpace(name))
	switch key {
	case "chat_completions", "text_completion":
		return "chat"
	case "rerank":
		return "reranking"
	default:
		return key
	}
}

func ValidateCapabilityProfile(profile *CapabilityProfile) error {
	if profile == nil {
		return nil
	}
	phase := strings.TrimSpace(profile.Phase)
	if phase == "" {
		return fmt.Errorf("capability_profile.phase is required")
	}
	hasIOS := len(profile.Inputs) > 0 || len(profile.Outcomes) > 0 || len(profile.Systems) > 0
	switch phase {
	case "ios_v1":
		if len(profile.Process) > 0 || len(profile.Contract) > 0 {
			return fmt.Errorf("ios_v1 does not allow process or contract")
		}
		if !hasIOS {
			return fmt.Errorf("ios_v1 requires at least one of inputs/outcomes/systems")
		}
	case "iospc_v1":
		if len(profile.Process) == 0 && len(profile.Contract) == 0 {
			return fmt.Errorf("iospc_v1 requires process or contract")
		}
	default:
		return fmt.Errorf("unsupported capability_profile phase: %s", phase)
	}
	return nil
}

func ClassifyError(m any, status int, providerCode, providerType string) (code string, ok bool) {
	cls, exists := errorClass(m)
	if !exists {
		return "", false
	}
	if providerCode != "" && cls.ByErrorCode != nil {
		if n, hit := cls.ByErrorCode[providerCode]; hit {
			return normalizeErrorNameToCode(n)
		}
	}
	if providerType != "" && cls.ByErrorType != nil {
		if n, hit := cls.ByErrorType[providerType]; hit {
			return normalizeErrorNameToCode(n)
		}
	}
	if cls.ByHTTPStatus != nil {
		if n, hit := cls.ByHTTPStatus[fmt.Sprintf("%d", status)]; hit {
			return normalizeErrorNameToCode(n)
		}
	}
	return "", false
}

func RetryMaxAttempts(m any) (int, bool) {
	switch v := m.(type) {
	case *V1Manifest:
		if v.RetryPolicy.MaxRetries > 0 {
			return v.RetryPolicy.MaxRetries + 1, true
		}
	case *V2Manifest:
		if v.RetryPolicy.MaxRetries > 0 {
			return v.RetryPolicy.MaxRetries + 1, true
		}
	}
	return 0, false
}

func errorClass(m any) (ErrorClass, bool) {
	switch v := m.(type) {
	case *V1Manifest:
		return v.ErrorClass, true
	case *V2Manifest:
		return v.ErrorClass, true
	default:
		return ErrorClass{}, false
	}
}

func normalizeErrorNameToCode(name string) (string, bool) {
	switch name {
	case "invalid_request":
		return "E1001", true
	case "authentication":
		return "E1002", true
	case "permission_denied":
		return "E1003", true
	case "not_found":
		return "E1004", true
	case "request_too_large":
		return "E1005", true
	case "rate_limited":
		return "E2001", true
	case "quota_exhausted":
		return "E2002", true
	case "server_error":
		return "E3001", true
	case "overloaded":
		return "E3002", true
	case "timeout":
		return "E3003", true
	case "conflict":
		return "E4001", true
	case "cancelled":
		return "E4002", true
	default:
		return "", false
	}
}

func upperOrPOST(m string) string {
	if m == "" {
		return "POST"
	}
	if m == "get" || m == "GET" {
		return "GET"
	}
	if m == "delete" || m == "DELETE" {
		return "DELETE"
	}
	return "POST"
}

// StreamingDecoderFormat returns the decoder format from manifest for ARCH-002.
// Supports "openai_sse", "anthropic_sse", "sse" (defaults to openai_sse).
func StreamingDecoderFormat(m any) string {
	var format, strategy string
	switch v := m.(type) {
	case *V1Manifest:
		if v.Streaming != nil && v.Streaming.Decoder != nil {
			format, strategy = v.Streaming.Decoder.Format, v.Streaming.Decoder.Strategy
		}
	case *V2Manifest:
		if v.Streaming != nil && v.Streaming.Decoder != nil {
			format, strategy = v.Streaming.Decoder.Format, v.Streaming.Decoder.Strategy
		}
	default:
		return "openai_sse"
	}
	if format == "anthropic_sse" || strategy == "anthropic_event_stream" {
		return "anthropic_sse"
	}
	if format == "openai_sse" || format == "sse" || strategy == "openai_chat" {
		return "openai_sse"
	}
	return "openai_sse"
}

// GetFeatureFlags returns feature flags from a V2 manifest.
func (m *V2Manifest) GetFeatureFlags() map[string]bool {
	if m.Capabilities.FeatureFlags == nil {
		return make(map[string]bool)
	}
	return m.Capabilities.FeatureFlags
}

// IsFeatureEnabled checks if a feature flag is enabled.
func (m *V2Manifest) IsFeatureEnabled(flag string) bool {
	flags := m.GetFeatureFlags()
	return flags[flag]
}

// GetAllCapabilities returns all capabilities (required + optional).
func (m *V2Manifest) GetAllCapabilities() []string {
	caps := make([]string, 0)
	caps = append(caps, m.Capabilities.Required...)
	caps = append(caps, m.Capabilities.Optional...)
	return caps
}

// HasCapability checks if a capability is declared.
func (m *V2Manifest) HasCapability(capability string) bool {
	for _, c := range m.Capabilities.Required {
		if c == capability {
			return true
		}
	}
	for _, c := range m.Capabilities.Optional {
		if c == capability {
			return true
		}
	}
	return false
}

// GetFeatureFlags returns feature flags from a V1 manifest (empty, as V1 doesn't have feature flags).
func (m *V1Manifest) GetFeatureFlags() map[string]bool {
	return make(map[string]bool)
}

// IsFeatureEnabled checks if a feature flag is enabled in V1 (always false).
func (m *V1Manifest) IsFeatureEnabled(flag string) bool {
	return false
}

// GetAllCapabilities returns all capabilities for V1.
func (m *V1Manifest) GetAllCapabilities() []string {
	return m.Capabilities
}

// HasCapability checks if a capability is declared in V1.
func (m *V1Manifest) HasCapability(capability string) bool {
	for _, c := range m.Capabilities {
		if c == capability {
			return true
		}
	}
	return false
}
