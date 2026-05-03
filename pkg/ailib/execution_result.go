// Package ailib — E/P boundary execution result types (Paper1 §3.1–3.2).
//
// The execution layer (E) returns ExecutionResult with ExecutionMetadata.
// The contact / policy layer (P) consumes metadata for routing, retry, and degradation.
//
// Micro-retry (E-only, bounded): E MAY perform at most 1–2 automatic retries for
// transient transport failures before surfacing an error to P. E MUST NOT implement
// cross-provider fallback or policy-driven retry loops.

package ailib

// ExecutionUsage is token usage aligned with driver usage fields.
type ExecutionUsage struct {
	PromptTokens        int  `json:"prompt_tokens"`
	CompletionTokens    int  `json:"completion_tokens"`
	TotalTokens         int  `json:"total_tokens"`
	ReasoningTokens     *int `json:"reasoning_tokens,omitempty"`
	CacheReadTokens     *int `json:"cache_read_tokens,omitempty"`
	CacheCreationTokens *int `json:"cache_creation_tokens,omitempty"`
}

// ExecutionMetadata is returned with every E-layer call for P-layer policy decisions.
type ExecutionMetadata struct {
	ProviderID           string          `json:"provider_id"`
	ModelID              string          `json:"model_id"`
	ExecutionLatencyMs   uint64          `json:"execution_latency_ms"`
	TranslationLatencyMs uint64          `json:"translation_latency_ms"`
	MicroRetryCount      uint8           `json:"micro_retry_count"`
	ErrorCode            *string         `json:"error_code,omitempty"`
	Usage                *ExecutionUsage `json:"usage,omitempty"`
}

// ExecutionResult is the successful execution envelope from E: payload plus metadata for P.
type ExecutionResult[T any] struct {
	Data     T                 `json:"data"`
	Metadata ExecutionMetadata `json:"metadata"`
}
