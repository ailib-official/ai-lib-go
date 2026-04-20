// Package compliance_test provides generative capability compliance tests.
// PT-066: ai-lib-go generative adaptation (gen-001~gen-007)
package compliance_test

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/ailib-official/ai-lib-go/internal/protocol"
	"github.com/ailib-official/ai-lib-go/pkg/ailib"
	"github.com/ailib-official/ai-lib-go/pkg/streaming"
	"gopkg.in/yaml.v3"
)

// TestGen001FeatureFlags tests feature flag consumption (gen-001)
func TestGen001FeatureFlags(t *testing.T) {
	// Load mock provider manifest
	manifestPath := os.Getenv("AI_PROTOCOL_DIR")
	if manifestPath == "" {
		manifestPath = "../ai-protocol"
	}
	manifestPath += "/tests/compliance/fixtures/providers/mock-openai.yaml"

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Skipf("skipping: fixture not found: %v", err)
	}

	var manifest protocol.V2Manifest
	if err := yaml.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("failed to parse manifest: %v", err)
	}

	// Test GetFeatureFlags
	featureFlags := manifest.GetFeatureFlags()
	if len(featureFlags) == 0 {
		t.Log("V1 manifest or no feature flags")
	} else {
		// V2 manifest should have feature flags
		if manifest.Capabilities.FeatureFlags != nil {
			if !manifest.IsFeatureEnabled("structured_output") {
				t.Error("expected structured_output to be enabled")
			}
		}
	}

	// Test GetAllCapabilities
	allCaps := manifest.GetAllCapabilities()
	if len(allCaps) == 0 {
		t.Error("expected at least one capability")
	}

	// Test HasCapability
	if !manifest.HasCapability("streaming") {
		t.Error("expected streaming capability")
	}
}

// TestGen002TokenUsage tests token usage extraction (gen-002)
func TestGen002TokenUsage(t *testing.T) {
	responseBody := `{
		"id": "chatcmpl-test-001",
		"object": "chat.completion",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Hello, world!"
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 5,
			"total_tokens": 15,
			"reasoning_tokens": 3
		}
	}`

	var response struct {
		Usage *ailib.Usage `json:"usage"`
	}
	if err := json.Unmarshal([]byte(responseBody), &response); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if response.Usage == nil {
		t.Fatal("expected usage object")
	}

	if response.Usage.PromptTokens != 10 {
		t.Errorf("expected prompt_tokens=10, got %d", response.Usage.PromptTokens)
	}
	if response.Usage.CompletionTokens != 5 {
		t.Errorf("expected completion_tokens=5, got %d", response.Usage.CompletionTokens)
	}
	if response.Usage.TotalTokens != 15 {
		t.Errorf("expected total_tokens=15, got %d", response.Usage.TotalTokens)
	}
	if response.Usage.ReasoningTokens != 3 {
		t.Errorf("expected reasoning_tokens=3, got %d", response.Usage.ReasoningTokens)
	}
}

// TestGen002UsageNormalization asserts parseUsage lifts provider-specific usage
// shapes (OpenAI completion_tokens_details, Anthropic input_tokens + cache_*_input_tokens)
// into the unified ailib.Usage (ARCH-003 parity with ai-lib-ts normalizeUsage).
func TestGen002UsageNormalization(t *testing.T) {
	// Simulate feeding through streaming mapper by crafting a chunk whose
	// `usage` field contains OpenAI-style nested details.
	openaiChunk := `{"choices":[{"delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":3}}}`
	data, err := streaming.DecodeFrame(openaiChunk)
	if err != nil {
		t.Fatalf("decode openai chunk: %v", err)
	}
	events := streaming.NewOpenAIEventMapper().Map(data)
	var u *ailib.Usage
	for _, e := range events {
		if e.Usage != nil {
			u = e.Usage
		}
	}
	if u == nil {
		t.Fatal("expected Metadata event with usage")
	}
	if u.PromptTokens != 10 || u.CompletionTokens != 5 || u.TotalTokens != 15 {
		t.Errorf("flat tokens wrong: %+v", u)
	}
	if u.ReasoningTokens != 3 {
		t.Errorf("expected reasoning_tokens=3 lifted from completion_tokens_details, got %d", u.ReasoningTokens)
	}
	if u.CacheReadTokens != 4 {
		t.Errorf("expected cache_read_tokens=4 lifted from prompt_tokens_details.cached_tokens, got %d", u.CacheReadTokens)
	}

	// Anthropic message_delta with input_tokens/output_tokens + cache aliases
	anthChunk := `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":200,"output_tokens":50,"cache_creation_input_tokens":120,"cache_read_input_tokens":80}}`
	data2, err := streaming.DecodeFrame(anthChunk)
	if err != nil {
		t.Fatalf("decode anthropic chunk: %v", err)
	}
	events2 := streaming.NewAnthropicEventMapper().Map(data2)
	var u2 *ailib.Usage
	for _, e := range events2 {
		if e.Usage != nil {
			u2 = e.Usage
		}
	}
	if u2 == nil {
		t.Fatal("expected Metadata event with usage (anthropic)")
	}
	if u2.PromptTokens != 200 {
		t.Errorf("expected prompt_tokens=200 from input_tokens, got %d", u2.PromptTokens)
	}
	if u2.CompletionTokens != 50 {
		t.Errorf("expected completion_tokens=50 from output_tokens, got %d", u2.CompletionTokens)
	}
	if u2.CacheCreationTokens != 120 {
		t.Errorf("expected cache_creation_tokens=120, got %d", u2.CacheCreationTokens)
	}
	if u2.CacheReadTokens != 80 {
		t.Errorf("expected cache_read_tokens=80, got %d", u2.CacheReadTokens)
	}
	if u2.TotalTokens != 250 {
		t.Errorf("expected total_tokens=prompt+completion=250 when absent, got %d", u2.TotalTokens)
	}
}

// TestGen003StructuredOutput tests JSON mode (gen-003)
func TestGen003StructuredOutput(t *testing.T) {
	opts := &ailib.ChatOptions{
		Model: "gpt-4o",
		ResponseFormat: map[string]any{
			"type": "json_object",
		},
	}

	if opts.ResponseFormat == nil {
		t.Error("expected response_format to be set")
	}
	if opts.ResponseFormat["type"] != "json_object" {
		t.Error("expected response_format.type=json_object")
	}
}

// TestGen004StreamingToolCall tests streaming tool call accumulation (gen-004)
func TestGen004StreamingToolCall(t *testing.T) {
	mapper := streaming.NewOpenAIEventMapper()
	accumulator := streaming.NewToolCallAccumulator()

	chunks := []string{
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\": \"SF\"}"}}]}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":15,"total_tokens":35}}`,
	}

	var allEvents []ailib.StreamingEvent
	for _, chunk := range chunks {
		data, err := streaming.DecodeFrame(chunk)
		if err != nil {
			t.Fatalf("failed to decode frame: %v", err)
		}

		events := mapper.Map(data)
		for _, event := range events {
			accumulator.Accumulate(event)
			allEvents = append(allEvents, event)
		}
	}

	// Check for ToolCallStarted
	started := false
	for _, e := range allEvents {
		if e.Type == "ToolCallStarted" {
			started = true
			if e.ToolCallID != "call_abc" {
				t.Errorf("expected tool_call_id=call_abc, got %s", e.ToolCallID)
			}
			if e.ToolName != "get_weather" {
				t.Errorf("expected tool_name=get_weather, got %s", e.ToolName)
			}
		}
	}
	if !started {
		t.Error("expected ToolCallStarted event")
	}

	// Check accumulated tool calls
	toolCalls := accumulator.GetAll()
	if len(toolCalls) != 1 {
		t.Fatalf("expected exactly one accumulated tool call, got %d", len(toolCalls))
	}

	tc := toolCalls[0]
	if tc["id"] != "call_abc" {
		t.Errorf("expected id=call_abc, got %v", tc["id"])
	}
	if tc["name"] != "get_weather" {
		t.Errorf("expected name=get_weather, got %v", tc["name"])
	}
	args, ok := tc["arguments"].(string)
	if !ok {
		t.Fatalf("arguments should be string, got %T", tc["arguments"])
	}
	// Partial chunks concatenate into a complete JSON payload
	if args != `{"location": "SF"}` {
		t.Errorf("expected concatenated arguments=%q, got %q", `{"location": "SF"}`, args)
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("accumulated args should parse as JSON: %v", err)
	}
	if parsed["location"] != "SF" {
		t.Errorf("expected parsed location=SF, got %v", parsed["location"])
	}
}

// TestGen005ContextOverflow tests context window error classification (gen-005)
// using the manifest-driven protocol.ClassifyError classifier.
func TestGen005ContextOverflow(t *testing.T) {
	manifest := &protocol.V2Manifest{
		ID:              "mock",
		ProtocolVersion: "2.0",
		ErrorClass: protocol.ErrorClass{
			ByHTTPStatus: map[string]string{
				"429": "rate_limited",
				"413": "request_too_large",
				"401": "authentication",
			},
			ByErrorCode: map[string]string{
				"context_length_exceeded": "request_too_large",
			},
		},
	}

	// by_error_code path — context_length_exceeded → E1005
	code, ok := protocol.ClassifyError(manifest, 400, "context_length_exceeded", "invalid_request_error")
	if !ok {
		t.Fatal("expected classification to match by_error_code")
	}
	if code != "E1005" {
		t.Errorf("expected E1005 (REQUEST_TOO_LARGE), got %s", code)
	}

	// by_http_status path — 429 → rate_limited → E2001
	code, ok = protocol.ClassifyError(manifest, 429, "", "")
	if !ok || code != "E2001" {
		t.Errorf("expected E2001 via by_http_status, got %q ok=%v", code, ok)
	}

	// Unmapped status returns !ok so caller can fall back to heuristics.
	if _, ok := protocol.ClassifyError(manifest, 599, "", ""); ok {
		t.Error("expected unmapped status to return ok=false")
	}

	// Manifest without error_classification cannot classify.
	empty := &protocol.V2Manifest{ID: "none", ProtocolVersion: "2.0"}
	if _, ok := protocol.ClassifyError(empty, 400, "context_length_exceeded", ""); ok {
		// Empty ErrorClass struct still returns ok=true from errorClass but no mapping hits;
		// ensure we don't accidentally produce a code.
		t.Error("expected empty error_classification to produce no code")
	}
}

// TestGen006ReasoningStreaming tests thinking block streaming (gen-006)
func TestGen006ReasoningStreaming(t *testing.T) {
	mapper := streaming.NewAnthropicEventMapper()

	chunks := []string{
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me analyze..."}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 42."}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":25}}`,
		`{"type":"message_stop"}`,
	}

	var allEvents []ailib.StreamingEvent
	for _, chunk := range chunks {
		data, err := streaming.DecodeFrame(chunk)
		if err != nil {
			t.Fatalf("failed to decode frame: %v", err)
		}

		events := mapper.Map(data)
		allEvents = append(allEvents, events...)
	}

	// Check for ThinkingDelta
	thinkingEvent := false
	for _, e := range allEvents {
		if e.Type == "ThinkingDelta" {
			thinkingEvent = true
			if e.Thinking == "" {
				t.Error("expected non-empty thinking content")
			}
			if e.Thinking != "Let me analyze..." {
				t.Errorf("expected thinking='Let me analyze...', got %s", e.Thinking)
			}
		}
	}
	if !thinkingEvent {
		t.Error("expected ThinkingDelta event")
	}

	// Check for text content
	contentEvent := false
	for _, e := range allEvents {
		if e.Type == "PartialContentDelta" {
			contentEvent = true
			if e.Delta == "" {
				t.Error("expected non-empty content")
			}
		}
	}
	if !contentEvent {
		t.Error("expected PartialContentDelta event")
	}

	// Check for finish_reason
	finishEvent := false
	for _, e := range allEvents {
		if e.Type == "Metadata" && e.FinishReason != "" {
			finishEvent = true
			if e.FinishReason != "end_turn" {
				t.Errorf("expected finish_reason=end_turn, got %s", e.FinishReason)
			}
		}
	}
	if !finishEvent {
		t.Error("expected Metadata event with finish_reason")
	}
}

// TestGen007MCPGating tests MCP capability gating (gen-007)
func TestGen007MCPGating(t *testing.T) {
	// Create a mock manifest with mcp_client capability
	manifest := &protocol.V2Manifest{
		ID:              "mock-provider",
		ProtocolVersion: "2.0",
		Capabilities: protocol.V2Caps{
			Required: []string{"chat", "streaming"},
			Optional: []string{"mcp_client", "tools"},
		},
	}

	// Check if mcp_client capability is declared
	if !manifest.HasCapability("mcp_client") {
		t.Error("expected mcp_client capability to be declared")
	}

	// Check if required capabilities are present
	if !manifest.HasCapability("chat") {
		t.Error("expected chat capability")
	}
	if !manifest.HasCapability("streaming") {
		t.Error("expected streaming capability")
	}
}
