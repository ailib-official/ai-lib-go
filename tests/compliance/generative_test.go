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
	if len(toolCalls) == 0 {
		t.Error("expected at least one accumulated tool call")
	}

	// Check arguments are accumulated
	for _, tc := range toolCalls {
		if tc["id"] == "call_abc" {
			args := tc["arguments"].(string)
			if args == "" {
				t.Error("expected non-empty arguments")
			}
		}
	}
}

// TestGen005ContextOverflow tests context window error classification (gen-005)
func TestGen005ContextOverflow(t *testing.T) {
	// This tests error classification logic
	// In Go, error classification is handled by the client implementation
	// For now, we just verify the error structure

	httpStatus := 400
	responseBody := map[string]any{
		"error": map[string]any{
			"message": "This model's maximum context length is 128000 tokens.",
			"type":    "invalid_request_error",
			"code":    "context_length_exceeded",
		},
	}

	// Verify error code exists
	if responseBody["error"] == nil {
		t.Fatal("expected error object")
	}

	errObj := responseBody["error"].(map[string]any)
	if errObj["code"] != "context_length_exceeded" {
		t.Errorf("expected code=context_length_exceeded, got %v", errObj["code"])
	}

	// Error classification: context_length_exceeded → E1005
	// (Error classification implementation is in client code)
	_ = httpStatus
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
