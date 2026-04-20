// Package streaming provides SSE decoding and event mapping for AI providers.
// 实现 AI-Protocol 的 operator pipeline 架构: Decoder → EventMapper
package streaming

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/ailib-official/ai-lib-go/pkg/ailib"
)

// SSEDecoder parses SSE streams into data frames.
type SSEDecoder struct {
	scanner *bufio.Scanner
	buffer  string
}

// NewSSEDecoder creates a new SSE decoder from a reader.
func NewSSEDecoder(r io.Reader) *SSEDecoder {
	return &SSEDecoder{
		scanner: bufio.NewScanner(r),
	}
}

// Next returns the next data frame, or empty string if no more data.
func (d *SSEDecoder) Next() (string, bool) {
	for d.scanner.Scan() {
		line := d.scanner.Text()

		// Empty line signals end of event
		if line == "" {
			if d.buffer != "" {
				data := d.buffer
				d.buffer = ""
				return data, true
			}
			continue
		}

		// Parse SSE line
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				return "", false
			}
			d.buffer = data
		} else if strings.HasPrefix(line, "event: ") {
			// Event type - we process after data
			continue
		}
	}

	// Return any remaining buffer
	if d.buffer != "" {
		data := d.buffer
		d.buffer = ""
		return data, true
	}

	return "", false
}

// Err returns any scanner error.
func (d *SSEDecoder) Err() error {
	return d.scanner.Err()
}

// EventMapper transforms parsed JSON frames into streaming events.
type EventMapper interface {
	Map(data map[string]any) []ailib.StreamingEvent
}

// OpenAIEventMapper maps OpenAI-style SSE events.
type OpenAIEventMapper struct {
	toolCallIDs map[int]string // index -> id mapping
}

// NewOpenAIEventMapper creates a new OpenAI event mapper.
func NewOpenAIEventMapper() *OpenAIEventMapper {
	return &OpenAIEventMapper{
		toolCallIDs: make(map[int]string),
	}
}

// Map transforms an OpenAI SSE frame into streaming events.
func (m *OpenAIEventMapper) Map(data map[string]any) []ailib.StreamingEvent {
	var events []ailib.StreamingEvent

	// Check for error
	if errObj, ok := data["error"].(map[string]any); ok {
		events = append(events, ailib.StreamingEvent{
			Type: "StreamError",
			Error: &ailib.EventError{
				Code:    fmt.Sprintf("%v", errObj["code"]),
				Message: fmt.Sprintf("%v", errObj["message"]),
			},
		})
		return events
	}

	// Extract reasoning_content (for thinking models)
	if choices, ok := data["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			if delta, ok := choice["delta"].(map[string]any); ok {
				// Reasoning/thinking content
				if reasoning, ok := delta["reasoning_content"].(string); ok && reasoning != "" {
					events = append(events, ailib.StreamingEvent{
						Type:     "ThinkingDelta",
						Thinking: reasoning,
					})
				}

				// Regular content
				if content, ok := delta["content"].(string); ok && content != "" {
					events = append(events, ailib.StreamingEvent{
						Type:  "PartialContentDelta",
						Delta: content,
					})
				}

				// Tool calls
				if toolCalls, ok := delta["tool_calls"].([]any); ok {
					for _, tc := range toolCalls {
						if tcMap, ok := tc.(map[string]any); ok {
							index := int(getFloat64(tcMap["index"]))
							id, _ := tcMap["id"].(string)

							if fn, ok := tcMap["function"].(map[string]any); ok {
								name, _ := fn["name"].(string)
								args, _ := fn["arguments"].(string)

								// Tool call started
								if id != "" && name != "" {
									m.toolCallIDs[index] = id
									events = append(events, ailib.StreamingEvent{
										Type:       "ToolCallStarted",
										ToolCallID: id,
										ToolName:   name,
										Index:      intPtr(index),
									})
								}

								// Partial tool call arguments
								if args != "" {
									// Use stored ID if available
									resolvedID := id
									if resolvedID == "" {
										resolvedID = m.toolCallIDs[index]
									}
									if resolvedID != "" {
										events = append(events, ailib.StreamingEvent{
											Type:       "PartialToolCall",
											ToolCallID: resolvedID,
											Arguments:  args,
											Index:      intPtr(index),
											IsComplete: false,
										})
									}
								}
							}
						}
					}
				}
			}

			// Finish reason
			if finishReason, ok := choice["finish_reason"].(string); ok && finishReason != "" {
				events = append(events, ailib.StreamingEvent{
					Type:         "Metadata",
					FinishReason: finishReason,
				})
			}
		}
	}

	// Usage
	if usage, ok := data["usage"].(map[string]any); ok {
		events = append(events, ailib.StreamingEvent{
			Type:  "Metadata",
			Usage: parseUsage(usage),
		})
	}

	return events
}

// AnthropicEventMapper maps Anthropic-style SSE events.
type AnthropicEventMapper struct{}

// NewAnthropicEventMapper creates a new Anthropic event mapper.
func NewAnthropicEventMapper() *AnthropicEventMapper {
	return &AnthropicEventMapper{}
}

// Map transforms an Anthropic SSE frame into streaming events.
func (m *AnthropicEventMapper) Map(data map[string]any) []ailib.StreamingEvent {
	var events []ailib.StreamingEvent

	eventType, _ := data["type"].(string)

	switch eventType {
	case "content_block_delta":
		if delta, ok := data["delta"].(map[string]any); ok {
			index := int(getFloat64(data["index"]))

			switch delta["type"] {
			case "text_delta":
				if text, ok := delta["text"].(string); ok && text != "" {
					events = append(events, ailib.StreamingEvent{
						Type:  "PartialContentDelta",
						Delta: text,
						Index: intPtr(index),
					})
				}
			case "thinking_delta":
				if thinking, ok := delta["thinking"].(string); ok && thinking != "" {
					events = append(events, ailib.StreamingEvent{
						Type:     "ThinkingDelta",
						Thinking: thinking,
					})
				}
			case "input_json_delta":
				if partialJSON, ok := delta["partial_json"].(string); ok {
					toolUseID := fmt.Sprintf("tool_%d", index)
					events = append(events, ailib.StreamingEvent{
						Type:       "PartialToolCall",
						ToolCallID: toolUseID,
						Arguments:  partialJSON,
						Index:      intPtr(index),
						IsComplete: false,
					})
				}
			}
		}

	case "content_block_start":
		if contentBlock, ok := data["content_block"].(map[string]any); ok {
			index := int(getFloat64(data["index"]))
			if contentBlock["type"] == "tool_use" {
				events = append(events, ailib.StreamingEvent{
					Type:       "ToolCallStarted",
					ToolCallID: contentBlock["id"].(string),
					ToolName:   contentBlock["name"].(string),
					Index:      intPtr(index),
				})
			}
		}

	case "message_delta":
		if delta, ok := data["delta"].(map[string]any); ok {
			if stopReason, ok := delta["stop_reason"].(string); ok {
				events = append(events, ailib.StreamingEvent{
					Type:         "Metadata",
					FinishReason: stopReason,
				})
			}
		}
		if usage, ok := data["usage"].(map[string]any); ok {
			events = append(events, ailib.StreamingEvent{
				Type:  "Metadata",
				Usage: parseUsage(usage),
			})
		}

	case "message_start":
		if msg, ok := data["message"].(map[string]any); ok {
			if usage, ok := msg["usage"].(map[string]any); ok {
				events = append(events, ailib.StreamingEvent{
					Type:  "Metadata",
					Usage: parseUsage(usage),
				})
			}
		}
	}

	return events
}

// ToolCallAccumulator accumulates partial tool call arguments.
type ToolCallAccumulator struct {
	accumulated map[string]struct {
		id        string
		name      string
		arguments string
	}
}

// NewToolCallAccumulator creates a new accumulator.
func NewToolCallAccumulator() *ToolCallAccumulator {
	return &ToolCallAccumulator{
		accumulated: make(map[string]struct {
			id        string
			name      string
			arguments string
		}),
	}
}

// Accumulate merges a partial tool call event.
func (a *ToolCallAccumulator) Accumulate(event ailib.StreamingEvent) *ailib.StreamingEvent {
	if event.Type != "PartialToolCall" && event.Type != "ToolCallStarted" {
		return nil
	}

	id := event.ToolCallID
	if id == "" {
		return nil
	}

	existing, exists := a.accumulated[id]

	switch event.Type {
	case "ToolCallStarted":
		a.accumulated[id] = struct {
			id        string
			name      string
			arguments string
		}{id: id, name: event.ToolName, arguments: ""}
		return &event

	case "PartialToolCall":
		if exists {
			existing.arguments += event.Arguments
			a.accumulated[id] = existing
			return &ailib.StreamingEvent{
				Type:       "PartialToolCall",
				ToolCallID: id,
				Arguments:  existing.arguments,
				Index:      event.Index,
				IsComplete: event.IsComplete,
			}
		}
		a.accumulated[id] = struct {
			id        string
			name      string
			arguments string
		}{id: id, name: event.ToolName, arguments: event.Arguments}
		return &event
	}

	return nil
}

// GetAll returns all accumulated tool calls.
func (a *ToolCallAccumulator) GetAll() []map[string]any {
	var result []map[string]any
	for id, tc := range a.accumulated {
		result = append(result, map[string]any{
			"id":   id,
			"name": tc.name,
			"arguments": tc.arguments,
		})
	}
	return result
}

// Clear resets the accumulator.
func (a *ToolCallAccumulator) Clear() {
	a.accumulated = make(map[string]struct {
		id        string
		name      string
		arguments string
	})
}

// Helper functions

// parseUsage normalizes provider-specific usage payloads into ailib.Usage.
// Supports:
//   - OpenAI flat fields (prompt_tokens/completion_tokens/total_tokens) plus
//     completion_tokens_details.reasoning_tokens and prompt_tokens_details.cached_tokens.
//   - Anthropic input_tokens/output_tokens plus cache_creation_input_tokens /
//     cache_read_input_tokens.
//   - Canonical ai-protocol extended fields (reasoning_tokens, cache_read_tokens,
//     cache_creation_tokens) and legacy cache_write_tokens alias.
// Returns nil when no numeric fields could be populated so callers can distinguish
// absence from all-zero usage.
func parseUsage(data map[string]any) *ailib.Usage {
	if len(data) == 0 {
		return nil
	}
	usage := &ailib.Usage{}
	populated := false
	setInt := func(dst *int, keys ...string) {
		for _, k := range keys {
			if v, ok := data[k].(float64); ok {
				*dst = int(v)
				populated = true
				return
			}
		}
	}

	setInt(&usage.PromptTokens, "prompt_tokens", "input_tokens")
	setInt(&usage.CompletionTokens, "completion_tokens", "output_tokens")
	setInt(&usage.TotalTokens, "total_tokens")
	setInt(&usage.ReasoningTokens, "reasoning_tokens")
	setInt(&usage.CacheReadTokens, "cache_read_tokens", "cache_read_input_tokens")
	setInt(&usage.CacheCreationTokens,
		"cache_creation_tokens",
		"cache_creation_input_tokens",
		"cache_write_tokens",
	)

	// OpenAI extended detail blocks
	if details, ok := data["completion_tokens_details"].(map[string]any); ok {
		if v, ok := details["reasoning_tokens"].(float64); ok && usage.ReasoningTokens == 0 {
			usage.ReasoningTokens = int(v)
			populated = true
		}
	}
	if details, ok := data["prompt_tokens_details"].(map[string]any); ok {
		if v, ok := details["cached_tokens"].(float64); ok && usage.CacheReadTokens == 0 {
			usage.CacheReadTokens = int(v)
			populated = true
		}
	}

	if !populated {
		return nil
	}
	if usage.TotalTokens == 0 && (usage.PromptTokens > 0 || usage.CompletionTokens > 0) {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	return usage
}

func getFloat64(v any) float64 {
	switch val := v.(type) {
	case float64:
		return val
	case int:
		return float64(val)
	case int64:
		return float64(val)
	default:
		return 0
	}
}

func intPtr(i int) *int {
	return &i
}

// DecodeFrame decodes a raw SSE data frame into a map.
func DecodeFrame(data string) (map[string]any, error) {
	var result map[string]any
	if err := json.Unmarshal([]byte(data), &result); err != nil {
		return nil, fmt.Errorf("failed to decode frame: %w", err)
	}
	return result, nil
}
