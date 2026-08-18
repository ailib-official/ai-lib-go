// Package thinking extracts OpenAI-compatible reasoning wire fields (ALG-RSN-001).
// 中文：从 delta/message 结构化字段提取思考文本；单一别名表，供 SSE mapper 与
// non-stream enrich 共用（GOV-007 / X-RUNTIME-MIRROR of ALR-RSN-001）。
package thinking

// OpenAICompatThinkingKeys are wire keys observed across OpenAI-compatible reasoners.
// Order is preference when multiple keys appear on the same object.
var OpenAICompatThinkingKeys = []string{
	"reasoning_content",
	"reasoning",
	"thinking",
	"thought",
	"reasoning_text",
	"analysis",
}

// FirstNonemptyStringField returns the first non-empty string among keys on obj.
func FirstNonemptyStringField(obj map[string]any, keys []string) string {
	if obj == nil {
		return ""
	}
	for _, key := range keys {
		if s, ok := obj[key].(string); ok && s != "" {
			return s
		}
	}
	return ""
}

// FromOpenaiCompatDelta returns thinking text from choices[0].delta.* (streaming).
func FromOpenaiCompatDelta(frame map[string]any) string {
	choices, ok := frame["choices"].([]any)
	if !ok || len(choices) == 0 {
		return ""
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		return ""
	}
	delta, ok := choice["delta"].(map[string]any)
	if !ok {
		return ""
	}
	return FirstNonemptyStringField(delta, OpenAICompatThinkingKeys)
}

// FromOpenaiCompatMessage returns thinking text from choices[0].message.* (non-streaming).
func FromOpenaiCompatMessage(frame map[string]any) string {
	choices, ok := frame["choices"].([]any)
	if !ok || len(choices) == 0 {
		return ""
	}
	choice, ok := choices[0].(map[string]any)
	if !ok {
		return ""
	}
	msg, ok := choice["message"].(map[string]any)
	if !ok {
		return ""
	}
	return FirstNonemptyStringField(msg, OpenAICompatThinkingKeys)
}
