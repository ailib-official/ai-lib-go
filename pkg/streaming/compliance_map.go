// Compliance-oriented streaming helpers used by ai-protocol conformance tests.
// 合规测试用的流式映射辅助函数（与 event_map 简化语义对齐）。
package streaming

import "github.com/ailib-official/ai-lib-go/pkg/ailib"

// ComplianceEventsFromOpenAIFrame maps a decoded OpenAI chat frame to protocol
// compliance event records (PartialContentDelta, PartialToolCall, StreamEnd).
func ComplianceEventsFromOpenAIFrame(frame map[string]any) []map[string]any {
	choices, _ := frame["choices"].([]any)
	if len(choices) == 0 {
		return nil
	}
	choice, _ := choices[0].(map[string]any)
	delta, _ := choice["delta"].(map[string]any)

	out := make([]map[string]any, 0, 3)
	if content, ok := delta["content"]; ok {
		out = append(out, map[string]any{"type": "PartialContentDelta", "content": content})
	}
	if toolCalls, ok := delta["tool_calls"]; ok {
		out = append(out, map[string]any{"type": "PartialToolCall", "tool_calls": toolCalls})
	}
	if fr, ok := choice["finish_reason"]; ok && fr != nil {
		out = append(out, map[string]any{"type": "StreamEnd", "finish_reason": fr})
	}
	return out
}

// AssembleToolCallPartials merges partial tool-call chunks via ToolCallAccumulator.
func AssembleToolCallPartials(chunks []map[string]any) []map[string]any {
	type key struct {
		idx int
		id  string
	}
	order := make([]key, 0, len(chunks))
	meta := make(map[key]map[string]any)
	seen := make(map[key]bool)

	accum := NewToolCallAccumulator()
	started := make(map[string]bool)
	for _, chunk := range chunks {
		k := key{idx: intFromAny(chunk["index"]), id: stringFromAny(chunk["id"])}
		fn, _ := chunk["function"].(map[string]any)
		name := stringFromAny(fn["name"])
		args := stringFromAny(fn["arguments"])

		if !seen[k] {
			seen[k] = true
			order = append(order, k)
			meta[k] = map[string]any{
				"type": firstString(stringFromAny(chunk["type"]), "function"),
				"name": name,
			}
		}
		if name != "" && !started[k.id] {
			started[k.id] = true
			accum.Accumulate(ailib.StreamingEvent{
				Type:       "ToolCallStarted",
				ToolCallID: k.id,
				ToolName:   name,
				Index:      intPtr(k.idx),
			})
		}
		accum.Accumulate(ailib.StreamingEvent{
			Type:       "PartialToolCall",
			ToolCallID: k.id,
			Arguments:  args,
			Index:      intPtr(k.idx),
		})
	}

	byID := make(map[string]map[string]any)
	for _, tc := range accum.GetAll() {
		byID[stringFromAny(tc["id"])] = tc
	}

	out := make([]map[string]any, 0, len(order))
	for _, k := range order {
		tc := byID[k.id]
		out = append(out, map[string]any{
			"index": k.idx,
			"id":    k.id,
			"type":  meta[k]["type"],
			"function": map[string]any{
				"name":      meta[k]["name"],
				"arguments": stringFromAny(tc["arguments"]),
			},
		})
	}
	return out
}

func stringFromAny(v any) string {
	s, _ := v.(string)
	return s
}

func intFromAny(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}

func firstString(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
