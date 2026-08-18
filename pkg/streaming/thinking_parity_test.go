package streaming

import "testing"

func TestOpenAIEventMapper_SameFrameThinkingAndContent(t *testing.T) {
	m := NewOpenAIEventMapper()
	events := m.Map(map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{
				"reasoning_content": "plan",
				"content":           "answer",
			},
		}},
	})
	if len(events) != 2 {
		t.Fatalf("want 2 events got %d", len(events))
	}
	if events[0].Type != "ThinkingDelta" || events[0].Thinking != "plan" {
		t.Fatalf("event0: %+v", events[0])
	}
	if events[1].Type != "PartialContentDelta" || events[1].Delta != "answer" {
		t.Fatalf("event1: %+v", events[1])
	}
}

func TestOpenAIEventMapper_ThinkingAlias(t *testing.T) {
	m := NewOpenAIEventMapper()
	events := m.Map(map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{"thinking": "via-alias"},
		}},
	})
	if len(events) != 1 || events[0].Type != "ThinkingDelta" || events[0].Thinking != "via-alias" {
		t.Fatalf("events: %+v", events)
	}
}
