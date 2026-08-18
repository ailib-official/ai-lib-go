package thinking

import "testing"

func TestDeltaPrefersReasoningContent(t *testing.T) {
	frame := map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{
				"reasoning_content": "a",
				"thinking":          "b",
				"content":           "c",
			},
		}},
	}
	if got := FromOpenaiCompatDelta(frame); got != "a" {
		t.Fatalf("got %q want a", got)
	}
}

func TestDeltaAliasThinking(t *testing.T) {
	frame := map[string]any{
		"choices": []any{map[string]any{
			"delta": map[string]any{"thinking": "plan"},
		}},
	}
	if got := FromOpenaiCompatDelta(frame); got != "plan" {
		t.Fatalf("got %q want plan", got)
	}
}

func TestMessageReasoningNotContent(t *testing.T) {
	frame := map[string]any{
		"choices": []any{map[string]any{
			"message": map[string]any{
				"content":           "",
				"reasoning_content": "only think",
			},
		}},
	}
	if got := FromOpenaiCompatMessage(frame); got != "only think" {
		t.Fatalf("got %q want only think", got)
	}
}
