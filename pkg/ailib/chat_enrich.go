package ailib

import (
	"encoding/json"
	"strings"

	"github.com/ailib-official/ai-lib-go/internal/protocol"
)

// EnrichNonstreamChatResponse fills assistant text / usage / finish_reason from manifest
// response_paths when needed (v2 parity with Rust/Python).
func EnrichNonstreamChatResponse(manifest any, root map[string]any, out *ChatResponse) {
	rp := protocol.ResponsePathsFor(manifest)
	if root == nil {
		return
	}

	firstNonEmpty := func(paths ...string) string {
		for _, p := range paths {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			if s, ok := protocol.GetStringAtPath(root, p); ok {
				return s
			}
		}
		return ""
	}

	var text string
	if rp != nil && strings.TrimSpace(rp.Content) != "" {
		if s, ok := protocol.GetStringAtPath(root, rp.Content); ok {
			text = s
		}
	}
	if text == "" {
		text = firstNonEmpty("choices[0].message.content")
	}
	if text == "" {
		reason := []string{}
		if rp != nil {
			reason = append(reason, rp.ReasoningContent, rp.Reasoning)
		}
		reason = append(reason, "choices[0].message.reasoning_content")
		text = firstNonEmpty(reason...)
	}

	if text != "" {
		ensureFirstChoiceContent(out, text)
	}

	if rp != nil && strings.TrimSpace(rp.Usage) != "" {
		if u, ok := protocol.GetAtPath(root, rp.Usage); ok {
			_ = applyUsage(out, u)
		}
	}
	if out.Usage == nil {
		if u, ok := root["usage"]; ok {
			_ = applyUsage(out, u)
		}
	}

	if rp != nil && strings.TrimSpace(rp.FinishReason) != "" {
		if fr, ok := protocol.GetStringAtPath(root, rp.FinishReason); ok {
			setFinishReason(out, fr)
		}
	}
}

func ensureFirstChoiceContent(out *ChatResponse, text string) {
	if len(out.Choices) == 0 {
		out.Choices = []Choice{{
			Index:   0,
			Message: Message{Role: RoleAssistant, Content: text},
		}}
		return
	}
	out.Choices[0].Message.Content = text
}

func setFinishReason(out *ChatResponse, fr string) {
	if len(out.Choices) == 0 {
		out.Choices = []Choice{{
			Index:        0,
			Message:      Message{Role: RoleAssistant, Content: ""},
			FinishReason: fr,
		}}
		return
	}
	if out.Choices[0].FinishReason == "" {
		out.Choices[0].FinishReason = fr
	}
}

func applyUsage(out *ChatResponse, u any) bool {
	b, err := json.Marshal(u)
	if err != nil {
		return false
	}
	var usage Usage
	if err := json.Unmarshal(b, &usage); err != nil {
		return false
	}
	out.Usage = &usage
	return true
}
