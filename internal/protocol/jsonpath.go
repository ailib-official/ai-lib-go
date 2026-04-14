// Package protocol — JSON path helpers for manifest response_paths (Rust/Python parity).
package protocol

import (
	"strconv"
	"strings"
)

func normalizeJSONPath(path string) string {
	p := strings.TrimSpace(path)
	p = strings.TrimPrefix(p, "$.")
	p = strings.TrimPrefix(p, "$")
	p = strings.ReplaceAll(p, "[", ".")
	p = strings.ReplaceAll(p, "]", "")
	return p
}

// GetAtPath resolves a dotted path (e.g. choices[0].message.content) against decoded JSON.
func GetAtPath(root any, path string) (any, bool) {
	if root == nil || strings.TrimSpace(path) == "" {
		return nil, false
	}
	cur := root
	for _, part := range strings.Split(normalizeJSONPath(path), ".") {
		if part == "" {
			continue
		}
		if idx, err := strconv.Atoi(part); err == nil {
			arr, ok := cur.([]any)
			if !ok || idx < 0 || idx >= len(arr) {
				return nil, false
			}
			cur = arr[idx]
			continue
		}
		m, ok := cur.(map[string]any)
		if !ok {
			return nil, false
		}
		v, ok := m[part]
		if !ok {
			return nil, false
		}
		cur = v
	}
	return cur, true
}

// GetStringAtPath returns a non-empty string at path, if any.
func GetStringAtPath(root map[string]any, path string) (string, bool) {
	v, ok := GetAtPath(root, path)
	if !ok || v == nil {
		return "", false
	}
	s, ok := v.(string)
	if !ok || strings.TrimSpace(s) == "" {
		return "", false
	}
	return s, true
}
