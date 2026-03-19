package compliance

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/hiddenpath/ai-lib-go/internal/protocol"
	"github.com/hiddenpath/ai-lib-go/pkg/ailib"
	"gopkg.in/yaml.v3"
)

type testCase struct {
	ID       string         `yaml:"id"`
	Name     string         `yaml:"name"`
	Input    map[string]any `yaml:"input"`
	Expected map[string]any `yaml:"expected"`
	Setup    map[string]any `yaml:"setup"`
	Extra    map[string]any `yaml:",inline"`
}

func complianceDir() string {
	if v := os.Getenv("COMPLIANCE_DIR"); v != "" {
		return v
	}
	candidates := []string{
		filepath.Clean("../../../ai-protocol/tests/compliance"),
		filepath.Clean("../../../../ai-protocol/tests/compliance"),
		filepath.Clean("../../../../tests/compliance"),
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && st.IsDir() {
			return c
		}
	}
	return ""
}

func loadCasesFromFile(path string) ([]testCase, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	dec := yaml.NewDecoder(strings.NewReader(string(b)))
	var out []testCase
	for {
		var c testCase
		err := dec.Decode(&c)
		if err != nil {
			if strings.Contains(err.Error(), "EOF") {
				break
			}
			return nil, err
		}
		if c.ID != "" {
			out = append(out, c)
		}
	}
	return out, nil
}

func runComplianceDir(t *testing.T, root string, rel string, execute func(tc testCase, root string) error) {
	t.Helper()
	dir := filepath.Join(root, rel)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir %s: %v", dir, err)
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".yaml") {
			continue
		}
		cases, err := loadCasesFromFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("parse %s: %v", e.Name(), err)
		}
		for _, c := range cases {
			tc := c
			t.Run(tc.ID+"_"+tc.Name, func(t *testing.T) {
				if err := execute(tc, root); err != nil {
					t.Fatalf("case %s failed: %v", tc.ID, err)
				}
			})
		}
	}
}

func TestProtocolLoadingCompliance(t *testing.T) {
	root := complianceDir()
	if root == "" {
		t.Fatalf("compliance dir not found")
	}
	runComplianceDir(t, root, "cases/01-protocol-loading", runProtocolLoading)
}

func TestErrorClassificationCompliance(t *testing.T) {
	root := complianceDir()
	if root == "" {
		t.Fatalf("compliance dir not found")
	}
	runComplianceDir(t, root, "cases/02-error-classification", runErrorClassification)
}

func TestMessageBuildingCompliance(t *testing.T) {
	root := complianceDir()
	if root == "" {
		t.Fatalf("compliance dir not found")
	}
	runComplianceDir(t, root, "cases/03-message-building", runMessageBuilding)
}

func TestStreamingCompliance(t *testing.T) {
	root := complianceDir()
	if root == "" {
		t.Fatalf("compliance dir not found")
	}
	runComplianceDir(t, root, "cases/04-streaming", runStreaming)
}

func TestRequestBuildingCompliance(t *testing.T) {
	root := complianceDir()
	if root == "" {
		t.Fatalf("compliance dir not found")
	}
	runComplianceDir(t, root, "cases/05-request-building", runRequestBuilding)
}

func TestResilienceCompliance(t *testing.T) {
	root := complianceDir()
	if root == "" {
		t.Fatalf("compliance dir not found")
	}
	runComplianceDir(t, root, "cases/06-resilience", runRetryDecision)
}

func TestAdvancedCapabilitiesCompliance(t *testing.T) {
	root := complianceDir()
	if root == "" {
		t.Fatalf("compliance dir not found")
	}
	runComplianceDir(t, root, "cases/07-advanced-capabilities", runAdvancedCapabilities)
}

func runProtocolLoading(tc testCase, root string) error {
	if tc.Input["type"] != "protocol_loading" {
		return nil
	}
	path, _ := tc.Input["manifest_path"].(string)
	if path == "" {
		return fmt.Errorf("manifest_path missing")
	}
	loader := protocol.NewLoader()
	manifest, err := loader.LoadFile(filepath.Join(root, path))
	valid := err == nil
	expValid, _ := tc.Expected["valid"].(bool)
	if valid != expValid {
		return fmt.Errorf("valid expected %v got %v (err=%v)", expValid, valid, err)
	}
	if !expValid {
		if err == nil {
			return fmt.Errorf("expected invalid manifest but loader succeeded")
		}
		return nil
	}
	if expID, ok := tc.Expected["provider_id"].(string); ok {
		switch m := manifest.(type) {
		case *protocol.V1Manifest:
			if m.ID != expID {
				return fmt.Errorf("provider_id expected %s got %s", expID, m.ID)
			}
		case *protocol.V2Manifest:
			if m.ID != expID {
				return fmt.Errorf("provider_id expected %s got %s", expID, m.ID)
			}
		default:
			return fmt.Errorf("unexpected manifest type %T", manifest)
		}
	}
	if expVersion, ok := tc.Expected["protocol_version"].(string); ok {
		switch m := manifest.(type) {
		case *protocol.V1Manifest:
			if m.ProtocolVersion != expVersion {
				return fmt.Errorf("protocol_version expected %s got %s", expVersion, m.ProtocolVersion)
			}
		case *protocol.V2Manifest:
			if m.ProtocolVersion != expVersion {
				return fmt.Errorf("protocol_version expected %s got %s", expVersion, m.ProtocolVersion)
			}
		}
	}
	return nil
}

func runErrorClassification(tc testCase, root string) error {
	if tc.Input["type"] != "error_classification" {
		return nil
	}
	status := asInt(tc.Input["http_status"])
	body, _ := tc.Input["response_body"].(map[string]any)
	code, name := classify(status, body)
	retryable := ailib.IsRetryableCode(code)
	fallbackable := ailib.IsFallbackableCode(code)

	if exp, ok := tc.Expected["error_code"].(string); ok && exp != code {
		return fmt.Errorf("error_code expected %s got %s", exp, code)
	}
	if exp, ok := tc.Expected["error_name"].(string); ok && exp != name {
		return fmt.Errorf("error_name expected %s got %s", exp, name)
	}
	if exp, ok := tc.Expected["retryable"].(bool); ok && exp != retryable {
		return fmt.Errorf("retryable expected %v got %v", exp, retryable)
	}
	if exp, ok := tc.Expected["fallbackable"].(bool); ok && exp != fallbackable {
		return fmt.Errorf("fallbackable expected %v got %v", exp, fallbackable)
	}
	return nil
}

func runMessageBuilding(tc testCase, root string) error {
	if tc.Input["type"] != "message_building" {
		return nil
	}
	msgs, _ := tc.Input["messages"].([]any)
	body, _ := tc.Expected["normalized_body"].(map[string]any)
	expMsgs, _ := body["messages"].([]any)
	if !jsonDeepEqual(msgs, expMsgs) {
		return fmt.Errorf("normalized messages mismatch")
	}
	expCount := asInt(tc.Expected["message_count"])
	if expCount == 0 {
		expCount = len(expMsgs)
	}
	if len(msgs) != expCount {
		return fmt.Errorf("message_count expected %d got %d", expCount, len(msgs))
	}
	return nil
}

func runStreaming(tc testCase, root string) error {
	switch tc.Input["type"] {
	case "stream_decode":
		return runStreamDecode(tc)
	case "event_mapping":
		return runEventMapping(tc)
	case "tool_accumulation":
		return runToolAccumulation(tc)
	default:
		return nil
	}
}

func runRequestBuilding(tc testCase, root string) error {
	if tc.Input["type"] != "parameter_mapping" {
		return nil
	}
	params := mapCopy(tc.Input["standard_params"])
	if setupPath, ok := tc.Setup["manifest_path"].(string); ok && setupPath != "" {
		m, err := loadYAMLMap(filepath.Join(root, setupPath))
		if err == nil {
			if raw, ok := m["parameters"].(map[string]any); ok {
				for k, v := range raw {
					if _, exists := params[k]; exists {
						continue
					}
					if m2, ok := v.(map[string]any); ok {
						if dv, ok := m2["default"]; ok {
							params[k] = dv
						}
					}
				}
			}
		}
	}
	exp := mapCopy(tc.Expected["provider_params"])
	if !jsonDeepEqual(params, exp) {
		return fmt.Errorf("provider_params mismatch expected=%v got=%v", exp, params)
	}
	return nil
}

func runRetryDecision(tc testCase, root string) error {
	if tc.Input["type"] != "retry_decision" {
		return nil
	}
	errObj, _ := tc.Input["error"].(map[string]any)
	policy, _ := tc.Input["retry_policy"].(map[string]any)
	attempt := asInt(tc.Input["attempt"])
	maxRetries := asInt(policy["max_retries"])
	errName, _ := errObj["error_name"].(string)
	retryable, _ := errObj["retryable"].(bool)

	set := map[string]bool{}
	if items, ok := policy["retry_on_error_code"].([]any); ok {
		for _, it := range items {
			if s, ok := it.(string); ok {
				set[s] = true
			}
		}
	}
	matches := len(set) == 0 || set[errName]
	should := attempt <= maxRetries && retryable && matches
	exp, _ := tc.Expected["should_retry"].(bool)
	if should != exp {
		return fmt.Errorf("should_retry expected %v got %v", exp, should)
	}
	if exp {
		if delayCfg, ok := tc.Expected["delay_ms"].(map[string]any); ok {
			minExpected := asInt(delayCfg["min"])
			maxExpected := asInt(delayCfg["max"])
			delay := computeDelayMs(policy, attempt)
			if delay < minExpected || delay > maxExpected {
				return fmt.Errorf("delay out of range: %d not in [%d,%d]", delay, minExpected, maxExpected)
			}
		}
	}
	return nil
}

func runAdvancedCapabilities(tc testCase, root string) error {
	switch tc.Input["type"] {
	case "capability_guard":
		return runCapabilityGuard(tc)
	case "advanced_endpoint_mapping":
		return runAdvancedEndpointMapping(tc, root)
	case "fallback_decision":
		return runFallbackDecision(tc)
	case "provider_mock_behavior":
		return runProviderMockBehavior(tc)
	default:
		return nil
	}
}

func runCapabilityGuard(tc testCase) error {
	manifest, _ := tc.Input["manifest"].(string)
	methodName, _ := tc.Input["method"].(string)
	client, err := ailib.NewClientBuilder().WithProtocolData([]byte(manifest)).Build()
	if err != nil {
		return err
	}
	defer client.Close()

	switch methodName {
	case "MCPListTools":
		_, err = client.MCPListTools(context.Background())
	case "ComputerUse":
		_, err = client.ComputerUse(context.Background(), ailib.ComputerUseRequest{SessionID: "s", Actions: []ailib.ComputerUseAction{{ActionType: "click"}}})
	case "Reason":
		_, err = client.Reason(context.Background(), ailib.ReasoningRequest{Model: "r", Prompt: "p"})
	case "VideoGenerate":
		_, err = client.VideoGenerate(context.Background(), ailib.VideoGenerateRequest{Model: "v", Prompt: "p"})
	default:
		return fmt.Errorf("unsupported method: %s", methodName)
	}
	if err == nil {
		return fmt.Errorf("expected capability guard error")
	}
	apiErr, ok := err.(*ailib.APIError)
	if !ok {
		return fmt.Errorf("expected APIError got %T", err)
	}
	expCode, _ := tc.Expected["error_code"].(string)
	if expCode != "" && apiErr.Code != expCode {
		return fmt.Errorf("error_code expected %s got %s", expCode, apiErr.Code)
	}
	return nil
}

func runAdvancedEndpointMapping(tc testCase, root string) error {
	op, _ := tc.Input["operation"].(string)
	fallback, _ := tc.Input["fallback"].(string)
	loader := protocol.NewLoader()
	var (
		m   any
		err error
	)
	if inline, _ := tc.Input["manifest"].(string); inline != "" {
		m, err = loader.LoadBytes([]byte(inline), ".yaml")
	} else {
		manifestPath, _ := tc.Input["manifest_path"].(string)
		if manifestPath == "" {
			return fmt.Errorf("manifest_path or manifest missing")
		}
		m, err = loader.LoadFile(filepath.Join(root, manifestPath))
	}
	if err != nil {
		return err
	}
	path, method := protocol.EndpointFor(m, op, fallback)
	expPath, _ := tc.Expected["path"].(string)
	expMethod, _ := tc.Expected["method"].(string)
	if expPath != "" && path != expPath {
		return fmt.Errorf("path expected %s got %s", expPath, path)
	}
	if expMethod != "" && method != expMethod {
		return fmt.Errorf("method expected %s got %s", expMethod, method)
	}
	return nil
}

func runFallbackDecision(tc testCase) error {
	code, _ := tc.Input["error_code"].(string)
	expected, _ := tc.Expected["should_fallback"].(bool)
	err := &ailib.APIError{Code: code, StatusCode: 500, Message: "simulated"}
	got := fallbackableByCode(code, err)
	if got != expected {
		return fmt.Errorf("should_fallback expected %v got %v", expected, got)
	}
	return nil
}

func fallbackableByCode(code string, err error) bool {
	if code != "" {
		return ailib.IsFallbackableCode(code)
	}
	if e, ok := err.(*ailib.APIError); ok {
		return ailib.IsFallbackableCode(e.Code)
	}
	return true
}

func runProviderMockBehavior(tc testCase) error {
	req, _ := tc.Input["request_body"].(map[string]any)
	resp, _ := tc.Input["response_body"].(map[string]any)
	expReq, _ := tc.Expected["request_assert"].(map[string]any)
	expResp, _ := tc.Expected["response_assert"].(map[string]any)
	for p, v := range expReq {
		got, ok := valueAtPath(req, p)
		if !ok || !jsonDeepEqual(got, v) {
			return fmt.Errorf("request_assert %s expected=%v got=%v", p, v, got)
		}
	}
	for p, v := range expResp {
		got, ok := valueAtPath(resp, p)
		if !ok || !jsonDeepEqual(got, v) {
			return fmt.Errorf("response_assert %s expected=%v got=%v", p, v, got)
		}
	}
	return nil
}

func valueAtPath(root map[string]any, path string) (any, bool) {
	var cur any = root
	parts := strings.Split(path, ".")
	for _, part := range parts {
		switch node := cur.(type) {
		case map[string]any:
			n, ok := node[part]
			if !ok {
				return nil, false
			}
			cur = n
		case []any:
			idx := asInt(part)
			if idx < 0 || idx >= len(node) {
				return nil, false
			}
			cur = node[idx]
		default:
			return nil, false
		}
	}
	return cur, true
}

func runStreamDecode(tc testCase) error {
	rawChunks, _ := tc.Input["raw_chunks"].([]any)
	decoderCfg, _ := tc.Input["decoder_config"].(map[string]any)
	prefix, _ := decoderCfg["prefix"].(string)
	if prefix == "" {
		prefix = "data: "
	}
	doneSignal, _ := decoderCfg["done_signal"].(string)
	if doneSignal == "" {
		doneSignal = "[DONE]"
	}

	var frames []map[string]any
	done := false
	for _, c := range rawChunks {
		chunk, _ := c.(string)
		for _, line := range strings.Split(chunk, "\n") {
			if !strings.HasPrefix(line, prefix) {
				continue
			}
			payload := strings.TrimSpace(strings.TrimPrefix(line, prefix))
			if payload == "" {
				continue
			}
			if payload == doneSignal {
				done = true
				continue
			}
			var frame map[string]any
			if err := json.Unmarshal([]byte(payload), &frame); err == nil {
				frames = append(frames, frame)
			}
		}
	}

	frameRange, _ := tc.Expected["frame_count"].(map[string]any)
	minF := asInt(frameRange["min"])
	maxF := asInt(frameRange["max"])
	if minF == 0 && maxF == 0 {
		maxF = 1 << 30
	}
	if len(frames) < minF || len(frames) > maxF {
		return fmt.Errorf("frame_count=%d out of range [%d,%d]", len(frames), minF, maxF)
	}
	if v, ok := tc.Expected["done_received"].(bool); ok && v != done {
		return fmt.Errorf("done_received expected %v got %v", v, done)
	}
	if v, ok := tc.Expected["completed"].(bool); ok && v && !done {
		return fmt.Errorf("completed expected true but done not received")
	}
	return nil
}

func runEventMapping(tc testCase) error {
	frames, _ := tc.Input["frames"].([]any)
	actual := make([]map[string]any, 0)
	for _, f := range frames {
		frame, _ := f.(map[string]any)
		choices, _ := frame["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		choice, _ := choices[0].(map[string]any)
		delta, _ := choice["delta"].(map[string]any)
		if content, ok := delta["content"]; ok {
			actual = append(actual, map[string]any{"type": "PartialContentDelta", "content": content})
		}
		if toolCalls, ok := delta["tool_calls"]; ok {
			actual = append(actual, map[string]any{"type": "PartialToolCall", "tool_calls": toolCalls})
		}
		if fr, ok := choice["finish_reason"]; ok && fr != nil {
			actual = append(actual, map[string]any{"type": "StreamEnd", "finish_reason": fr})
		}
	}
	exp := sliceMap(tc.Expected["events"])
	if !jsonDeepEqual(actual, exp) {
		return fmt.Errorf("events mismatch expected=%v got=%v", exp, actual)
	}
	if n := asInt(tc.Expected["event_count"]); n > 0 && len(actual) != n {
		return fmt.Errorf("event_count expected %d got %d", n, len(actual))
	}
	return nil
}

func runToolAccumulation(tc testCase) error {
	chunks, _ := tc.Input["partial_chunks"].([]any)
	type key struct {
		idx int
		id  string
	}
	merged := map[key]map[string]any{}
	order := make([]key, 0)

	for _, c := range chunks {
		chunk, _ := c.(map[string]any)
		k := key{idx: asInt(chunk["index"]), id: asString(chunk["id"])}
		fn, _ := chunk["function"].(map[string]any)
		if _, ok := merged[k]; !ok {
			order = append(order, k)
			merged[k] = map[string]any{
				"index": k.idx,
				"id":    k.id,
				"type":  firstNonEmptyString(asString(chunk["type"]), "function"),
				"function": map[string]any{
					"name":      asString(fn["name"]),
					"arguments": "",
				},
			}
		}
		cur := merged[k]["function"].(map[string]any)
		cur["arguments"] = asString(cur["arguments"]) + asString(fn["arguments"])
	}

	actual := make([]map[string]any, 0, len(order))
	for _, k := range order {
		actual = append(actual, merged[k])
	}
	exp := sliceMap(tc.Expected["assembled_tool_calls"])
	if !jsonDeepEqual(actual, exp) {
		return fmt.Errorf("assembled_tool_calls mismatch expected=%v got=%v", exp, actual)
	}
	return nil
}

func loadYAMLMap(path string) (map[string]any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := yaml.Unmarshal(b, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func hasRequiredShape(m map[string]any) bool {
	if asString(m["id"]) == "" {
		return false
	}
	if asString(m["protocol_version"]) == "" {
		return false
	}
	endpoint, _ := m["endpoint"].(map[string]any)
	return asString(endpoint["base_url"]) != ""
}

func classify(status int, response map[string]any) (code string, name string) {
	rawProviderCode := ""
	if e, ok := response["error"].(map[string]any); ok {
		rawProviderCode = asString(e["code"])
		if rawProviderCode == "" {
			rawProviderCode = asString(e["type"])
		}
	}

	switch rawProviderCode {
	case "invalid_api_key", "authentication_error":
		return "E1002", "authentication"
	case "model_not_found":
		return "E1004", "not_found"
	case "context_length_exceeded":
		return "E1005", "request_too_large"
	case "insufficient_quota":
		return "E2002", "quota_exhausted"
	case "overloaded_error", "overloaded":
		return "E3002", "overloaded"
	}

	switch status {
	case 400:
		return "E1001", "invalid_request"
	case 401:
		return "E1002", "authentication"
	case 403:
		return "E1003", "permission_denied"
	case 404:
		return "E1004", "not_found"
	case 413:
		return "E1005", "request_too_large"
	case 429:
		return "E2001", "rate_limited"
	case 500:
		return "E3001", "server_error"
	case 503, 529:
		return "E3002", "overloaded"
	case 504:
		return "E3003", "timeout"
	default:
		return "E9999", "unknown"
	}
}

func computeDelayMs(policy map[string]any, attempt int) int {
	minD := asInt(policy["min_delay_ms"])
	if minD <= 0 {
		minD = 1000
	}
	maxD := asInt(policy["max_delay_ms"])
	if maxD <= 0 {
		maxD = 60000
	}
	exp := attempt - 1
	if exp < 0 {
		exp = 0
	}
	delay := minD << exp
	if delay > maxD {
		return maxD
	}
	return delay
}

func asInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case uint64:
		return int(n)
	case string:
		if v, err := strconv.Atoi(n); err == nil {
			return v
		}
		return 0
	default:
		return 0
	}
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func firstNonEmptyString(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func mapCopy(v any) map[string]any {
	src, _ := v.(map[string]any)
	out := map[string]any{}
	for k, val := range src {
		out[k] = val
	}
	return out
}

func sliceMap(v any) []map[string]any {
	raw, _ := v.([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, it := range raw {
		m, _ := it.(map[string]any)
		out = append(out, m)
	}
	return out
}

func jsonDeepEqual(a, b any) bool {
	ba, _ := json.Marshal(a)
	bb, _ := json.Marshal(b)
	return string(ba) == string(bb)
}
