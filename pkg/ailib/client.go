// Package ailib client implementation.
// 客户端实现，基于协议驱动统一请求构造。
package ailib

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/ailib-official/ai-lib-go/internal/protocol"
	"github.com/ailib-official/ai-lib-go/internal/resilience"
	"github.com/ailib-official/ai-lib-go/internal/stream"
)

type client struct {
	manifest   any
	baseURL    string
	apiKey     string
	headers    map[string]string
	maxRetries int
	http       *http.Client
}

func newClient(b *ClientBuilder, httpClient *http.Client) (*client, error) {
	l := protocol.NewLoader()
	var manifest any
	var err error

	if len(b.protocolData) > 0 {
		manifest, err = l.LoadBytes(b.protocolData, ".yaml")
		if err != nil {
			return nil, err
		}
	} else if b.protocolPath != "" {
		manifest, err = l.LoadFile(b.protocolPath)
		if err != nil {
			return nil, err
		}
	}

	baseURL := b.baseURL
	if baseURL == "" && manifest != nil {
		baseURL, err = protocol.BaseURL(manifest)
		if err != nil {
			return nil, err
		}
	}
	baseURL = strings.TrimRight(baseURL, "/")
	if baseURL == "" {
		return nil, fmt.Errorf("resolved baseURL is empty")
	}

	return &client{
		manifest:   manifest,
		baseURL:    baseURL,
		apiKey:     b.apiKey,
		headers:    b.headers,
		maxRetries: b.maxRetries,
		http:       httpClient,
	}, nil
}

func (c *client) Close() error {
	c.http.CloseIdleConnections()
	return nil
}

func (c *client) Chat(ctx context.Context, messages []Message, opts *ChatOptions) (*ChatResponse, error) {
	if len(messages) == 0 {
		return nil, &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: "messages must not be empty"}
	}
	if opts == nil {
		opts = &ChatOptions{}
	}
	payload := map[string]any{
		"messages": messages,
		"stream":   false,
	}
	if opts.Model != "" {
		payload["model"] = opts.Model
	}
	if opts.Temperature != nil {
		payload["temperature"] = *opts.Temperature
	}
	if opts.MaxTokens != nil {
		payload["max_tokens"] = *opts.MaxTokens
	}
	if opts.TopP != nil {
		payload["top_p"] = *opts.TopP
	}
	if len(opts.Tools) > 0 {
		payload["tools"] = opts.Tools
	}
	if opts.ToolChoice != nil {
		payload["tool_choice"] = opts.ToolChoice
	}

	path, method := protocol.EndpointFor(c.manifest, "chat_completions", "/chat/completions")
	var out ChatResponse
	if err := c.sendJSON(ctx, method, path, payload, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) ChatStream(ctx context.Context, messages []Message, opts *ChatOptions) (Stream, error) {
	if len(messages) == 0 {
		return nil, &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: "messages must not be empty"}
	}
	if opts == nil {
		opts = &ChatOptions{}
	}
	payload := map[string]any{
		"messages": messages,
		"stream":   true,
	}
	if opts.Model != "" {
		payload["model"] = opts.Model
	}
	path, method := protocol.EndpointFor(c.manifest, "chat_completions", "/chat/completions")
	req, err := c.newRequest(ctx, method, path, payload)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		return nil, parseHTTPError(c.manifest, resp)
	}
	format := "openai_sse"
	if c.manifest != nil {
		format = protocol.StreamingDecoderFormat(c.manifest)
	}
	return &sseStream{
		body:    resp.Body,
		decoder: stream.NewDecoderWithFormat(resp.Body, format),
	}, nil
}

func (c *client) Embeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	if err := c.requireCapability("embeddings"); err != nil {
		return nil, err
	}
	path, method := protocol.EndpointFor(c.manifest, "embeddings", "/embeddings")
	var out EmbeddingResponse
	if err := c.sendJSON(ctx, method, path, req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) BatchCreate(ctx context.Context, req BatchCreateRequest) (*BatchJob, error) {
	if err := c.requireCapability("batch"); err != nil {
		return nil, err
	}
	path, method := protocol.EndpointFor(c.manifest, "batch_create", "/batches")
	var out BatchJob
	if err := c.sendJSON(ctx, method, path, req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) BatchGet(ctx context.Context, batchID string) (*BatchJob, error) {
	if err := c.requireCapability("batch"); err != nil {
		return nil, err
	}
	if strings.TrimSpace(batchID) == "" {
		return nil, &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: "batchID must not be empty"}
	}
	pathTpl, method := protocol.EndpointFor(c.manifest, "batch_get", "/batches/{id}")
	path := strings.ReplaceAll(pathTpl, "{id}", batchID)
	req, err := c.newRequest(ctx, method, path, nil)
	if err != nil {
		return nil, err
	}
	var out BatchJob
	if err := c.execute(req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) BatchCancel(ctx context.Context, batchID string) (*BatchJob, error) {
	if err := c.requireCapability("batch"); err != nil {
		return nil, err
	}
	if strings.TrimSpace(batchID) == "" {
		return nil, &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: "batchID must not be empty"}
	}
	pathTpl, method := protocol.EndpointFor(c.manifest, "batch_cancel", "/batches/{id}/cancel")
	path := strings.ReplaceAll(pathTpl, "{id}", batchID)
	var out BatchJob
	if err := c.sendJSON(ctx, method, path, map[string]any{}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) STTTranscribe(ctx context.Context, req STTRequest) (*STTResponse, error) {
	if err := c.requireCapability("stt"); err != nil {
		return nil, err
	}
	path, method := protocol.EndpointFor(c.manifest, "audio_transcriptions", "/audio/transcriptions")
	var out STTResponse
	if err := c.sendJSON(ctx, method, path, req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) TTSSpeak(ctx context.Context, req TTSRequest) (*TTSResponse, error) {
	if err := c.requireCapability("tts"); err != nil {
		return nil, err
	}
	path, method := protocol.EndpointFor(c.manifest, "audio_speech", "/audio/speech")
	httpReq, err := c.newRequest(ctx, method, path, req)
	if err != nil {
		return nil, err
	}

	var out TTSResponse
	err = resilience.Execute(ctx, resilience.DefaultPolicy(), func(_ context.Context) error {
		resp, reqErr := c.http.Do(httpReq)
		if reqErr != nil {
			return reqErr
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			return parseHTTPError(c.manifest, resp)
		}
		b, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return readErr
		}
		out = TTSResponse{
			AudioData: b,
			MimeType:  resp.Header.Get("Content-Type"),
		}
		return nil
	}, isRetryableErr)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) Rerank(ctx context.Context, req RerankRequest) (*RerankResponse, error) {
	if err := c.requireCapability("reranking"); err != nil {
		return nil, err
	}
	path, method := protocol.EndpointFor(c.manifest, "rerank", "/rerank")
	var out RerankResponse
	if err := c.sendJSON(ctx, method, path, req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) MCPListTools(ctx context.Context) (*MCPListToolsResponse, error) {
	if err := c.requireCapability("mcp"); err != nil {
		return nil, err
	}
	path, method := protocol.EndpointFor(c.manifest, "mcp_list_tools", "/mcp/tools/list")
	var out MCPListToolsResponse
	if err := c.sendJSON(ctx, method, path, map[string]any{}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) MCPCallTool(ctx context.Context, req MCPCallToolRequest) (*MCPCallToolResponse, error) {
	if err := c.requireCapability("mcp"); err != nil {
		return nil, err
	}
	path, method := protocol.EndpointFor(c.manifest, "mcp_call_tool", "/mcp/tools/call")
	var out MCPCallToolResponse
	if err := c.sendJSON(ctx, method, path, req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) ComputerUse(ctx context.Context, req ComputerUseRequest) (*ComputerUseResponse, error) {
	if err := c.requireCapability("computer_use"); err != nil {
		return nil, err
	}
	path, method := protocol.EndpointFor(c.manifest, "computer_use", "/computer-use/actions")
	var out ComputerUseResponse
	if err := c.sendJSON(ctx, method, path, req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) Reason(ctx context.Context, req ReasoningRequest) (*ReasoningResponse, error) {
	if err := c.requireCapability("reasoning"); err != nil {
		return nil, err
	}
	path, method := protocol.EndpointFor(c.manifest, "reasoning", "/reasoning")
	var out ReasoningResponse
	if err := c.sendJSON(ctx, method, path, req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) VideoGenerate(ctx context.Context, req VideoGenerateRequest) (*VideoJob, error) {
	if err := c.requireCapability("video"); err != nil {
		return nil, err
	}
	path, method := protocol.EndpointFor(c.manifest, "video_generate", "/video/generations")
	var out VideoJob
	if err := c.sendJSON(ctx, method, path, req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) VideoGet(ctx context.Context, jobID string) (*VideoJob, error) {
	if err := c.requireCapability("video"); err != nil {
		return nil, err
	}
	if strings.TrimSpace(jobID) == "" {
		return nil, &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: "jobID must not be empty"}
	}
	pathTpl, method := protocol.EndpointFor(c.manifest, "video_get", "/video/generations/{id}")
	path := strings.ReplaceAll(pathTpl, "{id}", jobID)
	req, err := c.newRequest(ctx, method, path, nil)
	if err != nil {
		return nil, err
	}
	var out VideoJob
	if err := c.execute(req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) sendJSON(ctx context.Context, method, path string, payload any, out any) error {
	req, err := c.newRequest(ctx, method, path, payload)
	if err != nil {
		return err
	}
	return c.execute(req, out)
}

func (c *client) newRequest(ctx context.Context, method, path string, payload any) (*http.Request, error) {
	if err := validateRequestMeta(method, path); err != nil {
		return nil, err
	}
	var body io.Reader
	if payload != nil {
		b, err := json.Marshal(payload)
		if err != nil {
			return nil, err
		}
		body = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	c.setHeaders(req)
	return req, nil
}

func (c *client) setHeaders(req *http.Request) {
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}
	if c.apiKey == "" {
		return
	}
	name := "Authorization"
	prefix := "Bearer "
	if c.manifest != nil {
		if h, p, err := protocol.AuthHeader(c.manifest); err == nil {
			name, prefix = h, p
		}
	}
	req.Header.Set(name, prefix+c.apiKey)
}

func (c *client) execute(req *http.Request, out any) error {
	ctx := req.Context()
	p := resilience.DefaultPolicy()
	if c.maxRetries > 0 {
		p.MaxAttempts = c.maxRetries
	} else if m, ok := protocol.RetryMaxAttempts(c.manifest); ok {
		p.MaxAttempts = m
	}

	return resilience.Execute(ctx, p, func(_ context.Context) error {
		resp, err := c.http.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			return parseHTTPError(c.manifest, resp)
		}
		if out == nil {
			return nil
		}
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(body, out); err != nil {
			return err
		}
		if cr, ok := out.(*ChatResponse); ok {
			var raw map[string]any
			if err := json.Unmarshal(body, &raw); err == nil {
				EnrichNonstreamChatResponse(c.manifest, raw, cr)
			}
		}
		return nil
	}, isRetryableErr)
}

func parseHTTPError(manifest any, resp *http.Response) error {
	b, _ := io.ReadAll(resp.Body)
	body := strings.TrimSpace(string(b))
	code := classifyStatus(resp.StatusCode)
	pCode, pType := extractProviderErrorTokens(b)
	if c, ok := protocol.ClassifyError(manifest, resp.StatusCode, pCode, pType); ok {
		code = c
	} else if c, ok := classifyProviderErrorCode(code, b); ok {
		code = c
	}
	return &APIError{
		Code:       code,
		StatusCode: resp.StatusCode,
		Message:    body,
	}
}

func extractProviderErrorTokens(body []byte) (code, kind string) {
	type nestedError struct {
		Code string `json:"code"`
		Type string `json:"type"`
	}
	type errorBody struct {
		Error nestedError `json:"error"`
	}
	var parsed errorBody
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", ""
	}
	return strings.TrimSpace(parsed.Error.Code), strings.TrimSpace(parsed.Error.Type)
}

func (c *client) requireCapability(name string) error {
	if c.manifest == nil {
		return nil
	}
	if protocol.HasCapability(c.manifest, name) {
		return nil
	}
	return &APIError{
		Code:       ErrUnsupported,
		StatusCode: 400,
		Message:    "capability not declared in protocol manifest: " + name,
	}
}

func isRetryableErr(err error) bool {
	e, ok := err.(*APIError)
	if !ok {
		return true
	}
	return IsRetryableCode(e.Code)
}

func isFallbackableErr(err error) bool {
	e, ok := err.(*APIError)
	if !ok {
		return true
	}
	return IsFallbackableCode(e.Code)
}

func validateRequestMeta(method, path string) error {
	if path == "" || !strings.HasPrefix(path, "/") {
		return &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: "endpoint path must start with /"}
	}
	switch strings.ToUpper(method) {
	case "GET", "POST", "DELETE":
		return nil
	default:
		return &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: "unsupported HTTP method: " + method}
	}
}

func classifyProviderErrorCode(defaultCode string, body []byte) (string, bool) {
	type nestedError struct {
		Code string `json:"code"`
		Type string `json:"type"`
	}
	type errorBody struct {
		Error nestedError `json:"error"`
	}
	var parsed errorBody
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", false
	}
	raw := strings.TrimSpace(parsed.Error.Code)
	if raw == "" {
		raw = strings.TrimSpace(parsed.Error.Type)
	}
	switch raw {
	case "invalid_api_key", "authentication_error":
		return ErrAuthentication, true
	case "model_not_found":
		return ErrNotFound, true
	case "context_length_exceeded":
		return ErrUnsupported, true
	case "insufficient_quota":
		return ErrQuotaExhausted, true
	case "overloaded", "overloaded_error":
		return ErrOverloaded, true
	case "conflict":
		return ErrConflict, true
	case "cancelled":
		return ErrCancelled, true
	default:
		return defaultCode, true
	}
}

type sseStream struct {
	body    io.ReadCloser
	decoder *stream.Decoder
	current StreamingEvent
	err     error
}

func (s *sseStream) Next() bool {
	ev, ok, err := s.decoder.Next()
	if err != nil {
		s.err = err
		return false
	}
	if !ok {
		return false
	}
	s.current = StreamingEvent{
		Type:         ev.Type,
		Delta:        ev.Delta,
		FinishReason: ev.FinishReason,
	}
	return true
}

func (s *sseStream) Event() StreamingEvent { return s.current }
func (s *sseStream) Err() error            { return s.err }
func (s *sseStream) Close() error          { return s.body.Close() }
