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
	"time"

	"github.com/ailib-official/ai-lib-go/internal/protocol"
	"github.com/ailib-official/ai-lib-go/internal/resilience"
	"github.com/ailib-official/ai-lib-go/internal/stream"
)

type client struct {
	manifest        any
	baseURL         string
	credential      protocol.ResolvedCredential
	authHeaders     map[string]string
	authQueryParams map[string]string
	headers         map[string]string
	maxRetries      int
	http            *http.Client
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

	credential := protocol.ResolveCredential(manifest, b.apiKey)
	authMetadata := protocol.BuildAuthMetadata(manifest, credential, false)
	if manifest == nil && b.apiKey != "" {
		authMetadata.Headers["Authorization"] = "Bearer " + b.apiKey
	}

	return &client{
		manifest:        manifest,
		baseURL:         baseURL,
		credential:      credential,
		authHeaders:     authMetadata.Headers,
		authQueryParams: authMetadata.QueryParams,
		headers:         b.headers,
		maxRetries:      b.maxRetries,
		http:            httpClient,
	}, nil
}

func (c *client) Close() error {
	c.http.CloseIdleConnections()
	return nil
}

func buildChatPayload(messages []Message, opts *ChatOptions, stream bool) map[string]any {
	payload := map[string]any{
		"messages": messages,
		"stream":   stream,
	}
	if opts == nil {
		return payload
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
	if opts.ResponseFormat != nil {
		payload["response_format"] = opts.ResponseFormat
	}
	if opts.User != "" {
		payload["user"] = opts.User
	}
	if len(opts.Metadata) > 0 {
		payload["metadata"] = opts.Metadata
	}
	return payload
}

func (c *client) Chat(ctx context.Context, messages []Message, opts *ChatOptions) (*ChatResponse, error) {
	if len(messages) == 0 {
		return nil, &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: "messages must not be empty"}
	}
	if opts == nil {
		opts = &ChatOptions{}
	}
	t0 := time.Now()
	payload := buildChatPayload(messages, opts, false)

	path, method := protocol.EndpointFor(c.manifest, "chat_completions", "/chat/completions")
	tTrans := time.Now()
	req, err := c.newRequest(ctx, method, path, payload)
	if err != nil {
		return nil, err
	}
	var out ChatResponse
	attempts, err := c.execute(req, &out)
	if err != nil {
		return nil, err
	}
	tEnd := time.Now()
	modelID := opts.Model
	if modelID == "" {
		modelID = out.Model
	}
	out.ExecutionMetadata = ExecutionMetadata{
		ProviderID:           protocol.ManifestProviderID(c.manifest),
		ModelID:              modelID,
		ExecutionLatencyMs:   uint64(tEnd.Sub(t0).Milliseconds()),
		TranslationLatencyMs: uint64(tTrans.Sub(t0).Milliseconds()),
		MicroRetryCount:      microRetriesFromAttempts(attempts),
		Usage:                chatUsageToExecutionUsage(out.Usage),
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
	payload := buildChatPayload(messages, opts, true)
	path, method := protocol.EndpointFor(c.manifest, "chat_completions", "/chat/completions")
	req, err := c.newRequest(ctx, method, path, payload)
	if err != nil {
		return nil, err
	}
	t0 := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		return nil, parseHTTPError(c.manifest, resp)
	}
	tTrans := time.Now()
	format := "openai_sse"
	if c.manifest != nil {
		format = protocol.StreamingDecoderFormat(c.manifest)
	}
	modelID := opts.Model
	return &sseStream{
		body:             resp.Body,
		decoder:          stream.NewDecoderWithFormat(resp.Body, format),
		started:          t0,
		transDone:        tTrans,
		providerID:       protocol.ManifestProviderID(c.manifest),
		requestedModelID: modelID,
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
	if _, err := c.execute(req, &out); err != nil {
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
	if _, err := c.execute(req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *client) sendJSON(ctx context.Context, method, path string, payload any, out any) error {
	req, err := c.newRequest(ctx, method, path, payload)
	if err != nil {
		return err
	}
	_, err = c.execute(req, out)
	return err
}

func (c *client) newRequest(ctx context.Context, method, path string, payload any) (*http.Request, error) {
	if err := validateRequestMeta(method, path); err != nil {
		return nil, err
	}
	var bodyBytes []byte
	var body io.Reader
	if payload != nil {
		b, err := json.Marshal(payload)
		if err != nil {
			return nil, err
		}
		bodyBytes = b
		body = bytes.NewReader(bodyBytes)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, body)
	if err != nil {
		return nil, err
	}
	if len(bodyBytes) > 0 {
		req.ContentLength = int64(len(bodyBytes))
		req.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader(bodyBytes)), nil
		}
	}
	if len(c.authQueryParams) > 0 {
		q := req.URL.Query()
		for k, v := range c.authQueryParams {
			q.Set(k, v)
		}
		req.URL.RawQuery = q.Encode()
	}
	req.Header.Set("Content-Type", "application/json")
	c.setHeaders(req)
	return req, nil
}

func (c *client) setHeaders(req *http.Request) {
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}
	for k, v := range c.authHeaders {
		req.Header.Set(k, v)
	}
}

func (c *client) execute(req *http.Request, out any) (attempts int, err error) {
	ctx := req.Context()
	p := resilience.DefaultPolicy()
	if c.maxRetries > 0 {
		p.MaxAttempts = c.maxRetries
	} else if m, ok := protocol.RetryMaxAttempts(c.manifest); ok {
		p.MaxAttempts = m
	}

	return resilience.ExecuteAttempts(ctx, p, func(_ context.Context) error {
		if req.GetBody != nil {
			rc, err := req.GetBody()
			if err != nil {
				return err
			}
			req.Body = rc
		}
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
	if c, ok := protocol.ClassifyProviderError(manifest, pCode, pType); ok {
		code = c
	} else if pCode != "" || pType != "" {
		if c, ok := classifyProviderErrorCode(code, b); ok {
			code = c
		}
	} else if c, ok := protocol.ClassifyHTTPStatusError(manifest, resp.StatusCode); ok {
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

// ClassifyHTTPResponse applies production manifest-aware HTTP error classification.
func ClassifyHTTPResponse(manifest any, status int, body []byte) string {
	code := classifyStatus(status)
	pCode, pType := extractProviderErrorTokens(body)
	if c, ok := protocol.ClassifyProviderError(manifest, pCode, pType); ok {
		return c
	}
	if pCode != "" || pType != "" {
		if c, ok := classifyProviderErrorCode(code, body); ok {
			return c
		}
	}
	if c, ok := protocol.ClassifyHTTPStatusError(manifest, status); ok {
		return c
	}
	return code
}

// StandardErrorName returns the protocol error class name for a standard code (e.g. E2001 → rate_limited).
func StandardErrorName(code string) string {
	switch code {
	case ErrInvalidRequest:
		return "invalid_request"
	case ErrAuthentication:
		return "authentication"
	case ErrPermission:
		return "permission_denied"
	case ErrNotFound:
		return "not_found"
	case ErrUnsupported:
		return "request_too_large"
	case ErrRateLimited:
		return "rate_limited"
	case ErrQuotaExhausted:
		return "quota_exhausted"
	case ErrServerError:
		return "server_error"
	case ErrOverloaded:
		return "overloaded"
	case ErrTimeout:
		return "timeout"
	case ErrConflict:
		return "conflict"
	case ErrCancelled:
		return "cancelled"
	default:
		return "unknown"
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
		return "", false
	}
}

func microRetriesFromAttempts(attempts int) uint8 {
	if attempts <= 1 {
		return 0
	}
	r := attempts - 1
	if r > 255 {
		return 255
	}
	return uint8(r)
}

func chatUsageToExecutionUsage(u *Usage) *ExecutionUsage {
	if u == nil {
		return nil
	}
	out := &ExecutionUsage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
	if u.ReasoningTokens != 0 {
		v := u.ReasoningTokens
		out.ReasoningTokens = &v
	}
	if u.CacheReadTokens != 0 {
		v := u.CacheReadTokens
		out.CacheReadTokens = &v
	}
	if u.CacheCreationTokens != 0 {
		v := u.CacheCreationTokens
		out.CacheCreationTokens = &v
	}
	return out
}

func usageFromMap(raw map[string]any) *Usage {
	if raw == nil {
		return nil
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return nil
	}
	var u Usage
	if err := json.Unmarshal(b, &u); err != nil {
		return nil
	}
	return &u
}

type sseStream struct {
	body             io.ReadCloser
	decoder          *stream.Decoder
	current          StreamingEvent
	err              error
	started          time.Time
	transDone        time.Time
	providerID       string
	requestedModelID string
	closed           bool
	meta             ExecutionMetadata
	metaFilled       bool
	lastUsage        map[string]any
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
	if ev.Usage != nil {
		s.lastUsage = ev.Usage
	}
	return true
}

func (s *sseStream) Event() StreamingEvent { return s.current }
func (s *sseStream) Err() error            { return s.err }

func (s *sseStream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	err := s.body.Close()
	s.fillExecutionMetadata(time.Now())
	return err
}

func (s *sseStream) fillExecutionMetadata(end time.Time) {
	if s.metaFilled {
		return
	}
	s.metaFilled = true
	s.meta = ExecutionMetadata{
		ProviderID:           s.providerID,
		ModelID:              s.requestedModelID,
		ExecutionLatencyMs:   uint64(end.Sub(s.started).Milliseconds()),
		TranslationLatencyMs: uint64(s.transDone.Sub(s.started).Milliseconds()),
		MicroRetryCount:      0,
	}
	if s.lastUsage != nil {
		if u := usageFromMap(s.lastUsage); u != nil {
			s.meta.Usage = chatUsageToExecutionUsage(u)
		}
	}
}

func (s *sseStream) ExecutionMetadata() (ExecutionMetadata, bool) {
	if !s.closed {
		return ExecutionMetadata{}, false
	}
	return s.meta, s.metaFilled
}
