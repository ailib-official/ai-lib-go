// Experimental generative L-Exec HTTP (ALG-GEN-002 / PT-GEN).
// 生成式 L-Exec：经现有 Client HTTP 栈；adapter 来自 manifest。
package ailib

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"mime/multipart"
	"os"
	"path/filepath"
	"strings"

	"github.com/ailib-official/ai-lib-go/internal/protocol"
	"github.com/ailib-official/ai-lib-go/internal/resilience"
)

// OpenAIImageBody builds OpenAI Images JSON.
func OpenAIImageBody(req ImageGenerationRequest) map[string]any {
	body := map[string]any{"model": req.Model, "prompt": req.Prompt}
	if req.Size != nil {
		body["size"] = *req.Size
	}
	if req.N != nil {
		body["n"] = *req.N
	}
	if req.ResponseFormat != nil {
		body["response_format"] = *req.ResponseFormat
	}
	return body
}

// DashscopeImageBody builds DashScope multimodal-generation JSON (PT-GEN-003).
func DashscopeImageBody(req ImageGenerationRequest) map[string]any {
	return map[string]any{
		"model": req.Model,
		"input": map[string]any{
			"messages": []map[string]any{
				{"role": "user", "content": []map[string]any{{"text": req.Prompt}}},
			},
		},
	}
}

// ParseOpenAIImage parses OpenAI-style image payloads.
func ParseOpenAIImage(model string, payload map[string]any) ImageGenerationResult {
	out := ImageGenerationResult{Model: model, Images: nil}
	data, _ := payload["data"].([]any)
	for _, item := range data {
		row, ok := item.(map[string]any)
		if !ok {
			continue
		}
		img := GeneratedImage{}
		if u, ok := row["url"].(string); ok {
			img.URL = &u
		}
		if b, ok := row["b64_json"].(string); ok {
			img.B64JSON = &b
		}
		out.Images = append(out.Images, img)
	}
	return out
}

// ParseDashscopeImage parses DashScope multimodal / results shapes.
func ParseDashscopeImage(model string, payload map[string]any) ImageGenerationResult {
	out := ImageGenerationResult{Model: model}
	if output, ok := payload["output"].(map[string]any); ok {
		if choices, ok := output["choices"].([]any); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]any); ok {
				if msg, ok := choice["message"].(map[string]any); ok {
					if content, ok := msg["content"].([]any); ok && len(content) > 0 {
						if block, ok := content[0].(map[string]any); ok {
							if u, ok := block["image"].(string); ok {
								out.Images = append(out.Images, GeneratedImage{URL: &u})
								return out
							}
						}
					}
				}
			}
		}
		if results, ok := output["results"].([]any); ok && len(results) > 0 {
			if row, ok := results[0].(map[string]any); ok {
				if u, ok := row["url"].(string); ok {
					out.Images = append(out.Images, GeneratedImage{URL: &u})
				}
			}
		}
	}
	return out
}

func (c *client) GenerateImage(ctx context.Context, req ImageGenerationRequest) (*ImageGenerationResult, error) {
	ep, err := protocol.RequireGenerativeEndpoint(c.manifest, req.Model, protocol.KeyImageGeneration)
	if err != nil {
		return nil, &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: err.Error()}
	}
	var body map[string]any
	if ep.Adapter == "dashscope" {
		body = DashscopeImageBody(req)
	} else {
		body = OpenAIImageBody(req)
	}
	var raw map[string]any
	if err := c.sendJSON(ctx, ep.Method, ep.Path, body, &raw); err != nil {
		return nil, err
	}
	var result ImageGenerationResult
	if ep.Adapter == "dashscope" {
		result = ParseDashscopeImage(req.Model, raw)
	} else {
		result = ParseOpenAIImage(req.Model, raw)
	}
	return &result, nil
}

func (c *client) TranscribeSpeech(ctx context.Context, req SpeechToTextRequest) (*SpeechToTextResult, error) {
	ep, err := protocol.RequireGenerativeEndpoint(c.manifest, req.Model, protocol.KeySpeechToText)
	if err != nil {
		return nil, &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: err.Error()}
	}
	if ep.Adapter != "openai" {
		return nil, &APIError{
			Code:       ErrInvalidRequest,
			StatusCode: 400,
			Message:    fmt.Sprintf("speech_to_text adapter %q not implemented in ALG-GEN-002 (openai only)", ep.Adapter),
		}
	}
	audio, filename, err := loadSpeechAudio(req)
	if err != nil {
		return nil, &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: err.Error()}
	}
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	part, err := w.CreateFormFile("file", filename)
	if err != nil {
		return nil, err
	}
	if _, err := part.Write(audio); err != nil {
		return nil, err
	}
	if err := w.WriteField("model", req.Model); err != nil {
		return nil, err
	}
	if req.Language != nil {
		_ = w.WriteField("language", *req.Language)
	}
	if req.Prompt != nil {
		_ = w.WriteField("prompt", *req.Prompt)
	}
	contentType := w.FormDataContentType()
	if err := w.Close(); err != nil {
		return nil, err
	}
	bodyBytes := buf.Bytes()
	httpReq, err := c.newMultipartRequest(ctx, ep.Method, ep.Path, bodyBytes, contentType)
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	if _, err := c.execute(httpReq, &payload); err != nil {
		return nil, err
	}
	text, _ := payload["text"].(string)
	return &SpeechToTextResult{Model: req.Model, Text: text}, nil
}

func (c *client) SynthesizeSpeech(ctx context.Context, req TextToSpeechRequest) (*TextToSpeechResult, error) {
	ep, err := protocol.RequireGenerativeEndpoint(c.manifest, req.Model, protocol.KeyTextToSpeech)
	if err != nil {
		return nil, &APIError{Code: ErrInvalidRequest, StatusCode: 400, Message: err.Error()}
	}
	if ep.Adapter != "openai" {
		return nil, &APIError{
			Code:       ErrInvalidRequest,
			StatusCode: 400,
			Message:    fmt.Sprintf("text_to_speech adapter %q not implemented in ALG-GEN-002 (openai only)", ep.Adapter),
		}
	}
	body := map[string]any{"model": req.Model, "input": req.Input}
	if req.Voice != nil {
		body["voice"] = *req.Voice
	}
	if req.ResponseFormat != nil {
		body["response_format"] = *req.ResponseFormat
	}
	httpReq, err := c.newRequest(ctx, ep.Method, ep.Path, body)
	if err != nil {
		return nil, err
	}
	var audio []byte
	var contentType string
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
		audio = b
		contentType = resp.Header.Get("Content-Type")
		return nil
	}, isRetryableErr)
	if err != nil {
		return nil, err
	}
	b64 := base64.StdEncoding.EncodeToString(audio)
	out := &TextToSpeechResult{Model: req.Model, AudioBase64: &b64}
	if contentType != "" {
		out.ContentType = &contentType
	}
	return out, nil
}

func loadSpeechAudio(req SpeechToTextRequest) ([]byte, string, error) {
	if len(req.Audio) > 0 {
		name := "audio.wav"
		if req.AudioSource != "" {
			name = filepath.Base(req.AudioSource)
		}
		return req.Audio, name, nil
	}
	if strings.TrimSpace(req.AudioSource) == "" {
		return nil, "", fmt.Errorf("SpeechToTextRequest requires Audio bytes or AudioSource path")
	}
	b, err := os.ReadFile(req.AudioSource)
	if err != nil {
		return nil, "", err
	}
	return b, filepath.Base(req.AudioSource), nil
}

// STTTranscribe prefers PT-GEN speech_to_text when declared; else legacy stt + audio_transcriptions.
func (c *client) STTTranscribe(ctx context.Context, req STTRequest) (*STTResponse, error) {
	if protocol.SupportsGenerativeForModel(c.manifest, req.Model, protocol.KeySpeechToText) {
		genReq := SpeechToTextRequest{Model: req.Model, AudioSource: req.File}
		if req.Language != "" {
			lang := req.Language
			genReq.Language = &lang
		}
		res, err := c.TranscribeSpeech(ctx, genReq)
		if err != nil {
			return nil, err
		}
		return &STTResponse{Text: res.Text}, nil
	}
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

// TTSSpeak prefers PT-GEN text_to_speech when declared; else legacy tts + audio_speech.
func (c *client) TTSSpeak(ctx context.Context, req TTSRequest) (*TTSResponse, error) {
	if protocol.SupportsGenerativeForModel(c.manifest, req.Model, protocol.KeyTextToSpeech) {
		genReq := TextToSpeechRequest{Model: req.Model, Input: req.Input}
		if req.Voice != "" {
			v := req.Voice
			genReq.Voice = &v
		}
		if req.Format != "" {
			f := req.Format
			genReq.ResponseFormat = &f
		}
		res, err := c.SynthesizeSpeech(ctx, genReq)
		if err != nil {
			return nil, err
		}
		var audio []byte
		if res.AudioBase64 != nil {
			decoded, decErr := base64.StdEncoding.DecodeString(*res.AudioBase64)
			if decErr != nil {
				return nil, decErr
			}
			audio = decoded
		}
		mime := ""
		if res.ContentType != nil {
			mime = *res.ContentType
		}
		return &TTSResponse{AudioData: audio, MimeType: mime}, nil
	}
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
