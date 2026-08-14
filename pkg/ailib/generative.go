// Experimental generative request/response types (ALG-GEN-001 / PT-GEN-001).
// 生成式请求类型：与 Rust/Python/TS 同键；HTTP driver 见 ALG-GEN-002（本任务 defer）。
package ailib

type ImageGenerationRequest struct {
	Model          string  `json:"model"`
	Prompt         string  `json:"prompt"`
	Size           *string `json:"size,omitempty"`
	N              *int    `json:"n,omitempty"`
	ResponseFormat *string `json:"response_format,omitempty"`
}

type GeneratedImage struct {
	URL     *string `json:"url,omitempty"`
	B64JSON *string `json:"b64_json,omitempty"`
}

type ImageGenerationResult struct {
	Model  string           `json:"model"`
	Images []GeneratedImage `json:"images"`
}

type SpeechToTextRequest struct {
	Model       string  `json:"model"`
	AudioSource string  `json:"audio_source"`
	Language    *string `json:"language,omitempty"`
	Prompt      *string `json:"prompt,omitempty"`
}

type SpeechToTextResult struct {
	Model string `json:"model"`
	Text  string `json:"text"`
}

type TextToSpeechRequest struct {
	Model          string  `json:"model"`
	Input          string  `json:"input"`
	Voice          *string `json:"voice,omitempty"`
	ResponseFormat *string `json:"response_format,omitempty"`
}

type TextToSpeechResult struct {
	Model       string  `json:"model"`
	AudioBase64 *string `json:"audio_base64,omitempty"`
	ContentType *string `json:"content_type,omitempty"`
}
