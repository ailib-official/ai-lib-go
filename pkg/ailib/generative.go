// Experimental generative request/response types (ALG-GEN-001 / PT-GEN-001).
// 生成式请求类型：与 Rust/Python/TS 同键；HTTP L-Exec 见 generative_lexec.go（ALG-GEN-002）。
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
	AudioSource string  `json:"audio_source,omitempty"`
	Audio       []byte  `json:"-"` // preferred over path for in-memory / tests
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
