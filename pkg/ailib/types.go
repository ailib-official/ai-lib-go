// Package ailib exposes the public runtime API.
// ai-lib-go 公共接口，提供跨厂商统一能力入口。
package ailib

import "context"

type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleTool      MessageRole = "tool"
)

type Message struct {
	Role       MessageRole    `json:"role"`
	Content    any            `json:"content"`
	Name       string         `json:"name,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall     `json:"tool_calls,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ToolDefinition struct {
	Type     string       `json:"type"`
	Function FunctionSpec `json:"function"`
}

type FunctionSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type ChatOptions struct {
	Model          string           `json:"model"`
	Temperature    *float64         `json:"temperature,omitempty"`
	MaxTokens      *int             `json:"max_tokens,omitempty"`
	TopP           *float64         `json:"top_p,omitempty"`
	Tools          []ToolDefinition `json:"tools,omitempty"`
	ToolChoice     any              `json:"tool_choice,omitempty"`
	ResponseFormat map[string]any   `json:"response_format,omitempty"`
	User           string           `json:"user,omitempty"`
	Metadata       map[string]any   `json:"metadata,omitempty"`
}

type ChatResponse struct {
	ID      string   `json:"id"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

type Usage struct {
	PromptTokens      int `json:"prompt_tokens"`
	CompletionTokens  int `json:"completion_tokens"`
	TotalTokens       int `json:"total_tokens"`
	ReasoningTokens   int `json:"reasoning_tokens,omitempty"`
	CacheReadTokens   int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens  int `json:"cache_write_tokens,omitempty"`
}

type StreamingEvent struct {
	Type         string      `json:"type"`
	Delta        string      `json:"delta,omitempty"`
	Thinking     string      `json:"thinking,omitempty"`
	FinishReason string      `json:"finish_reason,omitempty"`
	ToolCall     any         `json:"tool_call,omitempty"`
	ToolCallID   string      `json:"tool_call_id,omitempty"`
	ToolName     string      `json:"tool_name,omitempty"`
	Arguments    string      `json:"arguments,omitempty"`
	Index        *int        `json:"index,omitempty"`
	IsComplete   bool        `json:"is_complete,omitempty"`
	Usage        *Usage      `json:"usage,omitempty"`
	Error        *EventError `json:"error,omitempty"`
}

type EventError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type Stream interface {
	Next() bool
	Event() StreamingEvent
	Err() error
	Close() error
}

type Client interface {
	Chat(ctx context.Context, messages []Message, opts *ChatOptions) (*ChatResponse, error)
	ChatStream(ctx context.Context, messages []Message, opts *ChatOptions) (Stream, error)

	Embeddings(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error)
	BatchCreate(ctx context.Context, req BatchCreateRequest) (*BatchJob, error)
	BatchGet(ctx context.Context, batchID string) (*BatchJob, error)
	BatchCancel(ctx context.Context, batchID string) (*BatchJob, error)

	STTTranscribe(ctx context.Context, req STTRequest) (*STTResponse, error)
	TTSSpeak(ctx context.Context, req TTSRequest) (*TTSResponse, error)
	Rerank(ctx context.Context, req RerankRequest) (*RerankResponse, error)
	MCPListTools(ctx context.Context) (*MCPListToolsResponse, error)
	MCPCallTool(ctx context.Context, req MCPCallToolRequest) (*MCPCallToolResponse, error)
	ComputerUse(ctx context.Context, req ComputerUseRequest) (*ComputerUseResponse, error)
	Reason(ctx context.Context, req ReasoningRequest) (*ReasoningResponse, error)

	VideoGenerate(ctx context.Context, req VideoGenerateRequest) (*VideoJob, error)
	VideoGet(ctx context.Context, jobID string) (*VideoJob, error)

	Close() error
}
