# ai-lib-go

**Protocol runtime for [AI-Protocol](https://github.com/ailib-official/ai-protocol)** — Go reference implementation (**v1.0.0**, Go 1.21+).

`ai-lib-go` splits execution and policy across two public packages:

| Package | Layer | Role |
|---------|-------|------|
| `github.com/ailib-official/ai-lib-go/pkg/ailib` | Execution (E) | `Client`, manifest-aware HTTP chat, capability endpoints |
| `github.com/ailib-official/ai-lib-go/pkg/contact` | Policy (P) | `FallbackClient`, circuit-breaker policy, health snapshots |

## How it works

**Default chat path (`pkg/ailib`):** load manifest (optional) → build JSON payload → resolve endpoint/auth from manifest → HTTP via `net/http` → micro-retry (`internal/resilience`) → parse response → populate `ExecutionMetadata`.

**Without a manifest:** `WithBaseURL` + `WithAPIKey` uses OpenAI-compatible defaults (`/chat/completions`, `openai_sse` decoder). This is an escape hatch, not the primary integration mode.

**Streaming:** `ChatStream` → SSE decoder from manifest (default `openai_sse`) → `Stream.Next()` / `Event()`.

There is no public operator `Pipeline` API; chat is direct HTTP + decoder.

## Quick start (protocol-first)

```bash
go get github.com/ailib-official/ai-lib-go@v1.0.0
export OPENAI_API_KEY="your-key"
```

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ailib-official/ai-lib-go/pkg/ailib"
)

func main() {
	manifestYAML := `id: openai
protocol_version: "2.0"
endpoint:
  base_url: "https://api.openai.com/v1"
`
	client, err := ailib.NewClientBuilder().
		WithProtocolData([]byte(manifestYAML)).
		WithAPIKey(os.Getenv("OPENAI_API_KEY")).
		Build()
	if err != nil {
		panic(err)
	}
	defer client.Close()

	resp, err := client.Chat(context.Background(), []ailib.Message{
		{Role: ailib.RoleUser, Content: "Hello!"},
	}, &ailib.ChatOptions{Model: "gpt-4o"})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Choices[0].Message.Content)
	fmt.Println("provider:", resp.ExecutionMetadata.ProviderID)
}
```

Pattern source: `pkg/ailib/client_execution_metadata_test.go`.

### BaseURL-only mode (OpenAI-compatible)

```go
client, err := ailib.NewClientBuilder().
	WithBaseURL("https://api.openai.com/v1").
	WithAPIKey(os.Getenv("OPENAI_API_KEY")).
	Build()
```

Use when you do not have a local manifest checkout; behavior uses baked-in OpenAI-style paths.

### Streaming

```go
stream, err := client.ChatStream(ctx, messages, opts)
if err != nil {
	return err
}
defer stream.Close()

for stream.Next() {
	if ev := stream.Event(); ev.Delta != "" {
		fmt.Print(ev.Delta)
	}
}
return stream.Err()
```

### Policy layer: fallback client

Circuit breaker + multi-provider fallback live in **`pkg/contact`**, not `pkg/ailib`:

```go
import (
	"github.com/ailib-official/ai-lib-go/pkg/ailib"
	"github.com/ailib-official/ai-lib-go/pkg/contact"
)

primary, _ := ailib.NewClientBuilder(). /* ... */ Build()
secondary, _ := ailib.NewClientBuilder(). /* ... */ Build()

fb := contact.NewFallbackClient([]ailib.Client{primary, secondary})
resp, err := fb.Chat(ctx, messages, opts)
```

## Public API (`pkg/ailib`)

`Client` interface:

- `Chat`, `ChatStream`
- Capability-gated HTTP routes: `Embeddings`, `BatchCreate`/`BatchGet`/`BatchCancel`, `STTTranscribe`, `TTSSpeak`, `Rerank`, `MCPListTools`, `MCPCallTool`, `ComputerUse`, `Reason`, `VideoGenerate`, `VideoGet`
- `Close`

Undeclared capabilities return **E1005** (`request_too_large` name in `StandardErrorName` — also used as capability gate).

Types: `Message`, `ChatOptions`, `ChatResponse`, `StreamingEvent`, `ExecutionMetadata`, `ExecutionUsage`, standard error constants **E1001–E9999**.

## Honest capability boundaries

| Area | In the module | Not included |
|------|---------------|--------------|
| **MCP** | Manifest HTTP routes + capability gate | MCP wire-protocol client |
| **Computer Use** | HTTP route + request/response types | Action execution sandbox |
| **Multimodal** | `Message.Content` as pass-through JSON | High-level multimodal builders |
| **Retry** | E-layer micro-retry on non-stream chat | Circuit breaker (P-layer only) |
| **Circuit breaker / fallback** | `pkg/contact.FallbackClient` | Not on bare `pkg/ailib.Client` |

## API keys (BYOK chain)

1. `WithAPIKey` override
2. Manifest `endpoint.auth` env
3. `<PROVIDER_ID>_API_KEY`

## Proxies

Go does **not** auto-read `AI_PROXY_URL` / `HTTP_PROXY`. Pass a custom `*http.Client` via `WithHTTPClient` (e.g. `Transport: &http.Transport{Proxy: http.ProxyFromEnvironment}`).

Cross-runtime notes: [CROSS_RUNTIME.md](https://github.com/ailib-official/ai-protocol/blob/main/docs/CROSS_RUNTIME.md).

## Testing

```bash
go test ./...
```

Compliance (shared YAML):

```bash
COMPLIANCE_DIR=../ai-protocol/tests/compliance go test ./tests/compliance/ -v
```

## Related

- [AI-Protocol](https://github.com/ailib-official/ai-protocol)
- [ai-lib-rust](https://github.com/ailib-official/ai-lib-rust)
- [ai-lib-python](https://github.com/ailib-official/ai-lib-python)
- [ai-lib-ts](https://github.com/ailib-official/ai-lib-ts)

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
