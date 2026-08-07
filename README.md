# ai-lib-go

**Protocol runtime for [AI-Protocol](https://github.com/ailib-official/ai-protocol)** — Go reference implementation (**v1.2.0**, Go 1.21+).

[中文文档](README_CN.md)

`ai-lib-go` splits execution and policy across two public packages:

| Package | Layer | Role |
|---------|-------|------|
| `github.com/ailib-official/ai-lib-go/pkg/ailib` | Execution (E) | `Client`, manifest-aware HTTP chat, capability endpoints |
| `github.com/ailib-official/ai-lib-go/pkg/contact` | Policy (P) | `FallbackClient`, circuit-breaker policy, `HealthSnapshot` |

> **Pin:** Prefer module tag **`v1.2.0`**. CI pins `ai-protocol` **v1.2.0**. See [CHANGELOG](CHANGELOG.md).

## How it works

**Default chat path (`pkg/ailib`):** load manifest (optional) → build JSON payload → resolve endpoint/auth from manifest → HTTP via `net/http` → micro-retry (`internal/resilience`) → parse response → populate `ExecutionMetadata`.

**Without a manifest:** `WithBaseURL` + `WithAPIKey` uses OpenAI-compatible defaults (`/chat/completions`, `openai_sse` decoder). This is an escape hatch, not the primary integration mode.

**Streaming:** `ChatStream` → SSE decoder from manifest (default `openai_sse`) → `Stream.Next()` / `Event()`. After a successful stream, call `Close()` then `ExecutionMetadata()`.

There is no public operator `Pipeline` API; chat is direct HTTP + decoder.

Capabilities (embeddings, batch, STT/TTS, rerank, MCP, computer use, reasoning, video) use `protocol.EndpointFor` with **path-only** fallbacks (e.g. `/embeddings`, `/rerank`) — no silent vendor host defaults.

## Quick start (protocol-first)

```bash
go get github.com/ailib-official/ai-lib-go@v1.2.0
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
if err := stream.Err(); err != nil {
	return err
}
if meta, ok := stream.ExecutionMetadata(); ok {
	_ = meta // provider_id, model_id, usage, micro_retry_count
}
return nil
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

fb := contact.NewFallbackClient(primary, secondary)
resp, err := fb.Chat(ctx, messages, opts)
_ = fb.HealthSnapshot() // per-provider circuit state
```

`NewFallbackClientWithPolicy` accepts a custom `FallbackPolicy` (`FailureThreshold`, `CircuitOpenFor`).

## Public API (`pkg/ailib`)

**Builder:** `NewClientBuilder` → `WithProtocolPath` / `WithProtocolData` / `WithBaseURL` / `WithAPIKey` / `WithHeader` / `WithTimeout` / `WithMaxRetries` / `WithHTTPClient` → `Build()`. One of protocol data, protocol path, or base URL is required.

**`Client` interface:**

- `Chat`, `ChatStream`
- Capability-gated HTTP routes: `Embeddings`, `BatchCreate`/`BatchGet`/`BatchCancel`, `STTTranscribe`, `TTSSpeak`, `Rerank`, `MCPListTools`, `MCPCallTool`, `ComputerUse`, `Reason`, `VideoGenerate`, `VideoGet`
- `Close`

Undeclared capabilities return **E1005** (`ErrUnsupported`; `StandardErrorName` maps to `request_too_large` — also used as the capability gate).

**Types:** `Message`, `ChatOptions`, `ChatResponse`, `StreamingEvent`, `Stream`, `ExecutionMetadata`, `ExecutionUsage`, `ExecutionResult`, capability request/response structs, standard error constants **E1001–E9999**, `IsRetryableCode` / `IsFallbackableCode`.

**`pkg/streaming`:** compliance-oriented SSE / event mapping helpers (used by compliance tests; not required for normal `Client.ChatStream` usage).

## Honest capability boundaries

| Area | In the module | Not included |
|------|---------------|--------------|
| **MCP** | Manifest HTTP routes + capability gate | MCP wire-protocol client |
| **Computer Use** | HTTP route + request/response types | Action execution sandbox |
| **Multimodal** | `Message.Content` as pass-through JSON | High-level multimodal builders |
| **Retry** | E-layer micro-retry on non-stream chat | Circuit breaker (P-layer only) |
| **Circuit breaker / fallback** | `pkg/contact.FallbackClient` | Not on bare `pkg/ailib.Client` |
| **Provider identity** | `internal/protocol.Loader.LoadProvider` (main / Unreleased) | Not yet a `ClientBuilder.WithProviderID` public helper |

## Protocol manifests (loader)

`Client` loads manifests via `WithProtocolPath` / `WithProtocolData`. For id-based lookup, `internal/protocol.Loader` resolves:

1. `AI_PROTOCOL_DIR` / `AI_PROTOCOL_PATH`, or `Loader.Root`, or nearby `ai-protocol/` checkout
2. Per id: `dist/v2/providers/<id>.json` → `dist/v1/...` → source `v2`/`v1` YAML|JSON
3. **Identity / aliases (GO-ID-001):** if exact id is missing, alias → canonical via `dist/provider-identity.json` (multi-family map), e.g. `google` → `gemini`, `kimi` → `moonshot`. Parse/validation errors are never masked by alias fallthrough.

## API keys (BYOK chain)

1. `WithAPIKey` override
2. Manifest `endpoint.auth` / top-level `auth` env declarations
3. Conventional `<PROVIDER_ID>_API_KEY`

## Proxies

Go does **not** auto-read `AI_PROXY_URL` / `HTTP_PROXY`. Pass a custom `*http.Client` via `WithHTTPClient` (e.g. `Transport: &http.Transport{Proxy: http.ProxyFromEnvironment}`).

Cross-runtime notes: [CROSS_RUNTIME.md](https://github.com/ailib-official/ai-protocol/blob/main/docs/CROSS_RUNTIME.md).

## Standard error codes (V2)

| Code | Name | Retryable | Fallbackable |
|------|------|-----------|--------------|
| E1001 | `invalid_request` | No | No |
| E1002 | `authentication` | No | Yes |
| E1003 | `permission_denied` | No | No |
| E1004 | `not_found` | No | No |
| E1005 | `request_too_large` | No | No |
| E2001 | `rate_limited` | Yes | Yes |
| E2002 | `quota_exhausted` | No | Yes |
| E3001 | `server_error` | Yes | Yes |
| E3002 | `overloaded` | Yes | Yes |
| E3003 | `timeout` | Yes | Yes |
| E4001 | `conflict` | Yes | No |
| E4002 | `cancelled` | No | No |
| E9999 | `unknown` | No | No |

Classification priority: manifest `error_classification` → provider code/type → HTTP status → `E9999`.

## Testing

```bash
go test ./...
```

Compliance (shared YAML):

```bash
COMPLIANCE_DIR=../ai-protocol/tests/compliance go test ./tests/compliance/ -v
```

Mock server ([ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock)):

```bash
client, _ := ailib.NewClientBuilder().
	WithBaseURL("http://localhost:4010").
	WithAPIKey("test-key").
	Build()
```

## Related

- [AI-Protocol](https://github.com/ailib-official/ai-protocol)
- [ai-lib-rust](https://github.com/ailib-official/ai-lib-rust)
- [ai-lib-python](https://github.com/ailib-official/ai-lib-python)
- [ai-lib-ts](https://github.com/ailib-official/ai-lib-ts)

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
