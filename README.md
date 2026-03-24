# ai-lib-go

**Official Go Runtime for AI-Protocol** - A high-performance, idiomatic Go implementation for unified AI model interaction.

[![Go 1.21+](https://img.shields.io/badge/go-1.21+-00ADD8.svg)](https://go.dev/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green.svg)](LICENSE)

## 🎯 Design Philosophy

`ai-lib-go` is the official Go runtime implementation for the [AI-Protocol](https://github.com/ailib-official/ai-protocol) specification. It embodies the core design principle:

> **一切逻辑皆算子，一切配置皆协议** (All logic is operators, all configuration is protocol)

Unlike traditional adapter libraries that hardcode provider-specific logic, `ai-lib-go` is a **protocol-driven runtime** that executes AI-Protocol specifications. This means:

- **Zero hardcoded provider logic**: All behavior is driven by protocol manifests (YAML/JSON configurations)
- **Operator-based architecture**: Streaming uses manifest-driven decoders (openai_sse, anthropic_sse) with composable pipeline
- **Unified interface**: Developers interact with a single, consistent API regardless of the underlying provider
- **Cross-runtime consistency**: Aligned with ai-lib-rust, ai-lib-python, ai-lib-ts via shared compliance tests

## 🚀 Quick Start

### Installation

```bash
go get github.com/ailib-official/ai-lib-go
```

### Basic Usage

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ailib-official/ai-lib-go/pkg/ailib"
)

func main() {
	client, err := ailib.NewClientBuilder().
		WithProtocolData([]byte(manifestYAML)).
		WithAPIKey(os.Getenv("OPENAI_API_KEY")).
		Build()
	if err != nil {
		panic(err)
	}
	defer client.Close()

	resp, err := client.Chat(context.Background(), []ailib.Message{
		{Role: ailib.RoleUser, Content: "Hello! What's 2+2?"},
	}, &ailib.ChatOptions{Model: "gpt-4o"})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Choices[0].Message.Content)
}
```

### Streaming

```go
stream, err := client.ChatStream(ctx, messages, opts)
if err != nil {
	return err
}
defer stream.Close()

for stream.Next() {
	ev := stream.Event()
	if ev.Delta != "" {
		fmt.Print(ev.Delta)
	}
}
if err := stream.Err(); err != nil {
	return err
}
```

## ✨ Features

- **Protocol-Driven**: All behavior is driven by YAML/JSON protocol files
- **Unified Interface**: Single API for all AI providers (OpenAI, Anthropic, Gemini, DeepSeek, etc.)
- **Streaming First**: Native streaming with `Stream` interface and manifest-driven decoder (openai_sse, anthropic_sse)
- **Type Safe**: Full Go types for requests, responses, and errors
- **Production Ready**: Built-in retry, circuit breaker, and fallback executor
- **Extensible**: Easy to add new providers via protocol configuration
- **Multimodal**: Support for text, images, audio, video
- **Capability Guard**: E1005 for undeclared advanced features
- **Embeddings**: Embedding generation
- **Batch API**: BatchCreate, BatchGet, BatchCancel
- **STT/TTS**: Speech-to-Text and Text-to-Speech
- **Reranking**: Document reranking
- **MCP**: MCP tool bridge (list/call)
- **Fallback Client**: Health snapshot and circuit-breaker policy

## 🔄 V2 Protocol Alignment

`ai-lib-go` aligns with the **AI-Protocol V2** specification. V0.5.0 includes V1/V2 manifest parsing, manifest-driven streaming decoder, standard error codes, and compliance test coverage.

### Standard Error Codes (V2, ARCH-003)

All provider errors are classified into 13 standard error codes with unified retry/fallback semantics:

| Code   | Name             | Retryable | Fallbackable |
|--------|------------------|-----------|--------------|
| E1001  | invalid_request  | No        | No           |
| E1002  | authentication   | No        | Yes          |
| E1003  | permission_denied| No        | No           |
| E1004  | not_found        | No        | No           |
| E1005  | request_too_large| No        | No           |
| E2001  | rate_limited     | Yes       | Yes          |
| E2002  | quota_exhausted  | Yes       | Yes          |
| E3001  | server_error     | Yes       | Yes          |
| E3002  | overloaded       | Yes       | Yes          |
| E3003  | timeout          | Yes       | Yes          |
| E4001  | conflict         | Yes       | No           |
| E4002  | cancelled        | No        | No           |
| E9999  | unknown          | No        | No           |

Classification follows: manifest `error_classification` → provider code/type → HTTP status → `E9999`.

### Compliance Tests

Cross-runtime behavioral consistency is verified by a shared YAML-based test suite from the `ai-protocol` repository:

```bash
# Run unit tests
go test ./pkg/ailib/...

# Run compliance tests (requires ai-protocol in workspace or COMPLIANCE_DIR)
COMPLIANCE_DIR=../ai-protocol/tests/compliance go test ./tests/compliance/...
```

For details, see [CROSS_RUNTIME.md](https://github.com/ailib-official/ai-protocol/blob/main/docs/CROSS_RUNTIME.md).

### Testing with ai-protocol-mock

For integration tests without real API calls, use [ai-protocol-mock](https://github.com/ailib-official/ai-protocol-mock):

```bash
# Start mock server (from ai-protocol-mock repo)
docker-compose up -d

# Run tests with mock base URL
client, _ := ailib.NewClientBuilder().
	WithBaseURL("http://localhost:4010").
	WithAPIKey("test-key").
	Build()
```

## 📁 Package Layout

- `internal/protocol` — manifest model/loader, capability metadata, streaming decoder format
- `internal/stream` — SSE decoding (openai_sse, anthropic_sse)
- `internal/resilience` — retry and backoff utilities
- `pkg/ailib` — public SDK API (`Client`, `ClientBuilder`, capabilities, `FallbackClient`)
- `tests/compliance` — fixture-driven compliance runner

## 🗺️ Ecosystem

| Project          | Purpose                    |
|------------------|----------------------------|
| ai-protocol      | Spec, schemas, manifests   |
| ai-lib-rust      | Rust runtime               |
| ai-lib-python    | Python runtime             |
| ai-lib-ts        | TypeScript runtime         |
| **ai-lib-go**    | **Go runtime**             |
| ai-protocol-mock | Mock server for testing    |
| spiderswitch     | MCP-based model switching  |
| ailib.info       | Documentation site         |

## 📜 Release Policy

- Versioning follows semver and matrix release train
- Runtime release must align with `ai-protocol` compatibility window
- Public site sync (`ailib.info`) happens after runtime release closure

## 📄 License

MIT OR Apache-2.0
